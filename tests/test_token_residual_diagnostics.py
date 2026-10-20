import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from checkpoint_token_isolation_core import RATES,load_block_weights
from run_rate_aligned_nested import _find_layers
from token_residual_diagnostics import compare_selectors,forward_diagnostic_layer,midranks,top_ids
import test_checkpoint_token_isolation as checkpoint_tests


class ResidualDiagnosticTests(unittest.TestCase):
    def test_ranks_ties_budget_random_and_degenerate(self):
        values=torch.tensor([3.,1.,1.,4.])
        torch.testing.assert_close(midranks(values),torch.tensor([2.,.5,.5,3.],dtype=torch.float64))
        report=compare_selectors(values,values,fraction=.5,seed=3)
        self.assertAlmostEqual(report["spearman"],1.)
        self.assertEqual(report["top_overlap_fraction"],1.)
        self.assertEqual(report["selected_positions"],2)
        self.assertAlmostEqual(report["coverage_residual"],7/9)
        self.assertEqual(report,compare_selectors(values,values,fraction=.5,seed=3))
        zero=compare_selectors(torch.zeros(10),torch.ones(10))
        self.assertIsNone(zero["coverage_residual"])
        self.assertIsNone(zero["spearman"])
        self.assertEqual(top_ids(torch.ones(20)).tolist(),[0,1])
        with self.assertRaises(ValueError):
            compare_selectors(torch.tensor([float("nan")]),torch.ones(1))

    @torch.no_grad()
    def test_true_paths_and_scores_match_explicit_projection(self):
        with tempfile.TemporaryDirectory() as folder:
            _,scratch,dense,initial,attention,positions,cache=checkpoint_tests.CheckpointIsolationTests().fixture(folder)
            paths={r:initial for r in RATES}
            expected_paths={r:initial for r in RATES}
            for index,layer in enumerate(scratch.model.layers):
                captured={}
                for rate in RATES:
                    block=copy.deepcopy(dense.model.layers[index])
                    load_block_weights(block,cache.source(index,rate))
                    buffers={n:[] for n in _find_layers(block)}
                    def hook_for(name):
                        def hook(_,inputs,output):
                            buffers[name].append(inputs[0].detach().clone())
                        return hook
                    handles=[m.register_forward_hook(hook_for(n)) for n,m in _find_layers(block).items()]
                    expected_paths[rate]=torch.cat([block(x.unsqueeze(0),attention_mask=attention,
                                                        position_ids=positions)[0] for x in expected_paths[rate]])
                    for h in handles:
                        h.remove()
                    captured[rate]={n:torch.cat(v) for n,v in buffers.items()}
                dense_state=torch.load(cache.source(index,None),weights_only=True)
                paths,scores,cosine,trace=forward_diagnostic_layer(
                    layer,paths,{r:cache.source(index,r) for r in RATES},dense_state,
                    attention,positions,torch.device("cpu"),chunk_size=3)
                for rate in RATES:
                    torch.testing.assert_close(paths[rate],expected_paths[rate],rtol=0,atol=0)
                    self.assertEqual(trace["forward_counts"][str(rate)],2)
                    if rate==.2:
                        continue
                    sparse=torch.load(cache.source(index,rate),weights_only=True)
                    for name in scores[rate]:
                        delta=captured[rate][name]-captured[.2][name]
                        expected=F.linear(delta,dense_state[name]-sparse[name]).square().sum(-1,dtype=torch.float64)
                        torch.testing.assert_close(scores[rate][name],expected,rtol=1e-4,atol=1e-9)
                self.assertEqual(cosine.shape,(2,8))
                if index==0:
                    for rate in RATES[1:]:
                        self.assertEqual(float(scores[rate]["self_attn.q_proj"].sum()),0.)

    @unittest.skipUnless(torch.cuda.is_available(),"requires one visible GPU")
    @torch.no_grad()
    def test_complete_diagnostic_runner(self):
        from transformers import LlamaConfig,LlamaForCausalLM
        from experiment_common import _capture_first_layer_inputs
        from run_token_residual_diagnostics import parse_args,run
        with tempfile.TemporaryDirectory() as folder:
            torch.manual_seed(2)
            model=LlamaForCausalLM(LlamaConfig(vocab_size=64,hidden_size=40,intermediate_size=80,
                 num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=32,
                 _attn_implementation="sdpa",use_cache=False)).half().eval()
            model.hf_device_map={"":"cpu"}
            model.seqlen=8
            tokens=torch.randint(0,64,(1,16))
            args=parse_args(["--output",str(Path(folder)/"results"),"--cache",str(Path(folder)/"cache"),
                             "--layers","0","1","--nsamples","2","--seqlen","8","--chunk-size","3"])
            def capture(protocol,loaded,tokenizer,device):
                return _capture_first_layer_inputs(loaded,[(tokens[:,:8],),(tokens[:,8:],)],2,8,device)
            with patch("transformers.AutoTokenizer.from_pretrained",return_value=object()), \
                 patch("run_token_residual_diagnostics._load_model",return_value=model), \
                 patch("run_token_residual_diagnostics._load_calibration_inputs",capture):
                run(args)
            completion=json.loads((args.output/"completion.json").read_text())
            self.assertEqual(completion["status"],"completed")
            self.assertEqual(completion["record_count"],42)
            self.assertFalse(completion["ppl_evaluated"])
            self.assertEqual(list((args.cache/"target").iterdir()),[])
            scores=torch.load(args.output/"scores_layer_001.pt",weights_only=True)
            self.assertEqual(len(scores["cosine_top_ids"]),2)


if __name__=="__main__":
    unittest.main()
