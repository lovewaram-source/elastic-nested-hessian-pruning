import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from checkpoint_token_isolation_core import RATES,load_block_weights
from module_token_isolation import select_module_masks,collect_masked_module_hessians,collect_module_isolation
from run_checkpoint_token_isolation import CheckpointCache,parse_args,reconstruct,run
from run_rate_aligned_nested import _find_layers
from experiment_common import input_groups
import test_checkpoint_token_isolation as checkpoint_tests


class ModuleTokenTests(unittest.TestCase):
    def test_budget_random_and_per_module_scores(self):
        names=("a","b")
        scores={r:{"a":torch.arange(20.).reshape(2,10),"b":-torch.arange(20.).reshape(2,10)} for r in RATES[1:]}
        masks,stats=select_module_masks(names,(2,10),"module_residual",scores=scores)
        self.assertEqual(masks[.3]["a"].flatten().nonzero().flatten().tolist(),[18,19])
        self.assertEqual(masks[.3]["b"].flatten().nonzero().flatten().tolist(),[0,1])
        random,proof=select_module_masks(names,(2,10),"module_random",seed=8)
        again,proof2=select_module_masks(names,(2,10),"module_random",seed=8)
        self.assertEqual(proof,proof2)
        for r in RATES[1:]:
            for name in names:
                self.assertEqual(int(random[r][name].sum()),2)
                self.assertTrue(torch.equal(random[r][name],again[r][name]))

    @torch.no_grad()
    def test_module_hessians_equal_direct_mixed_inputs_and_true_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            _,scratch,dense,initial,attention,positions,cache=checkpoint_tests.CheckpointIsolationTests().fixture(folder)
            paths={r:initial for r in RATES}
            expected_paths={r:initial for r in RATES}
            for index,layer in enumerate(scratch.model.layers):
                captured={}
                for r in RATES:
                    block=copy.deepcopy(dense.model.layers[index])
                    load_block_weights(block,cache.source(index,r))
                    buffers={n:[] for n in _find_layers(block)}
                    def hook_for(name):
                        def hook(_,values,output):
                            buffers[name].append(values[0].detach().clone())
                        return hook
                    hooks=[m.register_forward_hook(hook_for(n)) for n,m in _find_layers(block).items()]
                    expected_paths[r]=torch.cat([block(x.unsqueeze(0),attention_mask=attention,position_ids=positions)[0]
                                                for x in expected_paths[r]])
                    for h in hooks:
                        h.remove()
                    captured[r]={n:torch.cat(values) for n,values in buffers.items()}
                masks,_=select_module_masks(tuple(_find_layers(layer)),(2,8),"module_random",seed=index)
                hs,paths,trace=collect_masked_module_hessians(
                    layer,paths,{r:cache.source(index,r) for r in RATES},masks,attention,positions,torch.device("cpu"))
                for name in hs:
                    for r in RATES:
                        mixed=captured[.2][name].clone()
                        if r!=.2:
                            mixed[masks[r][name]]=captured[r][name][masks[r][name]]
                        x=mixed.flatten(0,1)
                        torch.testing.assert_close(hs[name][r],x.t()@x,rtol=1e-5,atol=1e-5)
                for r in RATES:
                    torch.testing.assert_close(paths[r],expected_paths[r],rtol=0,atol=0)

    @torch.no_grad()
    def test_residual_selector_and_reconstruction_preserve_shared_prefix(self):
        target20_reference = {}
        for rule in ("module_residual","module_random","module_cosine"):
            with self.subTest(rule=rule),tempfile.TemporaryDirectory() as folder:
                args,model,_,initial,attention,positions,cache=checkpoint_tests.CheckpointIsolationTests().fixture(folder)
                args.selection_rule=rule
                args.private_fraction=.1
                args.shared_fusion="anchor20"
                args.seed=0
                args.save_final_target=True
                counts,total,_=reconstruct(args,model,initial,attention,positions,cache)
                for index in range(2):
                    saved=torch.load(args.output/f"module_positions_layer_{index:03d}.pt",weights_only=True)
                    self.assertEqual(saved["statistics"]["rule"],rule)
                    prior=None
                    for r in RATES:
                        current=torch.load(cache.path("target",index,r),weights_only=True)
                        if r == .2:
                            if index not in target20_reference:
                                target20_reference[index] = {n:w.clone() for n,w in current.items()}
                            for n,w in current.items():
                                torch.testing.assert_close(w,target20_reference[index][n],rtol=0,atol=0)
                        if prior is not None:
                            for n,w in current.items():
                                end=round(w.shape[1]*(r-.1)/.5)
                                torch.testing.assert_close(w[:,:end],prior[n][:,:end],rtol=0,atol=0)
                        prior=current
                for r in RATES:
                    self.assertAlmostEqual(counts[r]/total,r)

    @unittest.skipUnless(torch.cuda.is_available(),"requires one visible CUDA GPU")
    @torch.no_grad()
    def test_full_runner_each_selector(self):
        from transformers import LlamaConfig,LlamaForCausalLM
        from experiment_common import _capture_first_layer_inputs
        for rule in ("module_residual","module_random","module_cosine"):
            with self.subTest(rule=rule),tempfile.TemporaryDirectory() as folder:
                torch.manual_seed(11)
                model=LlamaForCausalLM(LlamaConfig(vocab_size=64,hidden_size=40,intermediate_size=80,
                    num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=32,
                    _attn_implementation="sdpa",use_cache=False)).half().eval()
                model.seqlen=8
                model.hf_device_map={"":"cpu"}
                tokens=torch.randint(0,64,(1,16))
                args=parse_args(["--mode","2of4","--selection-rule",rule,"--nsamples","2","--seqlen","8",
                    "--save-final-target","--output",str(Path(folder)/"results"),"--cache",str(Path(folder)/"cache")])
                def capture(protocol,loaded,tokenizer,device):
                    return _capture_first_layer_inputs(loaded,[(tokens[:,:8],),(tokens[:,8:],)],2,8,device)
                with patch("transformers.AutoTokenizer.from_pretrained",return_value=object()), \
                     patch("run_checkpoint_token_isolation._load_model",return_value=model), \
                     patch("run_checkpoint_token_isolation._load_eval_loader",return_value=SimpleNamespace(input_ids=tokens)), \
                     patch("run_checkpoint_token_isolation._load_calibration_inputs",capture):
                    run(args)
                rows=json.loads((args.output/"results.json").read_text())
                self.assertEqual(len(rows),4)
                self.assertTrue(all(r["status"]=="completed" and r["selection_rule"]==rule for r in rows))
                self.assertEqual(list((args.cache/"target").iterdir()),[])


if __name__=="__main__":
    unittest.main()
