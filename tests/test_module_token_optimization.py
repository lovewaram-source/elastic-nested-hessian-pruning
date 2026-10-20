import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from checkpoint_token_isolation_core import RATES
from module_token_isolation import collect_module_isolation,collect_masked_module_hessians,select_module_masks
from run_rate_aligned_nested import _find_layers
from run_checkpoint_token_isolation import reconstruct
import run_checkpoint_token_isolation as runner
import test_checkpoint_token_isolation as fixture_tests


class ModuleOptimizationTests(unittest.TestCase):
    def test_takeover_gate_rejects_numerical_changes(self):
        from experiment_common import benchmark_gate
        rows=[dict(implementation="direct"),dict(implementation="cached_direct",
            hessian_errors=[dict(max_abs=0.)],target_errors=[dict(mask_disagreements=0,max_abs=0.)])]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"benchmark.json"
            path.write_text(json.dumps(rows))
            self.assertTrue(benchmark_gate(path)[0])
            rows[1]["target_errors"][0]["mask_disagreements"]=1
            path.write_text(json.dumps(rows))
            self.assertFalse(benchmark_gate(path)[0])
            rows[1]["target_errors"][0]["mask_disagreements"]=0
            rows[1]["hessian_errors"][0]["max_abs"]=1e-8
            path.write_text(json.dumps(rows))
            self.assertFalse(benchmark_gate(path)[0])

    @torch.no_grad()
    def test_cached_direct_is_bitwise_equivalent(self):
        devices=("cpu","cuda:0") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            for rule in ("module_random","module_cosine"):
                with self.subTest(device=device,rule=rule),tempfile.TemporaryDirectory() as folder:
                    _,scratch,dense,initial,attention,positions,cache=fixture_tests.CheckpointIsolationTests().fixture(folder)
                    slow_paths={r:initial for r in RATES}
                    fast_paths={r:initial for r in RATES}
                    for index in range(2):
                        masks,_=select_module_masks(tuple(_find_layers(scratch.model.layers[index])),(2,8),rule,
                            cosine=torch.arange(16.).reshape(2,8),seed=index)
                        values=[]
                        for mode,paths in (("direct",slow_paths),("cached_direct",fast_paths)):
                            hs,out,_=collect_masked_module_hessians(copy.deepcopy(dense.model.layers[index]).to(device),
                                paths,{r:cache.source(index,r) for r in RATES},masks,attention,positions,
                                torch.device(device),implementation=mode)
                            values.append((hs,out))
                        slow_paths,fast_paths=values[0][1],values[1][1]
                        for n in values[0][0]:
                            for r in RATES:
                                torch.testing.assert_close(values[0][0][n][r],values[1][0][n][r],rtol=0,atol=0)
                        for r in RATES:
                            torch.testing.assert_close(slow_paths[r],fast_paths[r],rtol=0,atol=0)

    @torch.no_grad()
    def test_direct_and_delta_hessians_outputs_and_masks(self):
        devices = ("cpu","cuda:0") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            with self.subTest(device=device),tempfile.TemporaryDirectory() as folder:
                _,scratch,dense,initial,attention,positions,cache=fixture_tests.CheckpointIsolationTests().fixture(folder)
                direct_paths={r:initial for r in RATES}
                delta_paths={r:initial for r in RATES}
                for index in range(2):
                    sources={r:cache.source(index,r) for r in RATES}
                    masks,_=select_module_masks(tuple(_find_layers(scratch.model.layers[index])),(2,8),"module_random",seed=index)
                    original={r:x.clone() for r,x in direct_paths.items()}
                    hd,direct_paths,_=collect_masked_module_hessians(
                        copy.deepcopy(dense.model.layers[index]).to(device),direct_paths,sources,masks,
                        attention,positions,torch.device(device),implementation="direct")
                    hf,delta_paths,trace=collect_masked_module_hessians(
                        copy.deepcopy(dense.model.layers[index]).to(device),delta_paths,sources,masks,
                        attention,positions,torch.device(device),implementation="delta")
                    for name in hd:
                        for r in RATES:
                            torch.testing.assert_close(hf[name][r],hd[name][r],rtol=2e-5,atol=2e-5)
                            if r==.2:
                                torch.testing.assert_close(hf[name][r],hd[name][r],rtol=0,atol=0)
                    for r in RATES:
                        torch.testing.assert_close(direct_paths[r],delta_paths[r],rtol=0,atol=0)
                    self.assertEqual(trace["hessian_implementation"],"delta")
                    del hd,hf

    @torch.no_grad()
    def test_selector_unchanged_and_H20_plans_reused(self):
        for rule in ("module_residual","module_random","module_cosine"):
            with self.subTest(rule=rule),tempfile.TemporaryDirectory() as folder:
                args,model,dense,initial,attention,positions,cache=fixture_tests.CheckpointIsolationTests().fixture(folder)
                options=dict(selection_rule=rule,private_fraction=.1,shared_fusion="anchor20",seed=0,save_final_target=True)
                for k,v in options.items():
                    setattr(args,k,v)
                args.module_hessian_implementation="delta"
                with patch("run_checkpoint_token_isolation.prepare_device_plan",wraps=runner.prepare_device_plan) as factor:
                    counts,total,_=reconstruct(args,model,initial,attention,positions,cache)
                    # Seven modules x three private H + four distinct H20, per layer.
                    self.assertEqual(factor.call_count,2*(7*3+4))
                for r in RATES:
                    self.assertAlmostEqual(counts[r]/total,r)
                other_args=copy.copy(args)
                other_args.output=Path(folder)/"direct-results"
                other_args.output.mkdir()
                other_args.module_hessian_implementation="direct"
                other_cache=runner.CheckpointCache(Path(folder)/"direct-cache")
                # Reuse read-only reference files without copying or overwriting.
                other_cache.reference=cache.reference
                reconstruct(other_args,dense,initial,attention,positions,other_cache)
                for i in range(2):
                    a=torch.load(args.output/f"module_positions_layer_{i:03d}.pt",weights_only=True)
                    b=torch.load(other_args.output/f"module_positions_layer_{i:03d}.pt",weights_only=True)
                    self.assertEqual(a["statistics"],b["statistics"])
                    for r in RATES:
                        fast=torch.load(cache.path("target",i,r),weights_only=True)
                        slow=torch.load(other_cache.path("target",i,r),weights_only=True)
                        for name in fast:
                            self.assertTrue(torch.equal(fast[name].eq(0),slow[name].eq(0)))
                            torch.testing.assert_close(fast[name],slow[name],rtol=1e-4,atol=1e-5)


if __name__=="__main__":
    unittest.main()
