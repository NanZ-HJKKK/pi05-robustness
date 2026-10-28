import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import base64
import hashlib
import io
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"))
from pi05_robustness.statistics import wilson,summarize,paired_comparison
from patch_openpi import patched_sources
from prepare_protocol import base_name
from evaluate import plan_tasks,request_seed,physical_bddl_path


def row(i,success):
    return dict(pair_id=str(i),status="complete",success=success,suite="spatial",task_name="task"+str(i),base_task="task"+str(i),
        initial_state_sha256="same",settled_sim_state_sha256="same",bddl_sha256="same",prompt="same",seed=7,replan_steps=5,
        initial_main_rgb_sha256="same",initial_wrist_rgb_sha256="same",
        inference=[],episode_wall_seconds=1,environment_steps=10)


class ProtocolTests(unittest.TestCase):
    def test_truncated_response_retries_received_prefix(self):
        from fetch_checkpoint import download_checkpoint
        data=b"123456789"
        item={"name":"checkpoints/pi05_libero/params/truncated","size":"9",
              "md5Hash":base64.b64encode(hashlib.md5(data).digest()).decode()}
        class Response(io.BytesIO):
            status=206
            def __init__(self,data,start):
                super().__init__(data);self.headers={"Content-Range":f"bytes {start}-8/9"}
        replies=[Response(b"1234",0),Response(b"56789",4)]
        def respond(req,**kwargs):
            expected="bytes=0-8" if len(replies)==2 else "bytes=4-8"
            self.assertEqual(req.get_header("Range"),expected)
            return replies.pop(0)
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(ROOT.resolve()))
            with patch("urllib.request.urlopen",side_effect=respond),patch("fetch_checkpoint.time.sleep"):
                download_checkpoint([item],directory,lambda:None)
            self.assertEqual((Path(directory)/item["name"]).read_bytes(),data)
    def test_checkpoint_resumes_partial_range_bytes(self):
        from fetch_checkpoint import download_checkpoint
        data=b"123456789"
        item={"name":"checkpoints/pi05_libero/params/resume","size":"9",
              "md5Hash":base64.b64encode(hashlib.md5(data).digest()).decode()}
        class Response(io.BytesIO):
            status=206
            headers={"Content-Range":"bytes 4-8/9"}
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(ROOT.resolve()))
            target=Path(directory)/item["name"]
            chunk_dir=target.parent/(target.name+".chunks")
            chunk_dir.mkdir(parents=True)
            (chunk_dir/"0.partial").write_bytes(b"1234")
            def respond(req,**kwargs):
                self.assertEqual(req.get_header("Range"),"bytes=4-8")
                return Response(b"56789")
            with patch("urllib.request.urlopen",side_effect=respond):
                download_checkpoint([item],directory,lambda:None)
            self.assertEqual(target.read_bytes(),data)
    def test_plus_virtual_bddl_resolution(self):
        self.assertEqual(physical_bddl_path(Path("suite/base_view_0_0_100_0_0_initstate_1.bddl")),Path("suite/base.bddl"))
        self.assertEqual(physical_bddl_path(Path("suite/base_language_3_view_0_0_100_0_0_initstate_0.bddl")),Path("suite/base_language_3.bddl"))
    def test_checkpoint_range_validation_and_promotion(self):
        from fetch_checkpoint import download_checkpoint
        data=b"official fixture bytes"*30
        item={"name":"checkpoints/pi05_libero/params/example","size":str(len(data)),
              "md5Hash":base64.b64encode(hashlib.md5(data).digest()).decode()}
        class Response(io.BytesIO):
            status=206
            headers={"Content-Range":f"bytes 0-{len(data)-1}/{len(data)}"}
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(ROOT.resolve()))
            with patch("urllib.request.urlopen",side_effect=lambda *a,**k:Response(data)) as request:
                result=download_checkpoint([item],directory,lambda:None)
                self.assertEqual((Path(directory)/item["name"]).read_bytes(),data)
                self.assertEqual(result[0]["sha256"],hashlib.sha256(data).hexdigest())
                download_checkpoint([item],directory,lambda:None)
                self.assertEqual(request.call_count,1)

    @unittest.skipUnless(importlib.util.find_spec("google_crc32c"),"CRC32C C extension not installed")
    def test_composite_gcs_object_crc32c_without_md5(self):
        from fetch_checkpoint import download_checkpoint
        data=b"123456789"
        item={"name":"checkpoints/pi05_libero/params/composite","size":"9","crc32c":"4waSgw=="}
        class Response(io.BytesIO):
            status=206
            headers={"Content-Range":"bytes 0-8/9"}
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(ROOT.resolve()))
            with patch("urllib.request.urlopen",side_effect=lambda *a,**k:Response(data)):
                result=download_checkpoint([item],directory,lambda:None)
                self.assertEqual(result[0]["crc32c"],"4waSgw==")
                self.assertTrue((Path(directory)/item["name"]).exists())

    def test_subset_names_and_counts(self):
        from collections import Counter
        data=json.loads((ROOT/"configs/eval_subset.json").read_text(encoding="utf-8"))
        self.assertEqual(set(Counter(r["category"] for r in data["tasks"]).values()),{30})
        self.assertEqual(len(data["tasks"]),210)
        self.assertEqual(len(plan_tasks("clean","all",10)),400)
        self.assertEqual(len(plan_tasks("plus","all",1)),210)

    def test_base_matching_longest_prefix(self):
        self.assertEqual(base_name("put_bowl_on_plate_view_1",["put_bowl","put_bowl_on_plate"]),"put_bowl_on_plate")
        with self.assertRaises(ValueError):base_name("other",["put_bowl"])

    def test_request_rng_pairs_independent_of_model(self):
        self.assertEqual(request_seed("pair",0,7),request_seed("pair",0,7))
        self.assertNotEqual(request_seed("pair",0,7),request_seed("pair",5,7))

    def test_all_patched_sources_compile(self):
        manifest=json.loads((ROOT/"evidence/upstream/manifest.json").read_text(encoding="utf-8"))["openpi"]
        paths=["src/openpi/models/model.py","src/openpi/training/config.py","src/openpi/policies/policy.py","scripts/train.py"]
        raw={p:(ROOT/"evidence/upstream"/manifest["files"][p]["file"]).read_text(encoding="utf-8") for p in paths}
        result=patched_sources(raw)
        self.assertIn('jax.block_until_ready(outputs["actions"])',result[paths[2]])
        for path,source in result.items():ast.parse(source,filename=path)
        # A changed context must fail rather than patching an unknown upstream.
        with self.assertRaises(ValueError):patched_sources(result)

    def test_wilson_zero_and_all_successes(self):
        self.assertAlmostEqual(wilson(0,10)[0],0)
        self.assertGreater(wilson(0,10)[1],.27)
        self.assertAlmostEqual(wilson(10,10)[1],1)
        with self.assertRaises(ValueError):wilson(0,0)

    def test_pairing_rejects_changed_physical_states(self):
        a=[row(0,False),row(1,True)]
        b=[row(0,True),row(1,True)]
        self.assertEqual(paired_comparison(a,b,bootstraps=100)["treatment_minus_control_pp"],50)
        b[0]["initial_state_sha256"]="different"
        with self.assertRaises(ValueError):paired_comparison(a,b)

    def test_pairing_rejects_changed_camera_inputs(self):
        a=[row(0,False)];b=[row(0,True)]
        b[0]["initial_wrist_rgb_sha256"]="different"
        with self.assertRaises(ValueError):paired_comparison(a,b)

    def test_rgb_exception_is_scoped_and_disclosed(self):
        a=[row(0,False),row(1,True)];b=[row(0,True),row(1,True)]
        b[0]['initial_main_rgb_sha256']='audited_other'
        exception={'status':'audited_bounded_rgb_exception','pair_id':'0','raw_hashes':['same','audited_other']}
        result=paired_comparison(a,b,bootstraps=100,rgb_exception=exception)
        self.assertFalse(result['strict_pixel_identity']);self.assertEqual(len(result['audited_rgb_exceptions']),1)
        b[1]['initial_main_rgb_sha256']='audited_other'
        with self.assertRaises(ValueError):paired_comparison(a,b,rgb_exception=exception)
        b[1]['initial_main_rgb_sha256']='same';b[0]['initial_wrist_rgb_sha256']='different'
        with self.assertRaises(ValueError):paired_comparison(a,b,rgb_exception=exception)

    def test_feedback_cycles_are_separate_from_sampling_throughput(self):
        a=row(0,True)
        a["inference"]=[dict(warmup=False,sampling_ms=50,policy_wall_ms=60,request_wall_ms=70,request_start_elapsed_seconds=t) for t in [0.,.1,.2]]
        a["control_loop_wall_seconds"]=2
        s=summarize([a])
        self.assertAlmostEqual(s["median_simulation_feedback_hz"],10)
        self.assertEqual(s["amortized_action_throughput_hz"],100)
        self.assertEqual(s["simulation_actions_per_control_second_without_warmup"],5)

    def test_heldout_is_disjoint_and_keeps_clean_plan(self):
        from evaluate_v3 import plan_tasks as plan_v3
        a=plan_tasks("plus","all",1);b=plan_v3("plus","all",1,"heldout")
        self.assertEqual(len(b),210)
        self.assertFalse({(r["suite"],r["task_name"]) for r in a}&{(r["suite"],r["task_name"]) for r in b})
        self.assertEqual(plan_v3("clean","all",10),plan_tasks("clean","all",10))
        self.assertEqual(plan_v3("plus","all",1,"diagnostic"),a)

    def test_error_and_duplicate_not_silently_success(self):
        a=row(0,False)
        self.assertEqual(summarize([a])["successes"],0)
        with self.assertRaises(ValueError):summarize([a,a])
        a["status"]="error"
        with self.assertRaises(ValueError):summarize([a])
        with self.assertRaises(ValueError):summarize([])

    def test_warmup_removed_and_throughput_units(self):
        a=row(0,True)
        a["inference"]=[dict(warmup=True,sampling_ms=1000,policy_wall_ms=1100,request_wall_ms=1200),
                        dict(warmup=False,sampling_ms=50,policy_wall_ms=60,request_wall_ms=70)]
        s=summarize([a])
        self.assertEqual(s["median_sampling_ms"],50)
        self.assertEqual(s["amortized_action_throughput_hz"],100)


@unittest.skipUnless(importlib.util.find_spec("jax"),"CPU JAX not installed")
class AugmentationTests(unittest.TestCase):
    def test_registered_camera_mixture_and_finite_geometry(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from pi05_robustness.augmentation import augment_batch
        image=jnp.arange(8*8*3,dtype=jnp.float32).reshape(8,8,3)/(8*8*3)
        batch=jnp.stack([image]*4096);key=jax.random.key(123)
        fn=jax.jit(lambda x:augment_batch(key,x,'view_wrist_drop',camera='main'))
        main=np.asarray(fn(batch));original=np.asarray(batch)
        np.testing.assert_array_equal(main,fn(batch))
        self.assertTrue(np.isfinite(main).all())
        self.assertGreaterEqual(float(main.min()),0);self.assertLessEqual(float(main.max()),1)
        clean_fraction=np.mean(np.all(main==original,axis=(1,2,3)))
        self.assertGreater(clean_fraction,.47);self.assertLess(clean_fraction,.53)
        wrist=np.asarray(jax.jit(lambda x:augment_batch(key,x,'view_wrist_drop',camera='wrist'))(batch))
        black=np.all(wrist==0,axis=(1,2,3))
        self.assertGreater(black.mean(),.125);self.assertLess(black.mean(),.175)
        np.testing.assert_array_equal(wrist[~black],original[~black])
        np.testing.assert_array_equal(augment_batch(key,batch,'view_wrist_drop',camera='unused'),batch)
        np.testing.assert_array_equal(augment_batch(key,batch,'default',camera='wrist'),batch)

    def test_identity_determinism_and_jit(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from pi05_robustness.augmentation import random_perspective,augment_batch
        image=jnp.arange(24*24*3,dtype=jnp.float32).reshape(24,24,3)/(24*24*3)
        key=jax.random.key(42)
        np.testing.assert_allclose(random_perspective(key,image,0),image,atol=2e-5)
        fn=jax.jit(lambda x:augment_batch(key,x,"view_drop"))
        batch=jnp.stack([image]*32)
        a,b=fn(batch),fn(batch)
        np.testing.assert_array_equal(a,b)
        self.assertTrue(np.isfinite(a).all())
        self.assertGreaterEqual(float(a.min()),0)
        self.assertLessEqual(float(a.max()),1)
        self.assertFalse(np.array_equal(a,batch))
        np.testing.assert_array_equal(augment_batch(key,batch,"default"),batch)

    def test_drop_probability(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from pi05_robustness.augmentation import augment_batch
        out=jax.jit(lambda key:augment_batch(key,jnp.ones((1000,8,8,3)),"view_drop"))(jax.random.key(0))
        frac=np.mean(np.all(np.asarray(out)==0,axis=(1,2,3)))
        self.assertGreater(frac,.10)
        self.assertLess(frac,.20)


if __name__=="__main__":unittest.main(verbosity=2)
