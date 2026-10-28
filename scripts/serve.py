"""Run from the patched openpi environment: uv run /path/to/serve.py ..."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    parser.add_argument("--model-id", choices=["Ref", "B1", "A1"], required=True)
    parser.add_argument("--num-steps", type=int, choices=[3,10], default=10)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--metadata-out", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path)
    args = parser.parse_args()
    if args.metadata_out.exists():
        raise SystemExit("Use a new metadata file for each server process")
    if os.environ.get("OPENPI_AUG", "default") != "default":
        raise SystemExit("Unset training augmentation before serving")
    receipt = args.checkpoint_manifest
    if receipt is None and args.model_id == "Ref":
        receipt = Path(__file__).resolve().parents[1]/"runs/setup_probe/checkpoint_verified.json"
    if receipt is None or not receipt.is_file():
        raise SystemExit("A verified checkpoint file manifest is required before serving")
    import jax
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config
    import openpi.models.model as model_module
    import openpi.policies.policy as policy_module
    source = Path(policy_module.__file__).read_text(encoding="utf-8")
    if 'jax.block_until_ready(outputs["actions"])' not in source:
        raise SystemExit("Apply the synchronized latency patch before serving")
    train_config = config.get_config(args.config)
    if args.config == "pi05_libero_lora_ft":
        from local_training_config import localize
        train_config = localize(train_config)
    policy = policy_config.create_trained_policy(train_config, args.checkpoint,
                                               sample_kwargs={"num_steps": args.num_steps})
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"],
                         text=True, capture_output=True, timeout=15).stdout.strip()
    metadata = {"model_id": args.model_id, "config": args.config, "checkpoint": args.checkpoint,
                "num_steps": args.num_steps, "server_run_id": uuid.uuid4().hex,
                "gpu": gpu, "jax_devices": [str(d) for d in jax.devices()],
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "latency_synchronized": True,
                "source_sha256": {str(Path(m.__file__).name): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                                  for m in [model_module, policy_module, config]},
                "warmup_requests": 5}
    metadata["training_config_uses_verified_local_assets"] = args.config == "pi05_libero_lora_ft"
    metadata["xla_flags"] = os.environ.get("XLA_FLAGS", "")
    if cache_file := os.environ.get("OPENPI_INFERENCE_AUTOTUNE_FILE"):
        metadata["inference_autotune_cache_sha256"] = hashlib.sha256(Path(cache_file).read_bytes()).hexdigest()
    metadata["checkpoint_receipt_sha256"] = hashlib.sha256(receipt.read_bytes()).hexdigest()
    metadata["server_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata["packages"] = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.write_text(json.dumps(metadata,indent=2) + "\n",encoding="utf-8")

    class MeasuredPolicy:
        def __init__(self):
            self.requests = 0

        @property
        def metadata(self):
            return metadata

        def infer(self, obs):
            obs = dict(obs)
            request_seed = int(obs.pop("__request_seed"))
            # Pinned JAX Policy has no episode reset API. Set its PRNG explicitly
            # from the client's stable episode/step identifier before sampling.
            policy._rng = jax.random.key(request_seed)
            started = time.perf_counter()
            result = policy.infer(obs)
            result["timing"] = {"policy_wall_ms": (time.perf_counter()-started)*1000,
                "sampling_ms": float(result["policy_timing"]["infer_ms"]),
                "warmup": self.requests < 5, "request_index": self.requests,
                "server_run_id": metadata["server_run_id"]}
            self.requests += 1
            return result

    websocket_policy_server.WebsocketPolicyServer(policy=MeasuredPolicy(), host=args.host,
                                                  port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
