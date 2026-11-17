"""Live smoke test: validate new wire clients against a real container.

Phases (each in its own event loop — MLServerContainer uses asyncio.run):
  1. REST binary tensor roundtrip (build -> send -> parse)
  2. gRPC typed contents roundtrip
  3. gRPC raw input contents roundtrip
  4. content_type variants (np/pd/str)
  5. parameters.extra framework settings (OnnxSettings auto-resolve)
"""

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bench import wire  # noqa: E402
from bench.docker_server import MLServerContainer, detect_default_image  # noqa: E402

ZOO = REPO / "models-zoo" / "onnx-mlp-small"


def _image() -> str:
    return detect_default_image("cpu", "")


async def phase_wire(settings: dict, batch: dict, failures: list) -> None:
    """Phases 1-4: run inside a container with all path settings on."""
    model = settings["name"]
    env = {
        "MLSERVER_REST_BINARY_TENSOR": "true",
        "MLSERVER_REST_WORKER_BINARY": "true",
        "MLSERVER_GRPC_RAW_INPUT_BUFFERS": "true",
        "MLSERVER_GZIP_ENABLED": "true",
    }
    with MLServerContainer(
        name="wire-smoke",
        image=_image(),
        model_repo=ZOO.parent,
        target="cpu",
        env=env,
    ) as container:
        print(f"[smoke] container: {container.name}")
        base_rest = "http://127.0.0.1:18080"
        base_grpc = "127.0.0.1:18081"

        # ---- 1. REST binary tensor ----
        print("[smoke] 1. REST binary tensor...")
        from bench.docker_server import probe_settings

        support = probe_settings(
            container.name, "rest_binary_tensor", "rest_worker_binary"
        )
        body, header_len = wire.build_binary_request(settings, batch)
        if support.get("rest_binary_tensor"):
            sender = wire.rest_binary_sender(model, body, header_len)
            try:
                ok, bytes_out, bytes_in = await sender.send(0)
                print(f"  ok={ok} bytes_out={bytes_out} bytes_in={bytes_in} OK")
            except Exception as error:
                failures.append(f"binary-tensor: {error}")
                print(f"  FAILED: {error}")
            await sender.close()
        else:
            print(
                "  SKIPPED: image lacks rest_binary_tensor (expected "
                "on odh-master; suite records SKIPPED_UNSUPPORTED_IMAGE)"
            )

        # ---- 2/3. gRPC typed + raw ----
        for label, raw_input in (("typed", False), ("raw", True)):
            print(f"[smoke] {'2' if not raw_input else '3'}. gRPC {label}...")
            if not wire.grpc_available():
                failures.append("grpc-unavailable")
                print("  SKIPPED: grpcio/protos unavailable")
                continue
            request = wire.build_grpc_request(
                settings, batch, model, raw_input=raw_input
            )
            grpc = wire.grpc_sender(model, request, base_grpc)
            try:
                response = await grpc.infer_once()
                print(f"  outputs={[o.name for o in response.outputs]} OK")
            except Exception as error:
                failures.append(f"grpc-{label}: {error}")
                print(f"  FAILED: {error}")
            await grpc.close()

        # ---- 4. content_type variants ----
        print("[smoke] 4. content_type variants (np on onnx, pd on sklearn)...")
        import aiohttp

        zoo_root = ZOO.parent
        codec_targets = [("np", settings, batch, model)]
        sk_dir = zoo_root / "sklearn-random-forest-small"
        if sk_dir.is_dir():
            sk_settings = json.loads((sk_dir / "model-settings.json").read_text())
            sk_batch = dict(np.load(sk_dir / "validation.npz"))
            codec_targets.append(("pd", sk_settings, sk_batch, sk_settings["name"]))
        for content_type, c_settings, c_batch, c_model in codec_targets:
            request = wire.build_request_content_type(c_settings, c_batch, content_type)
            url = f"{base_rest}/v2/models/{c_model}/infer"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=request,
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        payload = await resp.json()
                        if resp.status == 200 and payload.get("outputs"):
                            n = len(payload["outputs"][0].get("data", []))
                            print(
                                f"  {content_type} ({c_model}): " f"200, {n} outputs OK"
                            )
                        else:
                            detail = f"{resp.status} {str(payload)[:100]}"
                            failures.append(f"content-type-{content_type}: {detail}")
                            print(f"  {content_type}: FAILED {detail}")
            except Exception as error:
                failures.append(f"content-type-{content_type}: {error}")
                print(f"  {content_type}: FAILED {error}")


async def phase_extra(settings: dict, batch: dict, failures: list) -> None:
    """Phase 5: parameters.extra framework settings (OnnxSettings)."""
    print("[smoke] 5. parameters.extra (OnnxSettings auto-resolve)...")
    model = settings["name"]
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        target = repo / ZOO.name
        shutil.copytree(ZOO, target)
        settings_path = target / "model-settings.json"
        patched = json.loads(settings_path.read_text())
        patched.setdefault("parameters", {})["extra"] = {
            "session_options": {
                "intra_op_num_threads": 2,
                "inter_op_num_threads": 1,
            },
            "providers": ["CPUExecutionProvider"],
        }
        settings_path.write_text(json.dumps(patched, indent=2))

        with MLServerContainer(
            name="wire-smoke-extra",
            image=_image(),
            model_repo=repo,
            target="cpu",
            env={},
        ):
            url = f"http://127.0.0.1:18080/v2/models/{model}/infer"
            request = {
                "inputs": [
                    {
                        "name": "input",
                        "shape": list(batch["input"].shape),
                        "datatype": "FP32",
                        "data": batch["input"].tolist(),
                    }
                ],
                "outputs": [{"name": "output"}],
            }
            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=request) as resp:
                        payload = await resp.json()
                        if resp.status == 200:
                            print("  parameters.extra accepted (200 OK)")
                        else:
                            detail = f"{resp.status} {str(payload)[:120]}"
                            failures.append(f"extra: {detail}")
                            print(f"  FAILED: {detail}")
            except Exception as error:
                failures.append(f"extra: {error}")
                print(f"  FAILED: {error}")


def main() -> int:
    settings = json.loads((ZOO / "model-settings.json").read_text())
    batch = dict(np.load(ZOO / "validation.npz"))
    failures: list[str] = []

    asyncio.run(phase_wire(settings, batch, failures))
    asyncio.run(phase_extra(settings, batch, failures))

    print()
    if failures:
        print(f"[smoke] FAILURES ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("[smoke] ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
