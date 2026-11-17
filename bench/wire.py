"""Wire-format clients: binary tensor REST, gRPC, content-type variants.

MLServer supports several request encodings that exercise different
processing paths with measurably different overheads:

- REST JSON (baseline, KServe v2 JSON dataplane)
- REST binary tensor (``rest_binary_tensor`` setting; the KServe v2
  Binary Tensor Data extension: JSON header + raw little-endian
  tensor bytes, zero-copy parse on the server)
- gRPC with typed contents (``contents.fp64_contents`` etc.)
- gRPC with raw input buffers (``raw_input_contents`` — pairs with
  the server's ``grpc_raw_input_buffers`` zero-copy path)
- content_type codec variants (np / pd / str) on the JSON path

This module builds requests in every encoding and provides async
senders compatible with ``loadgen.SendFn``.
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REST_PORT = 18080
GRPC_PORT = 18081

# ---------------------------------------------------------------------------
# Binary tensor wire format (KServe v2 Binary Tensor Data extension)
# ---------------------------------------------------------------------------

HEADER_NAME = "inference-header-content-length"

# KServe datatype -> little-endian numpy dtype (wire format mandate).
DATATYPE_MAP: dict[str, np.dtype] = {
    "BOOL": np.dtype("i1"),
    "UINT8": np.dtype("u1"),
    "UINT16": np.dtype("<u2"),
    "UINT32": np.dtype("<u4"),
    "UINT64": np.dtype("<u8"),
    "INT8": np.dtype("i1"),
    "INT16": np.dtype("<i2"),
    "INT32": np.dtype("<i4"),
    "INT64": np.dtype("<i8"),
    "FP16": np.dtype("<f2"),
    "FP32": np.dtype("<f4"),
    "FP64": np.dtype("<f8"),
}


def build_binary_request(
    model_settings: dict,
    batch: dict[str, np.ndarray],
    binary_output: bool = True,
) -> tuple[bytes, int]:
    """Build a REST binary tensor request body.

    Returns ``(body, header_length)`` where body is the JSON header
    followed by the flat little-endian tensor bytes for every input
    (in declaration order). Each binary input's header entry carries
    ``parameters.binary_data_size`` instead of inline ``data``.
    """
    inputs: list[dict[str, Any]] = []
    binary_chunks: list[bytes] = []
    for spec in model_settings["inputs"]:
        array = np.ascontiguousarray(batch[spec["name"]])
        dtype = DATATYPE_MAP[spec["datatype"]]
        payload = array.astype(dtype).tobytes(order="C")
        inputs.append(
            {
                "name": spec["name"],
                "shape": list(array.shape),
                "datatype": spec["datatype"],
                "parameters": {"binary_data_size": len(payload)},
            }
        )
        binary_chunks.append(payload)

    header: dict[str, Any] = {
        "id": "bench-binary",
        "inputs": inputs,
        "outputs": [{"name": spec["name"]} for spec in model_settings["outputs"]],
    }
    if binary_output:
        header["parameters"] = {"binary_data_output": True}

    header_bytes = json.dumps(header).encode()
    body = header_bytes + b"".join(binary_chunks)
    return body, len(header_bytes)


def parse_binary_response(body: bytes, header_length: int) -> dict[str, np.ndarray]:
    """Parse a binary tensor response into output name -> ndarray."""
    header = json.loads(body[:header_length])
    binary_region = memoryview(body)[header_length:]
    offset = 0
    outputs: dict[str, np.ndarray] = {}
    for out_spec in header.get("outputs", []):
        name = out_spec["name"]
        params = out_spec.get("parameters") or {}
        size = params.get("binary_data_size")
        if size is None:
            # Inline JSON data
            data = np.asarray(out_spec.get("data", []))
            outputs[name] = data
            continue
        datatype = out_spec["datatype"]
        shape = out_spec.get("shape", [])
        dtype = DATATYPE_MAP[datatype]
        count = int(np.prod(shape)) if shape else 1
        tensor = np.frombuffer(
            binary_region, dtype=dtype, count=count, offset=offset
        ).reshape(shape or [1])
        outputs[name] = tensor
        offset += size
    return outputs


class BinaryTensorSender:
    """REST binary tensor sender with a per-case aiohttp session.

    The session is created lazily inside the event loop that runs the
    load (aiohttp sessions are loop-bound; each case runs in a fresh
    worker-thread loop) and closed for real when the case finishes.
    """

    def __init__(
        self,
        model: str,
        body: bytes,
        header_length: int,
        host: str = "127.0.0.1",
        port: int = REST_PORT,
    ):
        self._url = f"http://{host}:{port}/v2/models/{model}/infer"
        self._body = body
        self._headers = {
            "Content-Type": "application/octet-stream",
            HEADER_NAME: str(header_length),
            # Measurement hygiene: the server's gzip_enabled defaults
            # to True — without an explicit identity request the
            # level-9 compressor silently runs on every response.
            "Accept-Encoding": "identity",
        }
        self._session: Any = None

    async def _ensure_session(self) -> None:
        if self._session is None:
            import aiohttp

            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300),
                connector=aiohttp.TCPConnector(limit=0),
            )

    async def send(self, seq: int) -> tuple[bool, int, int]:
        await self._ensure_session()
        async with self._session.post(
            self._url, data=self._body, headers=self._headers
        ) as resp:
            payload = await resp.read()
            if resp.status != 200:
                raise RuntimeError(
                    f"binary infer HTTP {resp.status}: {payload[:120]!r}"
                )
            resp_header_len = int(resp.headers.get(HEADER_NAME, "0") or "0")
            if resp_header_len:
                outputs = parse_binary_response(payload, resp_header_len)
                if not outputs:
                    raise RuntimeError("empty binary response")
            # SendFn contract: (ok, bytes_out=RESPONSE, bytes_in=REQUEST)
            # payload = response body; self._body = request body
            return True, len(payload), len(self._body)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


def rest_binary_sender(
    model: str,
    body: bytes,
    header_length: int,
    host: str = "127.0.0.1",
    port: int = REST_PORT,
) -> BinaryTensorSender:
    """Build a REST binary tensor sender (loadgen-compatible)."""
    return BinaryTensorSender(model, body, header_length, host, port)


# ---------------------------------------------------------------------------
# gRPC client (standalone pb2 loading — no mlserver import needed)
# ---------------------------------------------------------------------------

_MLSERVER_REPO_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "MLServer",
    Path(__file__).resolve().parents[3] / "MLServer",
]

_grpc_modules: dict[str, Any] | None = None


def grpc_available() -> bool:
    """True when grpcio + the MLServer proto modules can be loaded."""
    try:
        _load_grpc_modules()
        return True
    except Exception:
        return False


def _find_mlserver_repo() -> Path:
    for candidate in _MLSERVER_REPO_CANDIDATES:
        if (candidate / "mlserver" / "grpc" / "dataplane_pb2.py").is_file():
            return candidate
    raise FileNotFoundError(
        "MLServer checkout not found (needed for gRPC protos); "
        "expected a sibling ../MLServer directory"
    )


_PKG = "mlserver_bench_grpc_protos"
"""Synthetic parent package for the standalone proto modules (the
generated pb2_grpc file uses ``from . import dataplane_pb2``, so both
modules must live under a common parent package)."""


def _load_grpc_modules() -> dict[str, Any]:
    """Load dataplane_pb2 / dataplane_pb2_grpc standalone by file path."""
    global _grpc_modules
    if _grpc_modules is not None:
        return _grpc_modules

    import grpc  # noqa: F401  (fail fast if grpcio missing)

    repo = _find_mlserver_repo()
    grpc_dir = repo / "mlserver" / "grpc"

    # Parent package so the generated relative import resolves.
    if _PKG not in sys.modules:
        import types

        package = types.ModuleType(_PKG)
        package.__path__ = [str(grpc_dir)]
        sys.modules[_PKG] = package

    pb2_name = f"{_PKG}.dataplane_pb2"
    grpc_name = f"{_PKG}.dataplane_pb2_grpc"

    if pb2_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            pb2_name, str(grpc_dir / "dataplane_pb2.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[pb2_name] = module
        spec.loader.exec_module(module)

    if grpc_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            grpc_name, str(grpc_dir / "dataplane_pb2_grpc.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[grpc_name] = module
        spec.loader.exec_module(module)

    _grpc_modules = {
        "pb2": sys.modules[pb2_name],
        "pb2_grpc": sys.modules[grpc_name],
    }
    return _grpc_modules


# KServe datatype -> ModelInferRequest contents field name.
_DATATYPE_TO_FIELD = {
    "BOOL": "bool_contents",
    "UINT8": "uint_contents",
    "UINT16": "uint_contents",
    "UINT32": "uint_contents",
    "UINT64": "uint64_contents",
    "INT8": "int_contents",
    "INT16": "int_contents",
    "INT32": "int_contents",
    "INT64": "int64_contents",
    "FP16": "fp32_contents",  # promoted: protobuf has no fp16 field
    "FP32": "fp32_contents",
    "FP64": "fp64_contents",
}


def build_grpc_request(
    model_settings: dict,
    batch: dict[str, np.ndarray],
    model_name: str,
    raw_input: bool = False,
):
    """Build a ModelInferRequest.

    ``raw_input=True`` populates ``raw_input_contents`` (bytes per
    input, little-endian flat) — the zero-copy server path gated by
    ``grpc_raw_input_buffers``. Otherwise typed ``contents`` fields
    are populated from the numpy batch.
    """
    modules = _load_grpc_modules()
    pb2 = modules["pb2"]

    request = pb2.ModelInferRequest()
    request.model_name = model_name
    request.id = "bench-grpc"

    raw_chunks: list[bytes] = []
    for spec in model_settings["inputs"]:
        array = np.ascontiguousarray(batch[spec["name"]])
        inp = request.InferInputTensor()
        inp.name = spec["name"]
        inp.datatype = spec["datatype"]
        inp.shape.extend(array.shape)
        if raw_input:
            dtype = DATATYPE_MAP[spec["datatype"]]
            raw_chunks.append(array.astype(dtype).tobytes(order="C"))
        else:
            field = _DATATYPE_TO_FIELD[spec["datatype"]]
            if field == "fp32_contents":
                contents = array.astype(np.float32).reshape(-1).tolist()
            elif field == "fp64_contents":
                contents = array.astype(np.float64).reshape(-1).tolist()
            elif field in ("int_contents", "int64_contents"):
                contents = array.astype(np.int64).reshape(-1).tolist()
            elif field in ("uint_contents", "uint64_contents"):
                contents = array.astype(np.uint64).reshape(-1).tolist()
            else:
                contents = array.astype(np.float64).reshape(-1).tolist()
            getattr(inp.contents, field).extend(contents)
        request.inputs.append(inp)

    if raw_input:
        request.raw_input_contents.extend(raw_chunks)

    for spec in model_settings["outputs"]:
        out = request.InferRequestedOutputTensor()
        out.name = spec["name"]
        request.outputs.append(out)

    return request


class GrpcSender:
    """Reusable gRPC sender (one channel per sender, async).

    Follows loadgen.SendFn: ``await sender(seq) -> (ok, bytes_out,
    bytes_in)``.
    """

    def __init__(
        self,
        model_name: str,
        request: Any,
        target: str = f"127.0.0.1:{GRPC_PORT}",
        timeout: float = 30.0,
    ):
        import grpc

        modules = _load_grpc_modules()
        self._pb2_grpc = modules["pb2_grpc"]
        self._request = request
        self._model_name = model_name
        self._timeout = timeout
        self._channel = grpc.aio.insecure_channel(target)
        self._stub = self._pb2_grpc.GRPCInferenceServiceStub(self._channel)
        self._request_bytes = request.SerializeToString()

    async def send(self, seq: int) -> tuple[bool, int, int]:
        response = await self._stub.ModelInfer(self._request, timeout=self._timeout)
        # bytes_out = actual response bytes (raw chunks summed, or the
        # serialized size for typed outputs) — never a chunk count.
        if response.raw_output_contents:
            bytes_out = sum(len(chunk) for chunk in response.raw_output_contents)
        else:
            bytes_out = response.ByteSize()
        return True, bytes_out, len(self._request_bytes)

    async def infer_once(self) -> Any:
        """Single inference — for parity checks."""
        return await self._stub.ModelInfer(self._request, timeout=self._timeout)

    async def close(self) -> None:
        await self._channel.close()


def grpc_sender(
    model_name: str,
    request: Any,
    target: str = f"127.0.0.1:{GRPC_PORT}",
):
    """Build a GrpcSender wrapped for suites (close via .close())."""
    return GrpcSender(model_name, request, target)


# ---------------------------------------------------------------------------
# content_type codec variants (JSON path)
# ---------------------------------------------------------------------------


def build_request_content_type(
    model_settings: dict,
    batch: dict[str, np.ndarray],
    content_type: str,
) -> dict:
    """Build a JSON request whose inputs declare a specific content_type.

    ``np`` is the default (JSON lists). ``pd`` sends the pandas
    dataframe records encoding. ``str`` sends a JSON string payload.
    The server's codec registry decodes per the declared type.
    """
    request = {
        "id": f"bench-{content_type}",
        "inputs": [],
        "outputs": [{"name": spec["name"]} for spec in model_settings["outputs"]],
    }
    for spec in model_settings["inputs"]:
        array = batch[spec["name"]]
        entry: dict[str, Any] = {
            "name": spec["name"],
            "shape": list(array.shape),
            "datatype": spec["datatype"],
        }
        if content_type == "np":
            entry["data"] = array.tolist()
        elif content_type == "pd":
            # PandasCodec builds pd.Series(payload, dtype=...): a flat
            # list of values, same shape as np — the codec path (and
            # its pandas overhead) is the difference.
            entry["data"] = array.reshape(-1).tolist()
            entry["parameters"] = {"content_type": "pd"}
        elif content_type == "str":
            # StringCodec: JSON-encoded string payload. Only valid for
            # BYTES/string model inputs — numeric runtimes reject it.
            entry["data"] = json.dumps(array.tolist())
            entry["parameters"] = {"content_type": "str"}
        else:
            raise ValueError(f"unsupported content_type {content_type!r}")
        request["inputs"].append(entry)
    return request
