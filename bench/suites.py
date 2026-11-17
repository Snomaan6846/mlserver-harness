"""The seven benchmark suites covering the MLServer surface area.

Each suite receives a SuiteContext (container factory, model zoo, load
profile, sender factory) and returns a list of case results. Restarts are
owned by the suites themselves so every matrix cell runs on a clean,
correctly-configured server. The parity gate runs before any measurement.
"""

import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np

from . import loadgen, telemetry, wire
from .asyncutils import run_coroutine_blocking
from .config import RunConfig
from .docker_server import MLServerContainer, REST_PORT
from .parity import ParityError, verify_model

Case = dict[str, Any]


class _RegisteredContainer(MLServerContainer):
    """Container that registers itself with the SuiteContext while active,
    so the parity gate computes references inside the running image."""

    _ctx: "SuiteContext"

    def __enter__(self) -> "_RegisteredContainer":
        self._ctx._active_server = self
        try:
            return super().__enter__()
        except BaseException:
            self._ctx._active_server = None
            raise

    def __exit__(self, *exc: Any) -> None:
        # Keep the server's logs when an exception is unwinding
        # through this context: stop() removes the container, and with
        # it the only evidence of WHY a suite failed (the reviewer's
        # SUITE_FAILED connection-reset case had its cause destroyed
        # — "the container's logs were already removed"). Normal
        # exits don't dump: suites restart containers constantly and
        # only failures need the evidence.
        if exc[0] is not None:
            try:
                log_dir = self._ctx.output_dir / "container-logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / f"{self.name}.log").write_text(self.logs())
            except Exception:
                pass  # never mask the propagating failure
        try:
            super().__exit__(*exc)
        finally:
            self._ctx._active_server = None


class SuiteContext:
    def __init__(
        self,
        config: RunConfig,
        image: str,
        zoo_root: Path,
        output_dir: Path,
        run_index: int = 0,
        round_index: int = 0,
        suite_name: str = "",
        run_id: str = "",
    ):
        self.config = config
        self.image = image
        self.zoo_root = zoo_root
        self.output_dir = output_dir
        self.run_index = run_index
        self.round_index = round_index
        # Round index drives arm interleaving: paired on/off axes
        # (middleware states, processing variants) reverse their order
        # on odd rounds, so slow drift affects both arms equally
        # across a multi-round run. (A stray `self.round_index = 0`
        # here used to wipe the caller's round — arm_order_flip never
        # flipped and every round's containers looked identical.)
        self.suite_name = suite_name
        # Unique run identity for container names: explicit run_id
        # (runner or paired driver) or a random fallback so names stay
        # unique even for ad-hoc contexts (tests, notebooks).
        from .config import generate_run_id, sanitize_run_id

        self.run_id = sanitize_run_id(run_id) or generate_run_id("ctx")
        # Graded-control knob (paired driver): cap every container this
        # context starts with a CFS quota. Applied via env so the inner
        # bench.cli run of the candidate side degrades deterministically
        # WITHOUT loading the host — host CPU hogs (the previous
        # mechanism) triggered connection resets on the contended side
        # (F-06: 2/2 graded attempts aborted; un-hogged runs clean).
        self._paired_cpu_limit: float | None = None
        raw_limit = os.environ.get("PAIRED_CPU_LIMIT", "")
        if raw_limit:
            try:
                self._paired_cpu_limit = float(raw_limit)
            except ValueError:
                self._paired_cpu_limit = None
        self._counter = 0
        self._active_server: _RegisteredContainer | None = None

    @property
    def arm_order_flip(self) -> bool:
        """True on odd rounds — suites with paired arms should
        reverse their measurement order."""
        return self.round_index % 2 == 1

    def next_name(self, label: str = "") -> str:
        self._counter += 1
        # Every container name carries the run identity, the suite
        # label and the round, and a per-context counter:
        #   mlserver-bench-{run_id}-{suite}-r{round}-{n}
        # The old scheme (mlserver-bench-srv-{n}) reset the counter
        # for every suite and round, so names were reused across the
        # matrix — a leaked container collided with the next suite's
        # name, and `docker ps` gave no clue which run/side/suite a
        # container belonged to.
        from .docker_server import CONTAINER_PREFIX

        tag = label or self.suite_name or "srv"
        return (
            f"{CONTAINER_PREFIX}-{self.run_id}-{tag}-"
            f"r{self.round_index}-{self._counter}"
        )

    def server(
        self,
        model_repo: Path,
        env: dict[str, str] | None = None,
        parallel_workers: int | None = None,
        image: str | None = None,
        target: str | None = None,
        cpu_limit: float | None = None,
    ) -> MLServerContainer:
        """Start a server, optionally overriding the run's image/target.

        The hardware suite uses this to run the CPU arm on the CPU
        image (target=cpu, no --gpus, no GPU guard) and the GPU arm on
        the run's image (target=gpu, --gpus all, anti-silent-fallback
        guard enforced) — the production deployment matrix.

        cpu_limit: CFS quota for the container (graded control). Falls
        back to the PAIRED_CPU_LIMIT env when not given explicitly.
        """
        container = _RegisteredContainer(
            name=self.next_name(),
            image=image or self.image,
            model_repo=model_repo,
            target=target or self.config.target,
            env=env or {},
            parallel_workers=parallel_workers,
            cpu_limit=cpu_limit if cpu_limit is not None else self._paired_cpu_limit,
        )
        container._ctx = self
        return container

    async def parity(self, model_repo: Path, model_dir: Path) -> None:
        """Correctness gate. When a server from this context is active,
        the native reference is computed inside the image (exact library
        versions); otherwise the host libraries are used."""
        problems = await self.parity_problems(model_repo, model_dir)
        if problems:
            raise ParityError(
                f"pre-benchmark correctness gate failed for "
                f"{model_dir.name}: {problems}"
            )

    async def parity_problems(self, model_repo: Path, model_dir: Path) -> list[str]:
        """Parity gate as a FINDING COLLECTOR (non-raising).

        For per-config gates inside suites, a configuration that
        returns wrong results is a recorded finding — the suite notes
        PARITY_FAILED for that cell and keeps measuring the others.
        (The run-start framework gate stays fatal: systemic breakage
        invalidates everything.)"""
        container = self._active_server
        if container is not None:
            try:
                return await verify_model(
                    model_dir,
                    container=container.name,
                    model_dir_inside=f"/mnt/models/{model_dir.name}",
                )
            except ParityError as error:
                return [str(error)[:500]]
        try:
            return await verify_model(model_dir)
        except ParityError as error:
            return [str(error)[:500]]


# ---------------------------------------------------------------------------
# Sender construction (REST JSON / REST binary / gRPC)
# ---------------------------------------------------------------------------


def build_request(model_settings: dict, batch: dict[str, np.ndarray]) -> dict:
    inputs = []
    for spec in model_settings["inputs"]:
        inputs.append(
            {
                "name": spec["name"],
                "shape": list(batch[spec["name"]].shape),
                "datatype": spec["datatype"],
                "data": batch[spec["name"]].tolist(),
            }
        )
    return {
        "id": "bench",
        "inputs": inputs,
        "outputs": [{"name": spec["name"]} for spec in model_settings["outputs"]],
    }


def tile_batch(
    batch: dict[str, np.ndarray], request_batch: int
) -> dict[str, np.ndarray]:
    """Tile a validation batch to an exact request batch size.

    The first axis (batch dim) is tiled/truncated to request_batch —
    model inputs declare dynamic batch (-1), so any size is valid.
    Used by the hardware suite: the CPU-vs-GPU crossover point is a
    function of request batch size.
    """
    tiled: dict[str, np.ndarray] = {}
    for name, array in batch.items():
        if array.ndim >= 1 and array.shape[0] > 0:
            reps = -(-request_batch // array.shape[0])  # ceil
            tiled[name] = np.tile(array, (reps, *([1] * (array.ndim - 1))))[
                :request_batch
            ]
        else:
            tiled[name] = array
    return tiled


class RestJsonSender:
    """REST JSON sender with a per-case aiohttp session.

    The session is created lazily inside the event loop that runs the
    load (sessions are loop-bound) and closed when the case finishes.
    """

    def __init__(
        self,
        model: str,
        request: dict,
        port: int = REST_PORT,
        accept_encoding: str = "identity",
        track_encoding: bool = False,
    ):
        self._url = f"http://127.0.0.1:{port}/v2/models/{model}/infer"
        self._payload = json.dumps(request)
        self._accept_encoding = accept_encoding
        # Positive control for the gzip axis: tally whether responses
        # were actually compressed (a silent no-op would otherwise
        # measure nothing while claiming to measure compression).
        self._track_encoding = track_encoding
        self.compressed_responses = 0
        self.uncompressed_responses = 0
        self._session: Any = None

    async def _ensure_session(self) -> None:
        if self._session is None:
            import aiohttp

            # connector limit=0 (unlimited): aiohttp's default of 100
            # caps concurrency-ladder rungs above C100. Measurement
            # hygiene: Accept-Encoding identity — gzip_enabled defaults
            # to True server-side and would otherwise silently compress
            # every response (the middleware suite crosses gzip
            # explicitly with a gzip-accepting sender).
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300),
                connector=aiohttp.TCPConnector(limit=0),
                headers={"Accept-Encoding": self._accept_encoding},
            )

    async def send(self, seq: int) -> tuple[bool, int, int]:
        await self._ensure_session()
        async with self._session.post(
            self._url,
            data=self._payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            body = await response.read()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {body[:120]}")
            if self._track_encoding:
                if response.headers.get("Content-Encoding", "").lower() in (
                    "gzip",
                    "deflate",
                    "br",
                ):
                    self.compressed_responses += 1
                else:
                    self.uncompressed_responses += 1
            # SendFn contract: (ok, bytes_out=RESPONSE, bytes_in=REQUEST)
            return True, len(body), len(self._payload)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


def rest_sender(model: str, request: dict) -> RestJsonSender:
    """Build a REST JSON sender (sender object protocol)."""
    return RestJsonSender(model, request)


def run_closed(
    ctx: SuiteContext,
    model: str,
    request: dict,
    concurrency: int,
    recorder: telemetry.TelemetryRecorder | None = None,
    make_sender: Any = None,
    error_threshold: float | None = None,
) -> loadgen.LoadResult:
    """Run a closed-loop load test.

    ``make_sender``: optional async factory returning an object with
    ``.send(seq)`` and ``.close()`` — used for non-REST senders
    (binary tensor, gRPC). Defaults to the REST JSON sender.

    Loop-aware: safe to call from sync code or from inside a running
    event loop (the suite coroutines) — the load runs in a dedicated
    worker-thread loop with loop-bound resources (sessions, channels)
    created and closed within the case.
    """
    from .asyncutils import run_coroutine_blocking

    profile = ctx.config.profile()

    # Client-CPU guard: the single-process asyncio generator becomes
    # the bottleneck near ~100% of one core — every measured number is
    # then a GENERATOR bound, not a server result. (This guard would
    # have caught the unique-stream client artifact automatically.)
    import psutil

    proc = psutil.Process()
    proc.cpu_percent(interval=None)  # prime the counter

    async def go() -> loadgen.LoadResult:
        if make_sender is not None:
            sender_obj = await make_sender()
        else:
            sender_obj = RestJsonSender(model, request)
        sender = sender_obj.send
        if recorder:
            recorder.start()
        try:
            return await loadgen.closed_loop(
                concurrency,
                profile["window_seconds"],
                profile["warmup_seconds"],
                sender,
                error_threshold=(
                    error_threshold if error_threshold is not None else 0.001
                ),
            )
        finally:
            if recorder:
                await recorder.stop()
            await sender_obj.close()

    result = run_coroutine_blocking(go())
    # Client-CPU guard: annotate the result (summary() picks it up);
    # near ~100% of one core, the single-loop generator is the
    # bottleneck and every number is a generator bound.
    result.client_cpu_percent = round(  # type: ignore[attr-defined]
        proc.cpu_percent(interval=None), 1
    )
    if result.client_cpu_percent > 85:
        result.generator_bound = True  # type: ignore[attr-defined]
    return result


def base_case(suite: str, model: str, **params: Any) -> Case:
    return {"suite": suite, "model": model, "params": params}


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------


async def suite_protocol(ctx: SuiteContext, model_dirs: list[Path]) -> list[Case]:
    """REST vs gRPC, JSON vs binary tensor data, codecs, payload sizes."""
    cases: list[Case] = []
    profile = ctx.config.profile()
    for model_dir in model_dirs:
        settings = json.loads((model_dir / "model-settings.json").read_text())
        model = settings["name"]
        batch = dict(np.load(model_dir / "validation.npz"))
        request = build_request(settings, batch)
        with ctx.server(model_dir.parent):
            await ctx.parity(model_dir.parent, model_dir)
            for concurrency in profile["concurrency"][:4]:
                case = base_case(
                    "protocol",
                    model,
                    protocol="rest",
                    encoding="json",
                    concurrency=concurrency,
                    features=batch["input"].shape[1],
                )
                result = run_closed(ctx, model, request, concurrency)
                case["result"] = result.summary()
                case["stability"] = loadgen.stability_gate(
                    result, profile["warmup_seconds"]
                )
                cases.append(case)
        # gRPC pass
        if settings["implementation"].startswith("mlserver_onnx"):
            continue  # gRPC sender exercised in saturation suite for onnx
    return cases


async def suite_batching(ctx: SuiteContext, model_dirs: list[Path]) -> list[Case]:
    """Adaptive batching: max_batch_size x max_batch_time sweep."""
    cases: list[Case] = []
    profile = ctx.config.profile()
    sizes = [1, 4, 16, 64, 256]
    times = [0.0, 0.001, 0.005, 0.01, 0.025]
    if ctx.config.load_profile == "quick":
        sizes, times = [1, 16], [0.0, 0.01]
    elif ctx.config.load_profile == "standard":
        sizes, times = [1, 16, 64], [0.0, 0.005, 0.025]
    for model_dir in model_dirs[:1]:  # batching is model-agnostic: 1 model
        settings = json.loads((model_dir / "model-settings.json").read_text())
        model = settings["name"]
        batch = dict(np.load(model_dir / "validation.npz"))
        request = build_request(settings, batch)
        for max_batch_size in sizes:
            for max_batch_time in times:
                # max_batch_time=0.0 or max_batch_size<=1 disable
                # adaptive batching entirely (mlserver/batching) —
                # those cells duplicate the baseline; label them so
                # compare/analysis does not treat them as distinct
                # configurations.
                batching_disabled = max_batch_time <= 0.0 or max_batch_size <= 1
                repo = _patched_repo(
                    ctx,
                    model_dir,
                    max_batch_size=max_batch_size,
                    max_batch_time=max_batch_time,
                )
                with ctx.server(repo):
                    problems = await ctx.parity_problems(repo, repo / model_dir.name)
                    if problems:
                        # A config that returns wrong results is a
                        # FINDING, not a run-abort: record it and keep
                        # measuring the other cells. (Live-caught:
                        # max_batch_size>1 + max_batch_time>0 with a
                        # request declaring the metadata's dynamic
                        # [-1, N] shape — the batcher's minibatch_size
                        # becomes -1 and the response split drops the
                        # last row: 8 in, 7 out, HTTP 200.)
                        case = base_case(
                            "batching",
                            model,
                            max_batch_size=max_batch_size,
                            max_batch_time=max_batch_time,
                            batching_disabled=batching_disabled,
                            concurrency=profile["concurrency"][
                                len(profile["concurrency"]) // 2
                            ],
                        )
                        case["result"] = {
                            "status": "PARITY_FAILED",
                            "problems": problems,
                        }
                        cases.append(case)
                        continue
                    concurrency = profile["concurrency"][
                        len(profile["concurrency"]) // 2
                    ]
                    case = base_case(
                        "batching",
                        model,
                        max_batch_size=max_batch_size,
                        max_batch_time=max_batch_time,
                        concurrency=concurrency,
                        batching_disabled=batching_disabled,
                    )
                    result = run_closed(ctx, model, request, concurrency)
                    case["result"] = result.summary()
                    case["stability"] = loadgen.stability_gate(
                        result, profile["warmup_seconds"]
                    )
                    cases.append(case)
    return cases


async def suite_workers(ctx: SuiteContext, model_dirs: list[Path]) -> list[Case]:
    """Parallel worker pool sweep + thread contention axes."""
    import os

    cases: list[Case] = []
    profile = ctx.config.profile()
    cores = os.cpu_count() or 4
    ladder = [0, 1, 2, 4, 8, min(16, cores)]
    if ctx.config.load_profile == "quick":
        ladder = [0, 4]
    for model_dir in model_dirs:
        settings = json.loads((model_dir / "model-settings.json").read_text())
        model = settings["name"]
        batch = dict(np.load(model_dir / "validation.npz"))
        request = build_request(settings, batch)
        concurrency = profile["concurrency"][len(profile["concurrency"]) // 2]
        implementation = settings["implementation"]
        for workers in ladder:
            env = _thread_env(ctx.config, workers, implementation)
            with ctx.server(model_dir.parent, env=env, parallel_workers=workers):
                await ctx.parity(model_dir.parent, model_dir)
                case = base_case(
                    "workers",
                    model,
                    parallel_workers=workers,
                    warm_workers=ctx.config.warm_workers,
                    env_threads=True,
                    concurrency=concurrency,
                )
                result = run_closed(ctx, model, request, concurrency)
                case["result"] = result.summary()
                cases.append(case)
        # warm_workers deprecated no-op A/B (documented)
        case = base_case(
            "workers",
            model,
            warm_workers_ab=True,
            note="warm_workers is a deprecated no-op in "
            "MLServer (workers always warmed at start)",
        )
        case["result"] = {"skipped_reason": "deprecated_noop"}
        cases.append(case)
    return cases


async def suite_caching(ctx: SuiteContext, model_dirs: list[Path]) -> list[Case]:
    """Response caching. MLServer has no cache_enabled setting (audit
    finding; roadmap WI-04) — the suite probes for support, then measures
    repeat-vs-unique request distributions either way."""
    cases: list[Case] = []
    supported = _cache_supported()
    for model_dir in model_dirs[:1]:
        settings = json.loads((model_dir / "model-settings.json").read_text())
        model = settings["name"]
        if not supported:
            case = base_case("caching", model, cache_enabled="probe")
            case["result"] = {
                "status": "SKIPPED_UNSUPPORTED",
                "reason": "MLServer has no response cache (cache_enabled "
                "does not exist); see reports/PHASE3_ROADMAP.md "
                "WI-04",
            }
            cases.append(case)
        # Repeat vs unique distribution — characterizes stack-level
        # caching effects regardless of a dedicated response cache.
        # "repeat" sends one identical payload every request;
        # "unique" varies the payload per request (the old code built
        # ONE deterministic scaled payload — a second repeat, not a
        # unique stream).
        batch = dict(np.load(model_dir / "validation.npz"))
        request = build_request(settings, batch)
        with ctx.server(model_dir.parent):
            await ctx.parity(model_dir.parent, model_dir)
            profile = ctx.config.profile()
            concurrency = profile["concurrency"][len(profile["concurrency"]) // 2]

            case = base_case(
                "caching", model, distribution="repeat", concurrency=concurrency
            )
            result = run_closed(ctx, model, request, concurrency)
            case["result"] = result.summary()
            cases.append(case)

            async def factory() -> Any:
                return VaryingJsonSender(model, request)

            case = base_case(
                "caching",
                model,
                distribution="64-distinct",
                concurrency=concurrency,
            )
            result = run_closed(ctx, model, request, concurrency, make_sender=factory)
            case["result"] = result.summary()
            cases.append(case)
    return cases


def build_multimodel_repo(
    model_dirs: list[Path], repo: Path, count: int
) -> list[tuple[str, dict]]:
    """Build a repo of `count` DISTINCT model instances.

    Cycles the zoo and suffixes every copy (patched model-settings
    name) so any count can be built from a smaller zoo — the old
    `(model_dirs * 10)[:count]` repeated directory names and copytree
    raised FileExistsError whenever the tier had fewer distinct
    models than the rung. Returns (model_name, request) pairs for the
    round-robin sender.
    """
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir(parents=True)
    models_requests: list[tuple[str, dict]] = []
    for i in range(count):
        source = model_dirs[i % len(model_dirs)]
        dest_name = f"{source.name}-{i}"
        dest = repo / dest_name
        # Copy everything INCLUDING validation.npz: the server ignores
        # extra files, and parity needs the validation data from the
        # same directory as the (renamed) model it verifies.
        shutil.copytree(source, dest)
        settings_path = dest / "model-settings.json"
        settings = json.loads(settings_path.read_text())
        settings["name"] = dest_name
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
        batch = dict(np.load(dest / "validation.npz"))
        models_requests.append((dest_name, build_request(settings, batch)))
    return models_requests


def _scale_value(value: Any, factor: float) -> Any:
    """Scale numbers inside arbitrarily nested lists (real requests
    carry rows x features data, not flat lists — the original flat-only
    scaling raised 'can't multiply sequence by non-int of type float'
    on every 2-D input)."""
    if isinstance(value, list):
        return [_scale_value(v, factor) for v in value]
    return value * factor


def scaled_payload(request: dict, seq: int) -> dict:
    """Vary a JSON request payload per sequence number (unique stream).

    Pure function so the caching suite's "unique" distribution is
    testable: every seq yields a different payload.
    """
    factor = 1.0 + (seq % 977) / 31.0
    payload_request = json.loads(json.dumps(request))
    for inp in payload_request.get("inputs", []):
        if isinstance(inp.get("data"), list):
            inp["data"] = _scale_value(inp["data"], factor)
    return payload_request


class VaryingJsonSender:
    """Sends a different payload every request (unique stream).

    Pre-generates a POOL of unique bodies before the timed window and
    cycles through them. (The original built each body per request on
    the client's single event loop — json round-trip + recursive
    scaling + dumps — costing ~7.9ms on a ~117KB onnx payload: a
    single-core ceiling of ~126 rps that the reviewer proved was the
    GENERATOR's limit, not the server's. The measured 'unique' rate
    of 124 rps was a client artifact misread as a 5.6x server-side
    caching effect.)
    """

    POOL_SIZE = 64

    def __init__(
        self,
        model: str,
        request: dict,
        port: int = REST_PORT,
        accept_encoding: str = "identity",
    ):
        self._model = model
        self._request = request
        self._port = port
        # Must be set BEFORE _ensure_session reads it (the caching-suite
        # defect: this attribute was missing, every request raised
        # AttributeError and the error guard aborted the suite — caught
        # by the reviewer reading the committed evidence, missed by the
        # exit-code-judging orchestrator)
        self._accept_encoding = accept_encoding
        self._session: Any = None
        # Pre-generate BEFORE the timed window: the pool cycles, so
        # consecutive requests carry distinct payloads without any
        # per-request client CPU.
        self._pool = [
            json.dumps(scaled_payload(request, seq)).encode()
            for seq in range(self.POOL_SIZE)
        ]

    async def _ensure_session(self) -> None:
        if self._session is None:
            import aiohttp

            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300),
                connector=aiohttp.TCPConnector(limit=0),
                headers={"Accept-Encoding": self._accept_encoding},
            )

    async def send(self, seq: int) -> tuple[bool, int, int]:
        await self._ensure_session()
        payload = self._pool[seq % len(self._pool)]
        url = f"http://127.0.0.1:{self._port}/v2/models/{self._model}/infer"
        async with self._session.post(
            url, data=payload, headers={"Content-Type": "application/json"}
        ) as response:
            body = await response.read()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {body[:120]}")
            # SendFn contract: (ok, bytes_out=RESPONSE, bytes_in=REQUEST)
            return True, len(body), len(payload)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


class RoundRobinJsonSender:
    """Cycles inference load across N models (true noisy-neighbor).

    The old multimodel suite drove load to a single model while others
    sat idle — that measures an idle-neighbor RSS effect, not
    interference. This sender distributes requests round-robin so
    every loaded model is under load simultaneously.
    """

    def __init__(self, models_requests: list[tuple[str, dict]], port: int = REST_PORT):
        self._senders = [
            RestJsonSender(model, request, port) for model, request in models_requests
        ]
        self._count = len(self._senders)

    async def send(self, seq: int) -> tuple[bool, int, int]:
        return await self._senders[seq % self._count].send(seq)

    async def close(self) -> None:
        for sender in self._senders:
            await sender.close()


async def suite_multimodel(ctx: SuiteContext, model_dirs: list[Path]) -> list[Case]:
    """Multi-model serving: 2/5/10 models under simultaneous load.

    Model instances are DISTINCT copies (suffixed names, patched
    model-settings) so any count can be built from a smaller zoo —
    the old `(model_dirs * 10)[:count]` repeated directory names and
    copytree raised FileExistsError whenever the tier had fewer
    distinct models than the rung. Load is distributed round-robin
    across all loaded models (true noisy-neighbor), with container
    telemetry for RSS scaling.
    """
    cases: list[Case] = []
    if not model_dirs:
        case = base_case("multimodel", "-", reason="no models")
        case["result"] = {"status": "SKIPPED_INSUFFICIENT_MODELS"}
        return [case]
    profile = ctx.config.profile()
    concurrency = profile["concurrency"][len(profile["concurrency"]) // 2]
    for count in (2, 5, 10):
        repo = ctx.output_dir / f"repo-multi-{count}"
        models_requests = build_multimodel_repo(model_dirs, repo, count)
        with ctx.server(repo) as server:
            await ctx.parity(repo, repo / models_requests[0][0])
            case = base_case(
                "multimodel",
                models_requests[0][0],
                models=count,
                distinct_sources=len({d.name.split("-")[0] for d in model_dirs}),
                load_distribution="round-robin",
            )

            async def factory() -> Any:
                return RoundRobinJsonSender(models_requests)

            recorder = telemetry.TelemetryRecorder(container_name=server.name)
            result = run_closed(
                ctx,
                models_requests[0][0],
                models_requests[0][1],
                concurrency,
                recorder,
                make_sender=factory,
            )
            case["result"] = result.summary()
            telemetry_summary = telemetry.summarize(recorder.records)
            case["telemetry"] = telemetry_summary
            # Positive control: empty telemetry is a silent no-op —
            # surface it instead of shipping an empty summary.
            if not recorder.records or "host_cpu_percent_mean" not in (
                telemetry_summary or {}
            ):
                case["telemetry_status"] = "TELEMETRY_EMPTY"
            cases.append(case)
    return cases


async def suite_lifecycle(ctx: SuiteContext, model_dirs: list[Path]) -> list[Case]:
    """Load/unload latency, ready time, cold-start vs warm inference."""
    import aiohttp

    cases: list[Case] = []
    for model_dir in model_dirs[:1]:
        settings = json.loads((model_dir / "model-settings.json").read_text())
        model = settings["name"]
        batch = dict(np.load(model_dir / "validation.npz"))
        request = build_request(settings, batch)
        with ctx.server(model_dir.parent):
            await ctx.parity(model_dir.parent, model_dir)
            base = f"http://127.0.0.1:{REST_PORT}"

            async def timed(method: str, path: str) -> tuple[float, int]:
                started = time.perf_counter()
                async with aiohttp.ClientSession() as session:
                    async with session.request(method, base + path) as response:
                        await response.read()
                        return time.perf_counter() - started, response.status

            async def timed_post(path: str, payload: str) -> tuple[float, int]:
                started = time.perf_counter()
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        base + path,
                        data=payload,
                        headers={
                            "Content-Type": "application/json",
                            "Accept-Encoding": "identity",
                        },
                    ) as response:
                        await response.read()
                        return time.perf_counter() - started, response.status

            # Cold start: first inference on a freshly loaded model.
            # ONE timed request, immediately after load — the old code
            # ran a full warmup+window closed loop and reported its
            # p50, which is warm by definition.
            unload_s, status_u = await timed(
                "POST", f"/v2/repository/models/{model}/unload"
            )
            load_s, status_l = await timed(
                "POST", f"/v2/repository/models/{model}/load"
            )
            ready_s, status_r = await timed("GET", f"/v2/models/{model}/ready")
            cold_ms, cold_status = await timed_post(
                f"/v2/models/{model}/infer", json.dumps(request)
            )
            # Warm: steady-state under the standard closed loop
            warm = run_closed(ctx, model, request, 1)
            warm_case = warm.summary()
            case = base_case(
                "lifecycle",
                model,
                unload_s=round(unload_s, 4),
                load_s=round(load_s, 4),
                ready_s=round(ready_s, 4),
                unload_status=status_u,
                load_status=status_l,
                ready_status=status_r,
            )
            case["result"] = {
                "cold_first_inference_ms": round(cold_ms * 1000, 3),
                "cold_first_inference_status": cold_status,
                "warm_p50_ms": warm_case.get("latency_ms_p50"),
            }
            cases.append(case)

        # -- Lazy vs eager loading: startup-to-ready + deferred load ---
        # load_models_at_startup=false skips the boot-time load: the
        # server reports ready immediately (empty_registry_readiness
        # defaults True) and the model is loaded later via the explicit
        # repository API. NOTE: infer does NOT auto-load (404 until the
        # load call) — that is the real semantic, so the measured axes
        # are: startup-to-ready time (container start -> health ready)
        # and the deferred explicit-load call latency.
        first_s: dict[str, float] = {}
        load_api_s: float | None = None
        for load_mode in ("eager", "lazy"):
            env = (
                {"MLSERVER_LOAD_MODELS_AT_STARTUP": "false"}
                if load_mode == "lazy"
                else {}
            )
            t0 = time.perf_counter()
            with ctx.server(model_dir.parent, env=env):
                # ctx.server entry = docker run + wait_healthy: eager
                # includes the boot-time model load, lazy does not.
                first_s[load_mode] = time.perf_counter() - t0
                if load_mode == "lazy":
                    # Deferred load: the explicit repository call a
                    # client must make before the model serves.
                    load_api_s, load_status = await timed(
                        "POST", f"/v2/repository/models/{model}/load"
                    )
                case = base_case(
                    "lifecycle",
                    model,
                    load_mode=load_mode,
                )
                result: dict[str, Any] = {
                    "startup_to_ready_s": round(first_s[load_mode], 3)
                }
                if load_mode == "lazy" and load_api_s is not None:
                    result["deferred_load_api_s"] = round(load_api_s, 4)
                    result["load_status"] = load_status
                case["result"] = result
                cases.append(case)
        if "eager" in first_s and "lazy" in first_s:
            case = base_case(
                "lifecycle",
                model,
                comparison="lazy-vs-eager startup",
            )
            case["result"] = {
                "eager_startup_to_ready_s": round(first_s["eager"], 3),
                "lazy_startup_to_ready_s": round(first_s["lazy"], 3),
                "startup_savings_s": round(first_s["eager"] - first_s["lazy"], 3),
                "deferred_load_api_s": (
                    round(load_api_s, 4) if load_api_s is not None else None
                ),
            }
            cases.append(case)
    return cases


async def suite_saturation(ctx: SuiteContext, model_dirs: list[Path]) -> list[Case]:
    """Closed-loop ladder to peak, then open-loop at fractions of peak
    (constant + Poisson) measuring true queueing delay."""
    cases: list[Case] = []
    profile = ctx.config.profile()
    for model_dir in model_dirs:
        settings = json.loads((model_dir / "model-settings.json").read_text())
        model = settings["name"]
        # The paired driver keys open-loop rate overrides by framework
        # (its own loop variable); the runner stamps case["framework"]
        # only after the suite returns, so derive the framework here
        # from the model's implementation class.
        from .zoo import IMPLEMENTATIONS

        impl = settings.get("implementation", "")
        framework = next(
            (name for name, cls in IMPLEMENTATIONS.items() if cls == impl), impl
        )
        batch = dict(np.load(model_dir / "validation.npz"))
        request = build_request(settings, batch)
        with ctx.server(model_dir.parent):
            await ctx.parity(model_dir.parent, model_dir)
            peak = 0.0
            for concurrency in profile["concurrency"]:
                result = run_closed(ctx, model, request, concurrency)
                summary = result.summary()
                peak = max(peak, summary["throughput_rps"])
                case = base_case(
                    "saturation", model, mode="closed", concurrency=concurrency
                )
                case["result"] = summary
                case["stability"] = loadgen.stability_gate(
                    result, profile["warmup_seconds"]
                )
                cases.append(case)
            for fraction in profile["qps_fractions"]:
                target_qps = round(peak * fraction, 2)
                # Paired-driver override: when PAIRED_TARGET_QPS is
                # set, open-loop runs are driven at the BASELINE's
                # measured rate for this exact cell (framework, model,
                # fraction, arrival — the driver's per-model keying),
                # not this side's own peak-derived rate, so both sides
                # see identical load. Parsed once here; applied
                # per-arrival in the loop below so constant and
                # Poisson runs each get their own rate, and the rate
                # ACTUALLY driven is what gets reported in
                # result.target_qps.
                overrides: dict[str, float] | None = None
                paired_override = os.environ.get("PAIRED_TARGET_QPS")
                if paired_override:
                    try:
                        parsed = json.loads(paired_override)
                        if isinstance(parsed, dict):
                            overrides = parsed
                    except (ValueError, TypeError):
                        overrides = None

                async def go(poisson: bool, rate: float) -> loadgen.LoadResult:
                    # Fresh sender per invocation: each run_closed/
                    # open_loop call gets its own event loop, and
                    # aiohttp sessions are loop-bound.
                    sender_obj = rest_sender(model, request)
                    try:
                        return await loadgen.open_loop(
                            rate,
                            profile["window_seconds"],
                            profile["warmup_seconds"],
                            sender_obj.send,
                            poisson=poisson,
                        )
                    finally:
                        await sender_obj.close()

                for poisson in (False, True):
                    arrival = "poisson" if poisson else "constant"
                    rate = target_qps
                    if overrides is not None:
                        key = f"{framework}:{model}:{fraction}:{arrival}"
                        rate = overrides.get(key, rate)
                    result = run_coroutine_blocking(go(poisson, rate))
                    # Key by fraction_of_peak + arrival (stable across
                    # runs), NOT target_qps (derived from each round's
                    # measured peak — differs every round, so
                    # compare.py can never pair open-loop cells across
                    # runs). target_qps is the rate ACTUALLY DRIVEN —
                    # reported in the result, not a matching param.
                    case = base_case(
                        "saturation",
                        model,
                        mode="open",
                        fraction_of_peak=fraction,
                        arrival=arrival,
                    )
                    case["result"] = result.summary()
                    case["result"]["target_qps"] = rate
                    cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# Suite: processing (wire-format / processing-path matrix)
# ---------------------------------------------------------------------------


async def suite_processing(ctx: SuiteContext, model_dirs: list[Path]) -> list[Case]:
    """Wire format x processing path: JSON vs binary tensor on the
    frontend/worker axes, plus the inference IPC transport (queue vs zmq).

    Server settings swept (MLSERVER_* env vars):
      - rest_worker_json: JSON frames to workers (default off)
      - rest_binary_tensor: KServe v2 binary tensor extension (frontend
        zero-copy parse)
      - rest_worker_binary: binary frames to workers (requires
        rest_binary_tensor)
      - parallel_inference_transport: queue | zmq

    Also drives the request encodings from the client side: REST JSON
    baseline vs REST binary tensor (with binary outputs negotiated).

    Every variant is feature-probed against the running image: fork
    features (binary tensor paths) may postdate the published image,
    and an unapplied env var would silently duplicate the baseline —
    unsupported variants are recorded as SKIPPED_UNSUPPORTED_IMAGE.
    """
    from .docker_server import probe_image_import, probe_settings

    cases: list[Case] = []
    profile = ctx.config.profile()
    concurrency = profile["concurrency"][len(profile["concurrency"]) // 2]
    # Pre-start probe: a zmq-transport server crashes at startup when
    # the image lacks the zmq extra — probing the image up front
    # avoids burning the health timeout on a dead container.
    zmq_available = probe_image_import(ctx.image, "zmq")
    for model_dir in model_dirs:
        settings = json.loads((model_dir / "model-settings.json").read_text())
        model = settings["name"]
        batch = dict(np.load(model_dir / "validation.npz"))
        request = build_request(settings, batch)
        binary_body, header_len = wire.build_binary_request(settings, batch)

        variants: list[tuple[str, dict[str, str], tuple[str, ...]]] = [
            (
                "stock-json",
                {
                    "MLSERVER_REST_WORKER_JSON": "false",
                    "MLSERVER_REST_BINARY_TENSOR": "false",
                    "MLSERVER_REST_WORKER_BINARY": "false",
                },
                ("rest_worker_json",),
            ),
            (
                "json-worker",
                {
                    "MLSERVER_REST_WORKER_JSON": "true",
                    "MLSERVER_REST_BINARY_TENSOR": "false",
                    "MLSERVER_REST_WORKER_BINARY": "false",
                },
                ("rest_worker_json",),
            ),
            (
                "binary-frontend",
                {
                    "MLSERVER_REST_WORKER_JSON": "false",
                    "MLSERVER_REST_BINARY_TENSOR": "true",
                    "MLSERVER_REST_WORKER_BINARY": "false",
                },
                ("rest_binary_tensor",),
            ),
            (
                "binary-worker",
                {
                    "MLSERVER_REST_WORKER_JSON": "false",
                    "MLSERVER_REST_BINARY_TENSOR": "true",
                    "MLSERVER_REST_WORKER_BINARY": "true",
                },
                ("rest_binary_tensor", "rest_worker_binary"),
            ),
        ]
        ordered_variants = list(reversed(variants)) if ctx.arm_order_flip else variants
        for variant, env, needed in ordered_variants:
            with ctx.server(model_dir.parent, env=env) as server:
                await ctx.parity(model_dir.parent, model_dir)
                support = probe_settings(server.name, *needed)
                missing = [n for n in needed if not support.get(n)]
                if missing:
                    case = base_case(
                        "processing",
                        model,
                        variant=variant,
                        encoding="json",
                        concurrency=concurrency,
                    )
                    case["result"] = {
                        "status": "SKIPPED_UNSUPPORTED_IMAGE",
                        "missing_settings": missing,
                        "note": "image predates these fork features; "
                        "use --image-source source to test a fresh build",
                    }
                    cases.append(case)
                    continue
                # JSON request through this configuration
                case = base_case(
                    "processing",
                    model,
                    variant=variant,
                    encoding="json",
                    concurrency=concurrency,
                )
                result = run_closed(ctx, model, request, concurrency)
                case["result"] = result.summary()
                cases.append(case)
                # Binary tensor request (only valid when the extension
                # is enabled)
                if env["MLSERVER_REST_BINARY_TENSOR"] == "true":
                    case = base_case(
                        "processing",
                        model,
                        variant=variant,
                        encoding="binary-tensor",
                        concurrency=concurrency,
                    )

                    def make_sender(body=binary_body, hl=header_len, m=model) -> Any:
                        async def factory() -> Any:
                            return wire.rest_binary_sender(m, body, hl)

                        return factory

                    result = run_closed(
                        ctx,
                        model,
                        request,
                        concurrency,
                        make_sender=make_sender(),
                    )
                    case["result"] = result.summary()
                    cases.append(case)

        # Inference IPC transport: queue vs zmq
        for transport in ("queue", "zmq"):
            if transport == "zmq" and not zmq_available:
                case = base_case(
                    "processing",
                    model,
                    transport=transport,
                    concurrency=concurrency,
                )
                case["result"] = {
                    "status": "SKIPPED_UNSUPPORTED_IMAGE",
                    "missing": "zmq extra not installed in image",
                }
                cases.append(case)
                continue
            env = {"MLSERVER_PARALLEL_INFERENCE_TRANSPORT": transport}
            with ctx.server(model_dir.parent, env=env) as server:
                await ctx.parity(model_dir.parent, model_dir)
                case = base_case(
                    "processing",
                    model,
                    transport=transport,
                    concurrency=concurrency,
                )
                result = run_closed(ctx, model, request, concurrency)
                case["result"] = result.summary()
                cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# Suite: framework (model framework-level pydantic settings via extra)
# ---------------------------------------------------------------------------


ONNX_PROBE_KEYS = (
    "providers",
    "provider_options",
    "session_options",
    "run_options",
    "session_config_entries",
    "cuda_graph",
    "cuda_blocking_sync",
)
"""Every OnnxSettings field any framework-suite axis needs — probed
ONCE on the first server. (The old code probed only "session_options"
first, so the later re-probe guards never ran and every other axis was
wrongly recorded as SKIPPED_UNSUPPORTED_IMAGE even when supported.)"""

FRAMEWORK_AXIS_FIELDS = {
    "inter_op_threads_2": "session_options",
    "mem_pattern_off": "session_options",
    "cpu_arena_off": "session_options",
    "log_severity_3": "session_options",
    "providers_cpu_fallback_order": "providers",
    "run_options_configured": "run_options",
    "session_config_entries": "session_config_entries",
    "cuda_graph": "cuda_graph",
    "cuda_blocking_sync": "cuda_blocking_sync",
    "cudnn_exhaustive": "provider_options",
    "cudnn_heuristic": "provider_options",
    "arena_next_pow2": "provider_options",
}
"""Axis name -> OnnxSettings field (the axis names themselves are
never probe keys — the old GPU check looked up the axis name and
always skipped)."""


async def suite_framework(ctx: SuiteContext, model_dirs: list[Path]) -> list[Case]:
    """Framework-level settings that auto-resolve from
    ``parameters.extra`` (e.g. OnnxSettings: providers,
    provider_options, session_options, run_options, cuda_graph,
    cuda_blocking_sync) and MLSERVER_MODEL_ONNX_* env vars.

    This is the deep runtime-tuning surface: ORT session thread pools,
    execution mode, graph optimization level, memory arenas, provider
    options, CUDA graphs.

    OnnxSettings fields are feature-probed against the running image —
    fork additions (cuda_graph, run_options, ...) skip gracefully on
    images that predate them.
    """
    from .docker_server import probe_onnx_settings

    onnx_dirs = [
        d
        for d in model_dirs
        if json.loads((d / "model-settings.json").read_text())[
            "implementation"
        ].startswith("mlserver_onnx")
    ]
    if not onnx_dirs:
        case = base_case("framework", "-", reason="no ONNX models in zoo")
        case["result"] = {"status": "SKIPPED_NO_ONNX_MODELS"}
        return [case]

    cases: list[Case] = []
    profile = ctx.config.profile()
    concurrency = profile["concurrency"][len(profile["concurrency"]) // 2]
    is_quick = ctx.config.load_profile == "quick"

    onnx_support: dict[str, bool] = {}
    for model_dir in onnx_dirs:
        settings = json.loads((model_dir / "model-settings.json").read_text())
        model = settings["name"]
        batch = dict(np.load(model_dir / "validation.npz"))
        request = build_request(settings, batch)

        # -- session_options sweeps (ORT SessionOptions attributes) --
        intra_ladder = [1, 2, 4] if is_quick else [1, 2, 4, 8]
        for threads in intra_ladder:
            repo = _patched_repo_extra(
                ctx,
                model_dir,
                {
                    "session_options": {
                        "intra_op_num_threads": threads,
                        "inter_op_num_threads": 1,
                    }
                },
            )
            with ctx.server(repo) as server:
                await ctx.parity(repo, repo / model_dir.name)
                if not onnx_support:
                    onnx_support.update(
                        probe_onnx_settings(server.name, *ONNX_PROBE_KEYS)
                    )
                if not onnx_support.get("session_options", False):
                    case = base_case(
                        "framework",
                        model,
                        axis="session_options.intra_op_num_threads",
                        value=threads,
                        concurrency=concurrency,
                    )
                    case["result"] = {
                        "status": "SKIPPED_UNSUPPORTED_IMAGE",
                        "missing_onnx_setting": "session_options",
                    }
                    cases.append(case)
                    continue
                case = base_case(
                    "framework",
                    model,
                    axis="session_options.intra_op_num_threads",
                    value=threads,
                    concurrency=concurrency,
                )
                result = run_closed(ctx, model, request, concurrency)
                case["result"] = result.summary()
                cases.append(case)

        # NOTE: execution_mode and graph_optimization_level are
        # ORT pybind ENUM properties — setattr with a plain int (the
        # only thing JSON can carry) raises TypeError, so they cannot
        # be benchmarked through parameters.extra. Only JSON-settable
        # attributes are swept (verified against onnxruntime).
        axes: list[tuple[str, dict[str, Any]]] = [
            (
                "inter_op_threads_2",
                {"session_options": {"inter_op_num_threads": 2}},
            ),
            (
                "mem_pattern_off",
                {"session_options": {"enable_mem_pattern": False}},
            ),
            (
                "cpu_arena_off",
                {"session_options": {"enable_cpu_mem_arena": False}},
            ),
            (
                "log_severity_3",
                {"session_options": {"log_severity_level": 3}},
            ),
            (
                "providers_cpu_fallback_order",
                {
                    "providers": [
                        "CPUExecutionProvider",
                    ]
                },
            ),
            (
                "run_options_configured",
                {"run_options": {"log_severity_level": 3}},
            ),
            (
                "session_config_entries",
                {"session_config_entries": {"session.intra_op.allow_spinning": "1"}},
            ),
        ]
        for axis, extra in axes:
            repo = _patched_repo_extra(ctx, model_dir, extra)
            with ctx.server(repo):
                await ctx.parity(repo, repo / model_dir.name)
                needed_field = FRAMEWORK_AXIS_FIELDS.get(axis, axis)
                if not onnx_support.get(needed_field, False):
                    case = base_case(
                        "framework",
                        model,
                        axis=axis,
                        concurrency=concurrency,
                    )
                    case["result"] = {
                        "status": "SKIPPED_UNSUPPORTED_IMAGE",
                        "missing_onnx_setting": needed_field,
                    }
                    cases.append(case)
                    continue
                case = base_case(
                    "framework",
                    model,
                    axis=axis,
                    concurrency=concurrency,
                )
                case["params"]["extra"] = extra
                result = run_closed(ctx, model, request, concurrency)
                case["result"] = result.summary()
                cases.append(case)

        # -- GPU-gated axes (cuda_graph, cuda_blocking_sync, cudnn) --
        if ctx.config.target == "gpu":
            gpu_axes: list[tuple[str, dict[str, Any]]] = [
                (
                    "cuda_graph",
                    {
                        "providers": ["CUDAExecutionProvider"],
                        "cuda_graph": True,
                    },
                ),
                (
                    "cuda_blocking_sync",
                    {
                        "providers": ["CUDAExecutionProvider"],
                        "cuda_blocking_sync": True,
                    },
                ),
                (
                    "cudnn_exhaustive",
                    {
                        "providers": ["CUDAExecutionProvider"],
                        "provider_options": [
                            {
                                "cudnn_conv_algo_search": "EXHAUSTIVE",
                            }
                        ],
                    },
                ),
                (
                    "cudnn_heuristic",
                    {
                        "providers": ["CUDAExecutionProvider"],
                        "provider_options": [
                            {
                                "cudnn_conv_algo_search": "HEURISTIC",
                            }
                        ],
                    },
                ),
                (
                    "arena_next_pow2",
                    {
                        "providers": ["CUDAExecutionProvider"],
                        "provider_options": [
                            {"arena_extend_strategy": "kNextPowerOfTwo"}
                        ],
                    },
                ),
            ]
            for axis, extra in gpu_axes:
                repo = _patched_repo_extra(ctx, model_dir, extra)
                with ctx.server(repo):
                    await ctx.parity(repo, repo / model_dir.name)
                    needed = FRAMEWORK_AXIS_FIELDS.get(axis, axis)
                    if not onnx_support.get(needed, False):
                        case = base_case(
                            "framework",
                            model,
                            axis=axis,
                            concurrency=concurrency,
                        )
                        case["result"] = {
                            "status": "SKIPPED_UNSUPPORTED_IMAGE",
                            "missing_onnx_setting": needed,
                        }
                        cases.append(case)
                        continue
                    case = base_case(
                        "framework",
                        model,
                        axis=axis,
                        concurrency=concurrency,
                    )
                    case["params"]["extra"] = extra
                    result = run_closed(ctx, model, request, concurrency)
                    case["result"] = result.summary()
                    cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# Suite: middleware (server-level observability / compression overhead)
# ---------------------------------------------------------------------------


async def suite_middleware(ctx: SuiteContext, model_dirs: list[Path]) -> list[Case]:
    """Server-level middleware overhead: gzip response compression,
    access logging, metrics endpoint, structured logging, and
    OpenTelemetry tracing export.

    Each axis is measured on/off against the same model and load —
    the delta is the middleware cost paid per request.
    """
    cases: list[Case] = []
    profile = ctx.config.profile()
    concurrency = profile["concurrency"][len(profile["concurrency"]) // 2]
    model_dir = model_dirs[0]
    settings = json.loads((model_dir / "model-settings.json").read_text())
    model = settings["name"]
    batch = dict(np.load(model_dir / "validation.npz"))
    request = build_request(settings, batch)

    axes: list[tuple[str, dict[str, str], dict[str, str]]] = [
        # (label, env_on, env_off)
        (
            "gzip",
            {"MLSERVER_GZIP_ENABLED": "true"},
            {"MLSERVER_GZIP_ENABLED": "false"},
        ),
        (
            "access_log",
            {"MLSERVER_ACCESS_LOG": "true"},
            {"MLSERVER_ACCESS_LOG": "false"},
        ),
        (
            "metrics_endpoint",
            {"MLSERVER_METRICS_ENDPOINT": "/metrics"},
            {"MLSERVER_METRICS_ENDPOINT": ""},
        ),
        (
            "structured_logging",
            {"MLSERVER_USE_STRUCTURED_LOGGING": "true"},
            {"MLSERVER_USE_STRUCTURED_LOGGING": "false"},
        ),
        (
            "tracing_no_collector",
            {"MLSERVER_TRACING_SERVER": "otlp-collector.nowhere:4317"},
            {"MLSERVER_TRACING_SERVER": ""},
        ),
    ]
    state_order = ("off", "on") if ctx.arm_order_flip else ("on", "off")
    for label, env_on, env_off in axes:
        for state, env in (
            ("on", env_on) if state == "on" else ("off", env_off)
            for state in state_order
        ):
            with ctx.server(model_dir.parent, env=env):
                await ctx.parity(model_dir.parent, model_dir)
                case = base_case(
                    "middleware",
                    model,
                    middleware=label,
                    state=state,
                    concurrency=concurrency,
                )
                if label == "gzip" and state == "on":
                    # Positive control: the "on" arm sends
                    # Accept-Encoding: gzip and records whether the
                    # server actually compressed — without this the
                    # axis can silently measure nothing.
                    senders: list[RestJsonSender] = []

                    async def gzip_factory() -> Any:
                        sender = RestJsonSender(
                            model,
                            request,
                            accept_encoding="gzip",
                            track_encoding=True,
                        )
                        senders.append(sender)
                        return sender

                    result = run_closed(
                        ctx,
                        model,
                        request,
                        concurrency,
                        make_sender=gzip_factory,
                    )
                    case["result"] = result.summary()
                    if senders:
                        case["result"]["gzip_engaged"] = (
                            senders[0].compressed_responses > 0
                        )
                        case["result"]["compressed_responses"] = senders[
                            0
                        ].compressed_responses
                        case["result"]["uncompressed_responses"] = senders[
                            0
                        ].uncompressed_responses
                else:
                    result = run_closed(ctx, model, request, concurrency)
                    case["result"] = result.summary()
                cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# Suite: codec (content_type variants + sklearn predict_fn)
# ---------------------------------------------------------------------------


async def suite_codec(ctx: SuiteContext, model_dirs: list[Path]) -> list[Case]:
    """Input codec variants (parameters.content_type: np / pd / str)
    and sklearn's predict_fn selection (predict / predict_proba /
    transform via parameters.extra).

    Wire-format costs differ sharply by codec: np (JSON lists) is the
    baseline; pd sends dataframe records; str sends a JSON-encoded
    string the server must re-parse.
    """
    cases: list[Case] = []
    profile = ctx.config.profile()
    concurrency = profile["concurrency"][len(profile["concurrency"]) // 2]

    # -- content_type variants: np on any model, pd on pandas-native
    #    runtimes (sklearn). str/base64/datetime apply to BYTES/string
    #    payloads — the zoo is numeric-only, so they are documented
    #    skips rather than error cases.
    measured_pd = False
    for model_dir in model_dirs:
        settings = json.loads((model_dir / "model-settings.json").read_text())
        if len(settings.get("inputs", [])) != 1:
            continue
        if settings["inputs"][0]["datatype"] not in ("FP32", "FP64"):
            continue
        implementation = settings["implementation"]
        model = settings["name"]
        batch = dict(np.load(model_dir / "validation.npz"))
        variants = ["np"]
        if implementation.startswith("mlserver_sklearn") and not measured_pd:
            variants.append("pd")
        for content_type in variants:
            request = wire.build_request_content_type(settings, batch, content_type)
            with ctx.server(model_dir.parent):
                await ctx.parity(model_dir.parent, model_dir)
                case = base_case(
                    "codec",
                    model,
                    content_type=content_type,
                    concurrency=concurrency,
                )
                result = run_closed(ctx, model, request, concurrency)
                case["result"] = result.summary()
                cases.append(case)
        if "pd" in variants:
            measured_pd = True
        break  # one representative model is enough for the codec axis

    # Documented skip: codecs that need non-numeric payloads
    case = base_case("codec", "-", content_type="str")
    case["result"] = {
        "status": "SKIPPED_NO_BYTES_MODELS",
        "reason": "StringCodec/base64/datetime content types apply to "
        "BYTES/string model inputs; the zoo is numeric-only",
    }
    cases.append(case)

    # -- output dtype axis: INT64 labels vs FP regressions ----------
    by_variant: dict[str, Path] = {}
    for d in model_dirs:
        by_variant[d.name] = d
    classifier = next((d for n, d in by_variant.items() if "random-forest" in n), None)
    regressor = next((d for n, d in by_variant.items() if "grad-boost" in n), None)
    dtype_results: dict[str, dict[str, Any]] = {}
    for label, d in (("int64-labels", classifier), ("float-values", regressor)):
        if d is None:
            continue
        s = json.loads((d / "model-settings.json").read_text())
        b = dict(np.load(d / "validation.npz"))
        r = build_request(s, b)
        with ctx.server(d.parent):
            await ctx.parity(d.parent, d)
            case = base_case(
                "codec", s["name"], output_dtype=label, concurrency=concurrency
            )
            result = run_closed(ctx, s["name"], r, concurrency)
            case["result"] = result.summary()
            dtype_results[label] = case["result"]
            cases.append(case)
    if len(dtype_results) == 2:
        case = base_case("codec", "-", comparison="int64-vs-float output encoding")
        case["result"] = {
            label: {
                "throughput_rps": r.get("throughput_rps"),
                "latency_ms_p50": r.get("latency_ms_p50"),
            }
            for label, r in dtype_results.items()
        }
        cases.append(case)

    # -- NaN output axis: NaN->None per-element loop vs clean floats --
    nan_model = by_variant.get("onnx-nan-small")
    clean_model = by_variant.get("onnx-mlp-small")
    if nan_model is not None and clean_model is not None:
        nan_results: dict[str, dict[str, Any]] = {}
        for label, d in (
            ("nan-outputs", nan_model),
            ("clean-floats", clean_model),
        ):
            s = json.loads((d / "model-settings.json").read_text())
            b = dict(np.load(d / "validation.npz"))
            r = build_request(s, b)
            with ctx.server(d.parent):
                await ctx.parity(d.parent, d)
                case = base_case(
                    "codec", s["name"], output_content=label, concurrency=concurrency
                )
                result = run_closed(ctx, s["name"], r, concurrency)
                case["result"] = result.summary()
                nan_results[label] = case["result"]
                cases.append(case)
        if len(nan_results) == 2:
            case = base_case(
                "codec",
                "-",
                comparison="nan-vs-clean output encoding "
                "(NaN->None per-element loop)",
            )
            case["result"] = {
                label: {
                    "throughput_rps": r.get("throughput_rps"),
                    "latency_ms_p50": r.get("latency_ms_p50"),
                }
                for label, r in nan_results.items()
            }
            cases.append(case)

    # -- sklearn predict_fn via parameters.extra --
    sklearn_dirs = [
        d
        for d in model_dirs
        if json.loads((d / "model-settings.json").read_text())[
            "implementation"
        ].startswith("mlserver_sklearn")
    ]
    for model_dir in sklearn_dirs[:1]:
        settings = json.loads((model_dir / "model-settings.json").read_text())
        model = settings["name"]
        batch = dict(np.load(model_dir / "validation.npz"))
        request = build_request(settings, batch)
        for predict_fn in ("predict", "predict_proba"):
            repo = _patched_repo_extra(ctx, model_dir, {"predict_fn": predict_fn})
            with ctx.server(repo):
                await ctx.parity(repo, repo / model_dir.name)
                case = base_case(
                    "codec",
                    model,
                    predict_fn=predict_fn,
                    concurrency=concurrency,
                )
                result = run_closed(ctx, model, request, concurrency)
                case["result"] = result.summary()
                cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# Suite: grpc (gRPC protocol, raw input buffers, message limits)
# ---------------------------------------------------------------------------


async def suite_grpc(ctx: SuiteContext, model_dirs: list[Path]) -> list[Case]:
    """gRPC dataplane: typed contents vs raw input contents, the
    server's grpc_raw_input_buffers zero-copy path, and
    grpc_max_message_length limits. Skipped gracefully when grpcio
    or the MLServer protos are unavailable on the host."""
    if not wire.grpc_available():
        case = base_case("grpc", "-", reason="grpcio or MLServer protos unavailable")
        case["result"] = {"status": "SKIPPED_GRPC_UNAVAILABLE"}
        return [case]

    cases: list[Case] = []
    profile = ctx.config.profile()
    concurrency = profile["concurrency"][len(profile["concurrency"]) // 2]

    for model_dir in model_dirs:
        settings = json.loads((model_dir / "model-settings.json").read_text())
        model = settings["name"]
        batch = dict(np.load(model_dir / "validation.npz"))
        json_request = build_request(settings, batch)

        grpc_typed = wire.build_grpc_request(settings, batch, model, raw_input=False)
        grpc_raw = wire.build_grpc_request(settings, batch, model, raw_input=True)

        # REST baseline for the same model/concurrency
        with ctx.server(model_dir.parent):
            await ctx.parity(model_dir.parent, model_dir)
            case = base_case("grpc", model, protocol="rest", concurrency=concurrency)
            result = run_closed(ctx, model, json_request, concurrency)
            case["result"] = result.summary()
            cases.append(case)

        # gRPC typed contents (default parsing path)
        with ctx.server(model_dir.parent):
            await ctx.parity(model_dir.parent, model_dir)
            case = base_case(
                "grpc",
                model,
                protocol="grpc",
                contents="typed",
                concurrency=concurrency,
            )

            def factory_typed(m=model, r=grpc_typed) -> Any:
                async def factory() -> Any:
                    return wire.grpc_sender(m, r)

                return factory

            result = run_closed(
                ctx,
                model,
                json_request,
                concurrency,
                make_sender=factory_typed(),
            )
            case["result"] = result.summary()
            cases.append(case)

        # gRPC raw input contents + server zero-copy path
        for raw_buffers in ("false", "true"):
            env = {"MLSERVER_GRPC_RAW_INPUT_BUFFERS": raw_buffers}
            with ctx.server(model_dir.parent, env=env):
                await ctx.parity(model_dir.parent, model_dir)
                case = base_case(
                    "grpc",
                    model,
                    protocol="grpc",
                    contents="raw",
                    grpc_raw_input_buffers=raw_buffers,
                    concurrency=concurrency,
                )

                def factory_raw(m=model, r=grpc_raw) -> Any:
                    async def factory() -> Any:
                        return wire.grpc_sender(m, r)

                    return factory

                result = run_closed(
                    ctx,
                    model,
                    json_request,
                    concurrency,
                    make_sender=factory_raw(),
                )
                case["result"] = result.summary()
                cases.append(case)

        # grpc_max_message_length (server accept envelope)
        for limit in ("1048576", "33554432"):
            env = {"MLSERVER_GRPC_MAX_MESSAGE_LENGTH": limit}
            with ctx.server(model_dir.parent, env=env):
                await ctx.parity(model_dir.parent, model_dir)
                case = base_case(
                    "grpc",
                    model,
                    protocol="grpc",
                    contents="typed",
                    grpc_max_message_length=limit,
                    concurrency=concurrency,
                )

                def factory_len(m=model, r=grpc_typed) -> Any:
                    async def factory() -> Any:
                        return wire.grpc_sender(m, r)

                    return factory

                result = run_closed(
                    ctx,
                    model,
                    json_request,
                    concurrency,
                    make_sender=factory_len(),
                )
                case["result"] = result.summary()
                cases.append(case)
        break  # one model is enough for the transport axis
    return cases


# ---------------------------------------------------------------------------
# Suite: hardware (ONNX CPU image on CPU vs CUDA image on GPU)
# ---------------------------------------------------------------------------


async def suite_hardware(ctx: SuiteContext, model_dirs: list[Path]) -> list[Case]:
    """CPU image on CPU vs CUDA image on GPU — the deployment matrix
    a customer actually chooses between.

    Two arms, same model, same request-batch ladder, same load:
      - CPU arm: the CPU image (digest-pinned default), target=cpu —
        no --gpus, no GPU guard (CPU is the intended config, not a
        fallback)
      - GPU arm: the run's image (GPU image for --target gpu runs),
        target=gpu — --gpus all and the anti-silent-fallback guard
        ENFORCED (FAILED_GPU_PROVIDER_FALLBACK if CUDA is not active)

    Both arms run the images' STOCK provider behaviour (no
    parameters.extra pinning) — the images' defaults are the thing
    being compared. The request-batch-size sweep gives the crossover
    curve: small batches are latency-bound, large batches are
    throughput-bound. Each comparison case carries speedup_vs_cpu.
    """
    from .docker_server import detect_default_image, resolve_digest

    if ctx.config.target != "gpu":
        case = base_case(
            "hardware",
            "-",
            reason="CPU-vs-GPU comparison needs --target gpu "
            "(CUDA host + CUDA image)",
        )
        case["result"] = {"status": "SKIPPED_NEEDS_GPU_TARGET"}
        return [case]

    onnx_dirs = [
        d
        for d in model_dirs
        if json.loads((d / "model-settings.json").read_text())[
            "implementation"
        ].startswith("mlserver_onnx")
    ]
    if not onnx_dirs:
        case = base_case("hardware", "-", reason="no ONNX models in zoo")
        case["result"] = {"status": "SKIPPED_NO_ONNX_MODELS"}
        return [case]

    cases: list[Case] = []
    profile = ctx.config.profile()
    concurrency = profile["concurrency"][len(profile["concurrency"]) // 2]
    is_quick = ctx.config.load_profile == "quick"
    batch_ladder = [1, 32] if is_quick else [1, 8, 32, 128]

    # CPU arm image: the CPU default, digest-pinned (never a floating
    # tag). The GPU arm uses the run's image (already pinned by the
    # runner for --target gpu).
    cpu_image = resolve_digest(detect_default_image("cpu", ""))
    arms: list[tuple[str, str, str]] = [
        ("cpu", cpu_image, "cpu"),
        ("gpu", ctx.image, "gpu"),
    ]

    for model_dir in onnx_dirs:
        settings = json.loads((model_dir / "model-settings.json").read_text())
        model = settings["name"]
        batch = dict(np.load(model_dir / "validation.npz"))
        arm_results: dict[tuple[str, int], dict[str, Any]] = {}

        for device, image, target in arms:
            # GPU arm: target=gpu -> --gpus all + anti-silent-fallback
            # guard inside start(). CPU arm: target=cpu, no guard.
            with ctx.server(model_dir.parent, image=image, target=target):
                await ctx.parity(model_dir.parent, model_dir)
                for request_batch in batch_ladder:
                    tiled = tile_batch(batch, request_batch)
                    request = build_request(settings, tiled)
                    case = base_case(
                        "hardware",
                        model,
                        device=device,
                        image=image,
                        request_batch=request_batch,
                        concurrency=concurrency,
                    )
                    result = run_closed(ctx, model, request, concurrency)
                    case["result"] = result.summary()
                    arm_results[(device, request_batch)] = case["result"]
                    cases.append(case)

        # Comparison matrix: GPU arm vs CPU arm per request batch
        for request_batch in batch_ladder:
            cpu = arm_results.get(("cpu", request_batch))
            gpu = arm_results.get(("gpu", request_batch))
            if not (cpu and gpu):
                continue
            speedup = (
                gpu["throughput_rps"] / cpu["throughput_rps"]
                if cpu.get("throughput_rps")
                else None
            )
            latency_delta = (
                gpu["latency_ms_p50"] - cpu["latency_ms_p50"]
                if gpu.get("latency_ms_p50") is not None
                and cpu.get("latency_ms_p50") is not None
                else None
            )
            case = base_case(
                "hardware",
                model,
                comparison="gpu-image-on-gpu vs cpu-image-on-cpu",
                request_batch=request_batch,
                speedup_throughput=round(speedup, 3) if speedup else None,
                p50_delta_ms=(
                    round(latency_delta, 3) if latency_delta is not None else None
                ),
            )
            case["result"] = {
                "cpu_image_on_cpu": {
                    "throughput_rps": cpu.get("throughput_rps"),
                    "latency_ms_p50": cpu.get("latency_ms_p50"),
                    "latency_ms_p99": cpu.get("latency_ms_p99"),
                },
                "gpu_image_on_gpu": {
                    "throughput_rps": gpu.get("throughput_rps"),
                    "latency_ms_p50": gpu.get("latency_ms_p50"),
                    "latency_ms_p99": gpu.get("latency_ms_p99"),
                },
                "speedup_throughput": round(speedup, 3) if speedup else None,
                "p50_delta_ms": (
                    round(latency_delta, 3) if latency_delta is not None else None
                ),
            }
            cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# Suite: faults (fault injection — deliberate malformed/dynamic traffic)
# ---------------------------------------------------------------------------


def _row_count_outcomes(
    responses: list[tuple[int, dict | None, str | None]], request_rows: int
) -> dict[str, int]:
    """Classify response row-count outcomes for fault cases.

    Pure function (unit-tested): counts exact matches, truncations
    (fewer rows than requested — the F-01 single-request signature),
    overflows (MORE rows than requested — the F-01 multi-request
    leakage signature), empties and transport failures.
    """
    exact = truncated = overflow = empty = failed = 0
    for status, payload, error in responses:
        if error is not None or payload is None or "outputs" not in payload:
            failed += 1
            continue
        data = payload["outputs"][0].get("data", []) if payload["outputs"] else []
        if len(data) == 0:
            empty += 1
        elif len(data) < request_rows:
            truncated += 1
        elif len(data) > request_rows:
            overflow += 1
        else:
            exact += 1
    return {
        "exact": exact,
        "truncated": truncated,
        "overflow_leakage": overflow,
        "empty": empty,
        "failed": failed,
    }


async def suite_faults(ctx: SuiteContext, model_dirs: list[Path]) -> list[Case]:
    """Fault injection: malformed and dynamic-shape requests against
    valid traffic.

    Two deliberate axes (the class of failures the parity gate caught
    by accident as F-01, made reproducible on purpose):

    1. declared dynamic [-1, N] request shapes x batching on/off —
       every response's output row count is validated against the
       request (silent truncation, cross-request leakage and empty
       responses are MEASURED outcomes, not assumed).
    2. poison-request resilience — one malformed request among valid
       traffic under active batching: does the server keep serving
       valid requests (batcher-kill regression) and recover after?
    """
    import aiohttp

    cases: list[Case] = []
    if not model_dirs:
        case = base_case("faults", "-", reason="no models")
        case["result"] = {"status": "SKIPPED_NO_MODELS"}
        return [case]

    model_dir = model_dirs[0]
    settings = json.loads((model_dir / "model-settings.json").read_text())
    model = settings["name"]
    rows = dict(np.load(model_dir / "validation.npz"))
    input_name = settings["inputs"][0]["name"]
    input_spec = settings["inputs"][0]
    request_rows = rows[input_spec["name"]].shape[0]

    def build(shape_mode: str) -> dict:
        shape = (
            list(rows[input_name].shape)
            if shape_mode == "concrete"
            else list(input_spec["shape"])
        )
        return {
            "inputs": [
                {
                    "name": input_name,
                    "datatype": input_spec["datatype"],
                    "shape": shape,
                    "data": rows[input_name].tolist(),
                }
            ],
            "outputs": [{"name": settings["outputs"][0]["name"]}],
        }

    base = f"http://127.0.0.1:{REST_PORT}"

    async def send(
        session: aiohttp.ClientSession, payload: dict
    ) -> tuple[int, dict | None, str | None]:
        try:
            async with session.post(
                f"{base}/v2/models/{model}/infer",
                json=payload,
                headers={"Accept-Encoding": "identity"},
            ) as response:
                body = await response.read()
                parsed = None
                try:
                    parsed = json.loads(body)
                except ValueError:
                    pass
                return response.status, parsed, None
        except Exception as error:  # transport failure is an outcome
            return 0, None, str(error)[:120]

    # ---- Axis 1: declared dynamic shape x batching -----------------
    for batching in (False, True):
        if batching:
            repo = _patched_repo(ctx, model_dir, max_batch_size=16, max_batch_time=0.05)
            target_dir = repo / model_dir.name
        else:
            repo = model_dir.parent
            target_dir = model_dir
        with ctx.server(repo):
            problems = await ctx.parity_problems(repo, target_dir)
            for shape_mode in ("concrete", "dynamic"):
                case = base_case(
                    "faults",
                    model,
                    axis="declared-shape",
                    shape=shape_mode,
                    batching=batching,
                )
                parity_note = None
                if shape_mode == "dynamic" and problems:
                    # The parity gate flags the config (F-01) — record
                    # it as a finding AND still measure the corruption
                    # rates: quantifying the fault is this suite's job.
                    parity_note = {
                        "parity": "FAILED",
                        "problems": problems,
                    }

                async def run_shape(
                    shape_mode: str = shape_mode,
                ) -> dict[str, int]:
                    async with aiohttp.ClientSession() as session:
                        # sequential requests (single-request signature)
                        responses = [
                            await send(session, build(shape_mode)) for _ in range(20)
                        ]
                        # concurrent burst (multi-request merge signature)
                        burst = await asyncio.gather(
                            *[send(session, build(shape_mode)) for _ in range(8)]
                        )
                        return _row_count_outcomes(
                            responses + list(burst), request_rows
                        )

                outcomes = await run_shape()
                case["result"] = {
                    "request_rows": request_rows,
                    **outcomes,
                }
                if parity_note:
                    case["result"].update(parity_note)
                    if (
                        outcomes["truncated"]
                        or outcomes["overflow_leakage"]
                        or outcomes["empty"]
                    ):
                        case["result"]["status"] = "PARITY_FAILED"
                cases.append(case)

    # ---- Axis 2: poison request among valid traffic ----------------
    repo = _patched_repo(ctx, model_dir, max_batch_size=16, max_batch_time=0.05)
    with ctx.server(repo):
        poison = {
            "inputs": [
                {
                    "name": input_name,
                    "datatype": input_spec["datatype"],
                    "shape": [],  # malformed: empty shape
                    "data": [1.0, 2.0],
                }
            ]
        }

        async def poison_phase() -> dict[str, Any]:
            async with aiohttp.ClientSession() as session:
                # Baseline: valid requests succeed
                baseline = await asyncio.gather(
                    *[send(session, build("concrete")) for _ in range(5)]
                )
                baseline_ok = sum(1 for s, p, e in baseline if s == 200 and e is None)
                # Poison among concurrent valid traffic (same batch
                # window)
                valid_tasks = [
                    asyncio.create_task(send(session, build("concrete")))
                    for _ in range(8)
                ]
                await asyncio.sleep(0.01)  # land inside the window
                poison_status, _, poison_error = await send(session, poison)
                valid_during = await asyncio.gather(*valid_tasks)
                during_ok = sum(1 for s, p, e in valid_during if s == 200 and e is None)
                # Recovery: valid requests after the poison
                recovery = await asyncio.gather(
                    *[send(session, build("concrete")) for _ in range(5)]
                )
                recovery_ok = sum(1 for s, p, e in recovery if s == 200 and e is None)
                return {
                    "baseline_ok": baseline_ok,
                    "baseline_total": 5,
                    "poison_status": poison_status,
                    "poison_error": poison_error,
                    "valid_during_ok": during_ok,
                    "valid_during_total": 8,
                    "recovery_ok": recovery_ok,
                    "recovery_total": 5,
                }

        case = base_case("faults", model, axis="poison-among-valid", batching=True)
        case["result"] = await poison_phase()
        case["result"]["valid_traffic_unaffected"] = (
            case["result"]["valid_during_ok"] == case["result"]["valid_during_total"]
            and case["result"]["recovery_ok"] == case["result"]["recovery_total"]
        )
        cases.append(case)

    # ---- Axis 3: worker kill under load (1-of-1 vs 1-of-4) -----------
    # One inference worker is SIGKILLed mid-load at each pool size.
    # The audit's outage claim is about the 1-worker DEFAULT; the
    # contrast between pool sizes is the finding. Measures in-flight
    # failures, the readiness signal during the window (readiness
    # green while serving fails = silent outage) and recovery.
    for kill_workers in (1, 4):
        result = await _worker_kill_case(
            ctx, model_dir, model, build("concrete"), kill_workers
        )
        case = base_case(
            "faults", model, axis="worker-kill-under-load", workers=kill_workers
        )
        case["result"] = result
        cases.append(case)
    return cases


def _readiness_green_during_outage(
    during: list[tuple[float, int]], failed: int
) -> bool:
    """Silent-outage signature: readiness 200 while requests fail.

    Pure function (mutation-checked): unpacks the (ts, code) tuples —
    comparing the tuple itself to 200 was always False.
    """
    return len(during) > 0 and all(code == 200 for _, code in during) and failed > 0


async def _worker_kill_case(
    ctx: "SuiteContext",
    model_dir: Path,
    model: str,
    request: dict,
    workers: int,
) -> dict[str, Any]:
    """Kill one inference worker mid-load; measure the blast radius.

    Worker discovery targets REAL multiprocessing workers: their
    cmdline is `python -c "from multiprocessing.spawn import
    spawn_main; ..."`, which does NOT contain "mlserver" (the original
    implementation grepped for "mlserver", matched only the discovery
    shell itself, and the kill never landed — every "worker survived"
    result from it was invalid). The kill runs through `sh -c` (kill
    is a shell builtin; a bare `docker exec kill` returns rc 127),
    its return code is checked, and a positive control asserts the
    victim PID disappeared and a new worker PID appeared.
    """
    import subprocess as sp
    import threading

    import aiohttp

    base = f"http://127.0.0.1:{REST_PORT}"
    pick_cmd = (
        "for p in $(ls /proc | grep -E '^[0-9]+$'); do "
        "c=$(tr '\\0' ' ' < /proc/$p/cmdline 2>/dev/null); "
        'case "$c" in *spawn_main*) [ $p -ne $$ ] && echo $p;; esac; done'
    )

    with ctx.server(model_dir.parent, parallel_workers=workers) as container:

        def pick_workers() -> list[str]:
            result = sp.run(
                ["docker", "exec", container.name, "sh", "-c", pick_cmd],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return [p for p in result.stdout.split() if p.isdigit()]

        async with aiohttp.ClientSession() as session:
            readiness: list[tuple[float, int]] = []
            load_result: dict[str, Any] = {}

            def run_load() -> None:
                # error_threshold=1.0: the outage under test is
                # EXPECTED to fail requests — the loadgen guard must
                # not abort the measurement it is measuring.
                load_result["result"] = run_closed(
                    ctx, model, request, 8, error_threshold=1.0
                )

            load_thread = threading.Thread(target=run_load)
            load_thread.start()
            victim = ""
            kill_rc: int | None = None
            killed_at = 0.0
            started = time.perf_counter()
            while load_thread.is_alive():
                await asyncio.sleep(0.2)
                elapsed = time.perf_counter() - started
                try:
                    async with session.get(
                        f"{base}/v2/health/ready",
                        timeout=aiohttp.ClientTimeout(total=1),
                    ) as response:
                        readiness.append((round(elapsed, 2), response.status))
                except Exception:
                    readiness.append((round(elapsed, 2), 0))
                if not victim and elapsed > 3.0:
                    candidates = pick_workers()
                    if not candidates:
                        continue
                    victim = candidates[0]
                    kill = sp.run(
                        [
                            "docker",
                            "exec",
                            container.name,
                            "sh",
                            "-c",
                            f"kill -9 {victim}",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    kill_rc = kill.returncode
                    killed_at = round(elapsed, 2)
            load_thread.join()

            # ---- Positive control: the kill must have landed --------
            after_pids = pick_workers()
            if not victim or kill_rc != 0:
                return {
                    "status": "INVALID_KILL",
                    "reason": "no worker found or kill returned non-zero",
                    "kill_rc": kill_rc,
                    "victim": victim,
                }
            if victim in after_pids:
                return {
                    "status": "INVALID_KILL",
                    "reason": "victim PID still alive after kill",
                    "victim": victim,
                    "workers_after": after_pids,
                }
            respawned = bool(after_pids)

            # ---- Outage metrics from the sample timeline -------------
            result = load_result["result"]
            samples = result.samples
            failed_samples = [s for s in samples if not s.ok]
            outage_ms: float | None = None
            recovered_at_ns: int | None = None
            if failed_samples:
                first_fail = min(s.start_ns for s in failed_samples)
                last_fail = max(s.end_ns for s in failed_samples)
                later_ok = [
                    s.start_ns for s in samples if s.ok and s.start_ns > last_fail
                ]
                recovered_at_ns = min(later_ok) if later_ok else None
                outage_end = recovered_at_ns if recovered_at_ns else last_fail
                outage_ms = round((outage_end - first_fail) / 1e6, 1)
            status_histogram: dict[str, int] = {}
            for s in failed_samples:
                key = (s.status or "?")[:60]
                status_histogram[key] = status_histogram.get(key, 0) + 1

            summary = result.summary()
            not_ready = [(ts, code) for ts, code in readiness if code != 200]
            during = [
                (ts, code)
                for ts, code in readiness
                if killed_at <= ts <= (killed_at + (outage_ms or 0) / 1000.0)
            ]
            return {
                "status": "MEASURED",
                "victim_pid": victim,
                "killed_at_s": killed_at,
                "kill_rc": kill_rc,
                "workers_after": after_pids,
                "worker_respawned": respawned,
                "load_throughput_rps": summary.get("throughput_rps"),
                "load_ok": summary.get("ok"),
                "load_failed": summary.get("failed"),
                "load_error_rate": summary.get("error_rate"),
                "outage_duration_ms": outage_ms,
                "error_status_histogram": dict(
                    sorted(status_histogram.items(), key=lambda kv: -kv[1])[:5]
                ),
                "readiness_probes": len(readiness),
                "readiness_not_ready": len(not_ready),
                "readiness_during_outage": [code for _, code in during][:20],
                "readiness_green_during_outage": (
                    _readiness_green_during_outage(during, summary.get("failed") or 0)
                ),
            }


async def send_fault_request(
    session: Any, base: str, model: str, payload: dict
) -> tuple[int, dict | None, str | None]:
    """Shared fault-suite request sender (transport outcomes as data)."""
    try:
        async with session.post(
            f"{base}/v2/models/{model}/infer",
            json=payload,
            headers={"Accept-Encoding": "identity"},
        ) as response:
            body = await response.read()
            parsed = None
            try:
                parsed = json.loads(body)
            except ValueError:
                pass
            return response.status, parsed, None
    except Exception as error:
        return 0, None, str(error)[:120]


SUITES: dict[str, Callable[[SuiteContext, list[Path]], Awaitable[list[Case]]]] = {
    "protocol": suite_protocol,
    "batching": suite_batching,
    "workers": suite_workers,
    "caching": suite_caching,
    "multimodel": suite_multimodel,
    "lifecycle": suite_lifecycle,
    "saturation": suite_saturation,
    "processing": suite_processing,
    "framework": suite_framework,
    "middleware": suite_middleware,
    "codec": suite_codec,
    "grpc": suite_grpc,
    "hardware": suite_hardware,
    "faults": suite_faults,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patched_repo(ctx: SuiteContext, model_dir: Path, **patch: Any) -> Path:
    """Copy a model into a temp repo with patched model-settings."""
    repo = ctx.output_dir / f"repo-{ctx.next_name('patch')}"
    repo.mkdir(parents=True, exist_ok=True)
    target = repo / model_dir.name
    shutil.copytree(model_dir, target)
    settings_path = target / "model-settings.json"
    settings = json.loads(settings_path.read_text())
    settings.update(patch)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    return repo


def _patched_repo_extra(
    ctx: SuiteContext, model_dir: Path, extra: dict[str, Any]
) -> Path:
    """Copy a model with ``parameters.extra`` merged (framework settings).

    This is the mechanism by which model framework-level pydantic
    settings auto-resolve: e.g. OnnxSettings reads providers,
    provider_options, session_options, run_options, cuda_graph from
    ``parameters.extra`` (with MLSERVER_MODEL_ONNX_* env underneath).
    """

    repo = ctx.output_dir / f"repo-{ctx.next_name('extra')}"
    repo.mkdir(parents=True, exist_ok=True)
    target = repo / model_dir.name
    shutil.copytree(model_dir, target)
    settings_path = target / "model-settings.json"
    settings = json.loads(settings_path.read_text())
    parameters = settings.setdefault("parameters", {})
    merged_extra = dict(parameters.get("extra") or {})
    merged_extra.update(extra)
    parameters["extra"] = merged_extra
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    return repo


def _thread_env(
    config: RunConfig, workers: int, implementation: str = ""
) -> dict[str, str]:
    """Thread oversubscription axis: BLAS/OpenMP/ORT thread caps.

    Threads scale with the host (cores // workers), keeping total
    compute threads near the core count instead of oversubscribing
    N workers x all cores. ORT session options are only set for ONNX
    models (matched on the implementation, NOT config.target which is
    just "cpu"/"gpu" — the old check was dead code and the options
    were never applied).
    """
    import os

    cores = os.cpu_count() or 4
    threads = max(1, cores // max(workers, 1))
    env = {
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
    }
    if implementation.startswith("mlserver_onnx"):
        env["MLSERVER_MODEL_ONNX_SESSION_OPTIONS"] = json.dumps(
            {"intra_op_num_threads": threads, "inter_op_num_threads": 1}
        )
    return env


def _cache_supported() -> bool:
    """Probe the local MLServer source for a response-cache setting."""
    source = Path(__file__).resolve().parents[2] / "MLServer" / "mlserver"
    if not source.is_dir():
        return False
    for path in source.rglob("*.py"):
        try:
            if "cache_enabled" in path.read_text():
                return True
        except OSError:
            continue
    return False
