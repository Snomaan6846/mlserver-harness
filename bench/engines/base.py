"""Engine interface: the contract every execution environment implements.

An Engine is WHERE benchmarks/diagnostics run. Two implementations:
  - DockerEngine: local containers (fast, CI-friendly)
  - ISVCEngine: RHOAI InferenceServices on OpenShift (production path)

All activities (benchmark, diagnose, reference, compare) are
engine-agnostic — they receive an Engine and use its interface.
"""

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ServerHandle:
    """A running server instance, managed by an Engine."""

    name: str
    engine: str  # "docker" | "isvc"
    rest_url: str = ""
    grpc_target: str = ""
    metrics_url: str = ""
    image: str = ""
    model_repo: str = ""
    env: dict[str, str] = field(default_factory=dict)
    # Engine-specific metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class Engine(abc.ABC):
    """Abstract execution environment.

    All activities use this interface regardless of where they run.
    Implementations: DockerEngine (bench/engines/docker.py),
    ISVCEngine (bench/engines/isvc.py).
    """

    # -- Server lifecycle ------------------------------------------------

    @abc.abstractmethod
    def start_server(
        self,
        name: str,
        image: str,
        model_repo: str,
        env: dict[str, str] | None = None,
        resources: dict[str, Any] | None = None,
        target: str = "cpu",
    ) -> ServerHandle:
        """Start a server with the given configuration.

        Returns a ServerHandle with URLs/targets for accessing it.
        """

    @abc.abstractmethod
    def stop_server(self, handle: ServerHandle) -> None:
        """Stop and clean up a server."""

    @abc.abstractmethod
    def wait_ready(self, handle: ServerHandle, timeout: float = 300) -> bool:
        """Wait for the server to report ready."""

    @abc.abstractmethod
    def get_server_pids(self, handle: ServerHandle) -> list[int]:
        """Get the PIDs of the server process(es)."""

    @abc.abstractmethod
    def wait_fresh_server(
        self,
        handle: ServerHandle,
        old_identifiers: list[str],
        timeout: float = 300,
    ) -> str:
        """Restart the server and wait for a fresh instance.

        Used for the fresh-pod/fresh-container discipline:
        capture old identifiers, patch config, wait for new instance,
        prove old is gone. Returns the new instance identifier.
        """

    # -- Command execution ------------------------------------------------

    @abc.abstractmethod
    def exec_command(
        self,
        handle: ServerHandle,
        command: list[str],
        timeout: float = 60,
    ) -> str:
        """Execute a command inside the server's context.

        For Docker: docker exec <container> <command>
        For ISVC: oc exec <pod> -- <command>
        """

    @abc.abstractmethod
    def get_logs(self, handle: ServerHandle) -> str:
        """Get the server's logs."""

    # -- Resource discovery ------------------------------------------------

    @abc.abstractmethod
    def get_pod_uids(self, handle: ServerHandle) -> list[str]:
        """Get unique identifiers for running server instances.

        For Docker: container IDs
        For ISVC: pod UIDs
        """

    @abc.abstractmethod
    def gpu_available(self) -> bool:
        """Check if a GPU is available on this engine's host."""

    @abc.abstractmethod
    def gpu_guard(self, handle: ServerHandle) -> None:
        """Verify CUDAExecutionProvider is actively executing.

        Raises GPUProviderFallback if the GPU is not being used
        (anti-silent-fallback guard).
        """

    # -- Network access -----------------------------------------------------

    def setup_network(self, handle: ServerHandle) -> None:
        """Set up network access to the server (if needed).

        Docker: ports are already mapped (no-op).
        ISVC: may start oc port-forward processes.
        """

    def teardown_network(self, handle: ServerHandle) -> None:
        """Clean up network access (port-forwards, etc.)."""

    # -- Context manager -----------------------------------------------------

    def server(
        self,
        name: str,
        image: str,
        model_repo: str,
        **kwargs: Any,
    ) -> "_ServerContext":
        """Context manager for a server lifecycle."""
        return _ServerContext(self, name, image, model_repo, **kwargs)


class _ServerContext:
    """Context manager: start server on enter, stop on exit."""

    def __init__(
        self,
        engine: Engine,
        name: str,
        image: str,
        model_repo: str,
        **kwargs: Any,
    ):
        self.engine = engine
        self.name = name
        self.image = image
        self.model_repo = model_repo
        self.kwargs = kwargs
        self.handle: ServerHandle | None = None

    def __enter__(self) -> ServerHandle:
        self.handle = self.engine.start_server(
            self.name, self.image, self.model_repo, **self.kwargs
        )
        self.engine.wait_ready(self.handle)
        self.engine.setup_network(self.handle)
        return self.handle

    def __exit__(self, *exc: Any) -> None:
        if self.handle:
            self.engine.teardown_network(self.handle)
            self.engine.stop_server(self.handle)


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------


def create_engine(
    engine_type: str,
    namespace: str = "mlserver-perf",
    **kwargs: Any,
) -> Engine:
    """Create an engine by type.

    Args:
        engine_type: "docker" or "cluster" (alias: "isvc")
        namespace: Kubernetes namespace (cluster engine only)
    """
    if engine_type in ("local", "docker", "podman", "container"):
        from .docker import ContainerEngine

        return ContainerEngine(**kwargs)
    elif engine_type in ("cluster", "isvc", "openshift"):
        from .isvc import ISVCEngine

        return ISVCEngine(namespace=namespace, **kwargs)
    else:
        raise ValueError(
            f"Unknown engine type: {engine_type!r} " f"(expected 'docker' or 'cluster')"
        )
