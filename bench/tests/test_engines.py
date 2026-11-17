"""Tests for the engine abstraction layer."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bench.engines.base import Engine, ServerHandle, create_engine  # noqa: E402
from bench.image_manager import (  # noqa: E402
    ACTIVITY_PACKAGES,
    ImageManager,
    ImageSource,
)


class TestEngineInterface:
    """The Engine abstract interface contract."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Engine()

    def test_create_engine_local(self):
        engine = create_engine("local")
        assert isinstance(engine, Engine)

    def test_create_engine_docker(self):
        engine = create_engine("docker")
        assert isinstance(engine, Engine)

    def test_create_engine_podman(self):
        engine = create_engine("podman")
        assert isinstance(engine, Engine)
        assert hasattr(engine, "start_server")
        assert hasattr(engine, "stop_server")
        assert hasattr(engine, "exec_command")

    def test_create_engine_cluster(self):
        engine = create_engine("cluster")
        assert isinstance(engine, Engine)

    def test_create_engine_isvc_alias(self):
        engine = create_engine("isvc")
        assert isinstance(engine, Engine)

    def test_create_engine_invalid(self):
        with pytest.raises(ValueError, match="Unknown engine"):
            create_engine("kubernetes")

    def test_server_handle_dataclass(self):
        handle = ServerHandle(
            name="test",
            engine="docker",
            rest_url="http://localhost:8080",
            grpc_target="localhost:8081",
        )
        assert handle.name == "test"
        assert handle.engine == "docker"
        assert handle.metadata == {}


class TestDockerEngine:
    """Docker adapter tests (mocked subprocess)."""

    @patch("bench.engines.docker._docker")
    def test_start_server(self, mock_docker):
        from bench.engines.docker import ContainerEngine as DockerEngine

        engine = DockerEngine()
        handle = engine.start_server("test", "mlserver:latest", "/tmp/models")
        assert handle.engine == "docker"
        # Caller's name is honored (restart reuses handle.name —
        # discarding it leaked the restarted container).
        assert handle.name == "test"
        assert "8080" in handle.rest_url
        assert mock_docker.called

    @patch("bench.engines.docker._docker")
    def test_start_server_fallback_name_is_run_scoped(self, mock_docker):
        """Empty name falls back to a run-scoped unique name: two
        engines (two runs) can never produce colliding containers."""
        from bench.engines.docker import ContainerEngine as DockerEngine

        engine_a = DockerEngine()
        engine_b = DockerEngine()
        handle_a = engine_a.start_server("", "mlserver:latest", "/tmp/models")
        handle_b = engine_b.start_server("", "mlserver:latest", "/tmp/models")
        assert handle_a.name.startswith("mlserver-bench-")
        assert handle_b.name.startswith("mlserver-bench-")
        assert (
            handle_a.name != handle_b.name
        ), "two engine instances produced the same container name"
        # Same engine: counter increments, no reuse
        handle_c = engine_a.start_server("", "mlserver:latest", "/tmp/models")
        assert handle_c.name != handle_a.name

    @patch("bench.engines.docker._docker")
    def test_exec_command(self, mock_docker):
        from bench.engines.docker import ContainerEngine as DockerEngine

        mock_docker.return_value = MagicMock(stdout="pid1\npid2\n")
        engine = DockerEngine()
        handle = ServerHandle(name="test", engine="docker")
        result = engine.exec_command(handle, ["pgrep", "mlserver"])
        assert "pid1" in result
        assert mock_docker.called


class TestImageManager:
    """Image sourcing and derived builder tests."""

    def test_activity_packages_mapping(self):
        assert ACTIVITY_PACKAGES["benchmark"] == []
        assert len(ACTIVITY_PACKAGES["diagnose"]) == 3
        assert "py-spy" in ACTIVITY_PACKAGES["diagnose"][0]

    def test_no_derived_for_benchmark(self):
        manager = ImageManager()
        assert not manager._needs_derived("benchmark")
        assert manager._needs_derived("diagnose")

    def test_cache_key_deterministic(self):
        manager = ImageManager()
        key1 = manager._derive_cache_key("mlserver@sha256:abc", ["py-spy", "memray"])
        key2 = manager._derive_cache_key("mlserver@sha256:abc", ["py-spy", "memray"])
        assert key1 == key2

    def test_cache_key_changes_with_base(self):
        manager = ImageManager()
        key1 = manager._derive_cache_key("mlserver@sha256:abc", ["py-spy"])
        key2 = manager._derive_cache_key("mlserver@sha256:def", ["py-spy"])
        assert key1 != key2

    def test_cache_key_changes_with_packages(self):
        manager = ImageManager()
        key1 = manager._derive_cache_key("mlserver@sha256:abc", ["py-spy"])
        key2 = manager._derive_cache_key("mlserver@sha256:abc", ["py-spy", "memray"])
        assert key1 != key2

    def test_override_passthrough_when_digest(self):
        manager = ImageManager(
            ImageSource(mode="override", override_ref="img@sha256:" + "a" * 64)
        )
        result = manager._pin_override("img@sha256:" + "a" * 64)
        assert result == "img@sha256:" + "a" * 64

    def test_image_source_dataclass(self):
        source = ImageSource(
            mode="source",
            repo_path="/path/to/MLServer",
            dockerfile="Dockerfile.cuda",
            registry="quay.io",
        )
        assert source.mode == "source"
        assert source.dockerfile == "Dockerfile.cuda"


class TestEngineInConfig:
    """The engine field flows through the run configuration."""

    def test_config_has_engine(self):
        from bench.config import RunConfig

        config = RunConfig()
        assert config.engine == "local"

    def test_config_engine_cluster(self):
        from bench.config import RunConfig

        config = RunConfig(engine="cluster")
        assert config.engine == "cluster"
