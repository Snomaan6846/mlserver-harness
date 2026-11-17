"""Image resolution policy tests — the precedence matrix from
bench/images.py, one test per row of the matrix table."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bench.images import is_digest_pinned, resolve_image  # noqa: E402


def fake_resolver(tag: str) -> str:
    """Deterministic resolver: <repo>/<name>:<tag> -> repo@sha256:<tag>."""
    repo = tag.rsplit(":", 1)[0]
    return f"{repo}@sha256:{tag.rsplit(':', 1)[1] or '0000'}"


class TestPrecedenceMatrix:
    # LOCAL (Docker engine)

    def test_local_no_image_default_resolves(self):
        result = resolve_image(
            None, "quay.io/opendatahub/mlserver:odh-master", resolver=fake_resolver
        )
        assert result == ("quay.io/opendatahub/mlserver@sha256:odh-master")

    def test_local_override_tag_resolves(self):
        result = resolve_image(
            "quay.io/syedali/mlserver:my-test",
            "quay.io/opendatahub/mlserver:odh-master",
            resolver=fake_resolver,
        )
        assert result == "quay.io/syedali/mlserver@sha256:my-test"

    def test_local_override_digest_used_as_is(self):
        pinned = "quay.io/syedali/mlserver@sha256:" + "a" * 64
        result = resolve_image(
            pinned, "quay.io/opendatahub/mlserver:odh-master", resolver=fake_resolver
        )
        assert result == pinned

    # CLUSTER (ISVC mode)

    def test_cluster_no_image_inherits_template(self):
        """No override + no default -> '' (inherit the serving runtime
        template's baked-in image, used as-is)."""
        result = resolve_image(None, None, resolver=fake_resolver)
        assert result == ""

    def test_cluster_override_tag_resolves(self):
        result = resolve_image(
            "quay.io/opendatahub/mlserver:odh-master", None, resolver=fake_resolver
        )
        assert result == "quay.io/opendatahub/mlserver@sha256:odh-master"

    def test_cluster_override_digest_used_as_is(self):
        pinned = "registry.redhat.io/rhoai/odh-mlserver-rhel9@sha256:" + "b" * 64
        result = resolve_image(pinned, None, resolver=fake_resolver)
        assert result == pinned

    # Guards

    def test_float_without_resolver_raises(self):
        with pytest.raises(ValueError, match="no resolver"):
            resolve_image("quay.io/x:y", None)

    def test_default_float_without_resolver_raises(self):
        with pytest.raises(ValueError, match="no resolver"):
            resolve_image(None, "quay.io/x:y")

    def test_empty_string_override_is_absent(self):
        """Empty string (IMAGE= passed but blank) counts as not provided."""
        result = resolve_image("", "quay.io/x:y", resolver=fake_resolver)
        assert result == "quay.io/x@sha256:y"


class TestIsDigestPinned:
    def test_tag_is_not_pinned(self):
        assert not is_digest_pinned("quay.io/repo:tag")

    def test_digest_is_pinned(self):
        assert is_digest_pinned("quay.io/repo@sha256:" + "c" * 64)

    def test_bare_sha_is_not_repo_pinned(self):
        assert not is_digest_pinned("sha256:" + "d" * 64)
