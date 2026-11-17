"""Image resolution policy (the single source of truth).

Precedence matrix:

  LOCAL (Docker engine)
    - no image provided       -> engine default (floating odh-master tag),
                                 resolved to the latest sha256 at run start
    - override w/ tag         -> resolved to its sha256 at run start
    - override w/ @sha256:... -> used as-is (already immutable)

  CLUSTER (OpenShift engine, ISVC mode)
    - no image provided       -> the cluster's serving runtime template
                                 baked-in image is used AS-IS (product
                                 templates arrive digest-pinned) — the
                                 template has HIGHER precedence than any
                                 engine default
    - override w/ tag         -> that image, resolved to its sha256
    - override w/ @sha256:... -> used as-is

Rule: a benchmark NEVER runs a floating tag. Every tag resolves to its
immutable digest at run start; digest-pinned inputs pass through
untouched. The only case where no resolution happens is the ISVC inherit
(empty image) — because the cluster template's image is already pinned.
"""

from typing import Optional


def is_digest_pinned(reference: str) -> bool:
    """True when the reference is already immutable (repo@sha256:...)."""
    return "@sha256:" in reference


def resolve_image(
    provided: Optional[str],
    default: Optional[str] = None,
    resolver=None,
) -> str:
    """Apply the precedence policy and return the reference to use.

    provided:  the user override (IMAGE= / --image); empty/None = absent
    default:   the engine default — Docker: the floating odh-master tag;
               ISVC/cluster: None (inherit the serving runtime template's
               baked-in image)
    resolver:  callable(tag) -> digest-pinned reference (docker-based);
               required whenever a floating tag must be pinned

    Returns the digest-pinned reference, or "" when the caller should
    inherit the cluster template's image (ISVC mode, nothing provided,
    no default).
    """
    # 1. Explicit override always wins.
    if provided:
        if is_digest_pinned(provided):
            return provided  # already immutable — use as-is
        if resolver is None:
            raise ValueError(
                f"image {provided!r} is a floating tag but no resolver "
                "was supplied to pin it"
            )
        return resolver(provided)

    # 2. No override: engine default (local Docker) or template inherit
    #    (cluster ISVC — the template's baked-in image has higher
    #    precedence than any engine default).
    if not default:
        return ""  # ISVC inherit: use the serving runtime's image as-is
    if is_digest_pinned(default):
        return default
    if resolver is None:
        raise ValueError(
            f"default image {default!r} is a floating tag but no "
            "resolver was supplied to pin it"
        )
    return resolver(default)
