"""Local unit tests for the pure tool logic (no cluster required)."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import run as run_module  # noqa: E402
from checkpoint import Checkpoint  # noqa: E402
from run import (  # noqa: E402
    estimate_duration_seconds,
    referenced_models,
    validate_plan,
)


def materialize_plan_payloads(plan: dict, root: Path) -> None:
    """Create marker payload files for every name a plan references.

    Payload .bin files are input-tensor serializations regenerated
    before benchmarking (never committed — see payloads/README.md);
    validate_plan only needs the referenced NAMES to resolve under
    payloads/*/. The tests materialize markers in a throwaway tree
    (pytest's tmp_path cleans up), making them fully self-contained:
    no dependence on materialized artifacts, no skips."""
    payload_dir = root / "payloads" / "rfdetr-small"
    payload_dir.mkdir(parents=True, exist_ok=True)
    for entry in plan.get("variants", []):
        for payload in entry.get("spec", {}).get("payloads", []):
            (payload_dir / payload.rsplit("/", 1)[-1]).touch()


def make_plan(**overrides):
    plan = {
        "session": "test",
        "rounds": 3,
        "concurrency": [1, 8],
        "clients": 2,
        "seconds": 60,
        "warmup_seconds": 60,
        "warmup_requests": 8,
        "timeout_seconds": 120,
        "ready_timeout_seconds": 300,
        "prepare_timeout_seconds": 900,
        "max_start_lateness_ms": 20,
        "variants": [
            {
                "name": "a",
                "prepare_argv": ["python3", "/store/tools/hook.py", "va"],
                "spec": {
                    "protocol": "rest",
                    "url": "http://x/v2/models/m/infer",
                    "payloads": ["/store/payloads/j0-0.bin"],
                },
            }
        ],
    }
    plan.update(overrides)
    return plan


def make_variants():
    return {
        "va": {
            "deployment": "d",
            "service": "s",
            "device": "cuda",
            "expect_cuda": True,
            "image": "img",
        }
    }


class TestValidatePlan:
    def test_valid_plan_resolves_payloads(self, tmp_path, monkeypatch):
        plan = make_plan()
        materialize_plan_payloads(plan, tmp_path)
        monkeypatch.setattr(run_module, "REPO", tmp_path)
        uploads, problems = validate_plan(plan, make_variants(), None)
        assert not problems
        assert uploads == [str(tmp_path / "payloads" / "rfdetr-small" / "j0-0.bin")]

    def test_unknown_variant_is_a_problem(self, monkeypatch):
        monkeypatch.chdir(REPO)
        plan = make_plan()
        plan["variants"][0]["prepare_argv"] = [
            "python3",
            "/store/tools/hook.py",
            "nope",
        ]
        _, problems = validate_plan(plan, make_variants(), None)
        assert any("unknown variant" in p for p in problems)

    def test_missing_payload_is_a_problem(self, monkeypatch):
        monkeypatch.chdir(REPO)
        plan = make_plan()
        plan["variants"][0]["spec"]["payloads"] = ["/store/payloads/does-not-exist.bin"]
        _, problems = validate_plan(plan, make_variants(), None)
        assert any("does-not-exist" in p for p in problems)

    def test_missing_image_is_a_problem(self, monkeypatch):
        monkeypatch.chdir(REPO)
        variants = make_variants()
        del variants["va"]["image"]
        _, problems = validate_plan(make_plan(), variants, None)
        assert any("no image" in p for p in problems)

    def test_image_override_satisfies_images(self, monkeypatch):
        monkeypatch.chdir(REPO)
        variants = make_variants()
        del variants["va"]["image"]
        _, problems = validate_plan(make_plan(), variants, "override")
        assert not any("no image" in p for p in problems)

    def test_duplicate_variant_names_are_a_problem(self, monkeypatch):
        monkeypatch.chdir(REPO)
        plan = make_plan()
        plan["variants"] = plan["variants"] * 2
        _, problems = validate_plan(plan, make_variants(), None)
        assert any("duplicate" in p for p in problems)

    def test_missing_key_is_a_problem(self, monkeypatch):
        monkeypatch.chdir(REPO)
        plan = make_plan()
        del plan["rounds"]
        _, problems = validate_plan(plan, make_variants(), None)
        assert any("rounds" in p for p in problems)


class TestHelpers:
    def test_referenced_models_defaults_to_rfdetr(self):
        models = referenced_models(make_plan(), make_variants())
        assert models == ["rfdetr-small"]

    def test_referenced_models_reads_model_key(self):
        variants = make_variants()
        variants["va"]["model"] = "mlp-small"
        assert referenced_models(make_plan(), variants) == ["mlp-small"]

    def test_duration_estimate_scales_with_cases(self):
        plan = make_plan()
        small = estimate_duration_seconds(plan)
        plan["rounds"] = 6
        big = estimate_duration_seconds(plan)
        assert big > small
        assert small >= 600


class TestCheckpoint:
    def test_state_machine(self, tmp_path):
        state = tmp_path / "STATE.json"
        cp = Checkpoint(state)
        assert not cp.is_complete("d", "p")
        cp.mark_running("d", "p", cases=30)
        assert cp.entry("d", "p")["status"] == "running"
        cp.record("d", "p", trials=30, invalid=0, complete=True)
        assert cp.is_complete("d", "p")
        cp2 = Checkpoint(state)  # reload from disk
        assert cp2.is_complete("d", "p")
        assert cp2.pending_plans() == []
        cp2.mark_running("d", "q", cases=6)
        assert cp2.pending_plans() == ["d/q"]
        cp2.reset("d", "p")
        assert not cp2.is_complete("d", "p")


class TestGeneratedPlans:
    """Every generated plan must validate against its own variants."""

    @pytest.mark.parametrize(
        "domain,plan",
        [
            (d.name.replace("-", "-"), p.stem)
            for d in sorted((REPO / "benchmarks").iterdir())
            if (d / "plans").is_dir()
            for p in sorted((d / "plans").glob("*.json"))
        ],
    )
    def test_plan_validates(self, domain, plan, tmp_path, monkeypatch):
        plan_data = json.loads(
            (REPO / "benchmarks" / domain / "plans" / f"{plan}.json").read_text()
        )
        variants = json.loads(
            (REPO / "benchmarks" / domain / "variants.json").read_text()
        )
        materialize_plan_payloads(plan_data, tmp_path)
        monkeypatch.setattr(run_module, "REPO", tmp_path)
        uploads, problems = validate_plan(plan_data, variants, None)
        assert not problems, f"{domain}/{plan}: {problems}"
        assert uploads, f"{domain}/{plan}: no payload uploads resolved"
