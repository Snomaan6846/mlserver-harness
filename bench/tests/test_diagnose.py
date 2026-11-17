"""Tests for the diagnose tool: collectors, findings engine, report."""

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from bench.diagnose.collectors import system  # noqa: E402
from bench.diagnose.findings import engine  # noqa: E402
from bench.diagnose import report as report_mod  # noqa: E402
from bench.reference.measure import (  # noqa: E402
    compare_references,
    generate_reference_report,
)


class TestSystemCollector:
    def test_collect_all_returns_layers(self):
        data = system.collect_all(duration_s=0.1)
        assert "cpu" in data
        assert "memory" in data
        assert "io" in data
        assert "network" in data
        assert "interrupts" in data
        assert "thermal" in data

    def test_cpu_has_cores(self):
        cpu = system.collect_cpu(duration_s=0)
        assert cpu.get("cores", 0) > 0

    def test_memory_has_memtotal(self):
        mem = system.collect_memory()
        assert mem.get("memtotal_kb", 0) > 0

    def test_psi_parsing(self):
        text = "some avg10=1.50 avg60=2.00 avg300=1.00 total=12345\n"
        result = system._parse_psi(text)
        assert result["some_avg10"] == 1.50
        assert result["some_total"] == 12345


class TestFindingsEngine:
    def _make_data(self, **overrides):
        """Synthetic diagnostic data for testing rules."""
        data = {
            "system": {
                "cpu": {
                    "cores": 8,
                    "cpu_user_pct": 90,
                    "cpu_system_pct": 5,
                    "cpu_idle_pct": 5,
                    "cpu_iowait_pct": 0,
                    "context_switches_per_s": 10000,
                    "load_1m": 4.0,
                },
                "memory": {"memory_used_pct": 50, "swap_used_pct": 0},
            },
            "process": {
                "per_thread_cpu": {"total_cpu_pct": 100, "active_threads": []},
                "fd_count": {"fd_count": 100, "fd_limit": 65536, "fd_usage_pct": 0.2},
                "thread_states": {"thread_count": 10},
                "offcpu_snapshot": {"blocked_count": 2},
            },
            "gpu": {"error": "no GPU"},
            "oncpu": {},
            "latency": {},
            "memory": {},
        }
        data.update(overrides)
        return data

    def test_no_findings_on_healthy(self):
        findings = engine.evaluate(self._make_data())
        assert findings["total"] == 0

    def test_frontend_bound_detected(self):
        data = self._make_data()
        data["gpu"] = {
            "nvidia_smi": {"gpu_utilization_pct": 20},
        }
        data["process"]["per_thread_cpu"]["total_cpu_pct"] = 400
        data["oncpu"] = {"top_functions": [{"function": "json.loads", "cpu_pct": 60}]}
        findings = engine.evaluate(data)
        names = [f["finding"] for f in findings["findings"]]
        assert "frontend_bound_serving" in names

    def test_swap_detected(self):
        data = self._make_data()
        data["system"]["memory"]["swap_used_pct"] = 30
        findings = engine.evaluate(data)
        names = [f["finding"] for f in findings["findings"]]
        assert "swap_active" in names
        # CRITICAL should sort first
        assert findings["findings"][0]["severity"] == "CRITICAL"

    def test_gpu_underutilized_detected(self):
        data = self._make_data()
        data["gpu"] = {
            "nvidia_smi": {"gpu_utilization_pct": 30},
            "dcgm": {"sm_active_pct_mean": 30},
        }
        findings = engine.evaluate(data)
        names = [f["finding"] for f in findings["findings"]]
        assert "gpu_underutilized" in names

    def test_thermal_throttling_detected(self):
        data = self._make_data()
        data["gpu"] = {
            "nvidia_smi": {
                "gpu_utilization_pct": 80,
                "is_throttled": True,
                "throttle_detail": ["HW thermal cap"],
            },
        }
        findings = engine.evaluate(data)
        names = [f["finding"] for f in findings["findings"]]
        assert "thermal_throttling" in names

    def test_thread_thrashing_detected(self):
        data = self._make_data()
        data["system"]["cpu"]["context_switches_per_s"] = 200000
        findings = engine.evaluate(data)
        names = [f["finding"] for f in findings["findings"]]
        assert "thread_thrashing" in names

    def test_fd_exhaustion_detected(self):
        data = self._make_data()
        data["process"]["fd_count"] = {
            "fd_count": 60000,
            "fd_limit": 65536,
            "fd_usage_pct": 91.5,
        }
        findings = engine.evaluate(data)
        names = [f["finding"] for f in findings["findings"]]
        assert "fd_exhaustion_risk" in names

    def test_findings_sorted_by_severity(self):
        data = self._make_data()
        data["system"]["memory"]["swap_used_pct"] = 50  # CRITICAL
        data["system"]["cpu"]["context_switches_per_s"] = 200000  # MEDIUM
        findings = engine.evaluate(data)
        severities = [f["severity"] for f in findings["findings"]]
        assert severities == sorted(
            severities, key=lambda s: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}.get(s, 99)
        )

    def test_all_rules_have_required_fields(self):
        for rule in engine.RULES:
            assert "name" in rule
            assert "severity" in rule
            assert "action" in rule
            assert "check" in rule
            assert "description" in rule


class TestReportGenerator:
    def test_generates_diagnosis_md(self, tmp_path):
        data = self._sample_data()
        findings = engine.evaluate(data)
        report = report_mod.generate_report(data, findings, tmp_path)
        assert report.is_file()
        text = report.read_text()
        assert "MLServer Diagnostic Report" in text
        assert "Executive Summary" in text

    def test_diagnosis_json_written(self, tmp_path):
        data = self._sample_data()
        findings = engine.evaluate(data)
        report_mod.generate_report(data, findings, tmp_path)
        diagnosis = json.loads((tmp_path / "diagnosis.json").read_text())
        assert "findings" in diagnosis
        assert "layers" in diagnosis

    def _sample_data(self):
        return {
            "system": {
                "cpu": {
                    "cores": 8,
                    "cpu_user_pct": 40,
                    "cpu_system_pct": 10,
                    "cpu_idle_pct": 50,
                    "cpu_iowait_pct": 0,
                    "load_1m": 2,
                    "context_switches_per_s": 5000,
                },
                "memory": {"memory_used_pct": 60, "swap_used_pct": 0},
                "io": {},
            },
            "process": {
                "per_thread_cpu": {
                    "total_cpu_pct": 150,
                    "active_threads": [{"tid": 1, "cpu_pct": 100}],
                },
                "fd_count": {"fd_count": 50, "fd_limit": 65536, "fd_usage_pct": 0.1},
                "thread_states": {"thread_count": 5},
                "offcpu_snapshot": {"blocked_count": 1},
            },
            "oncpu": {
                "top_functions": [
                    {"function": "test_func", "cpu_pct": 50, "samples": 100}
                ]
            },
            "memory": {
                "rss_timeline": {
                    "samples": 10,
                    "rss_min_kb": 100000,
                    "rss_max_kb": 110000,
                    "rss_growth_rate_kb_per_s": 100,
                }
            },
            "latency": {
                "server_infer_ms_per_req": 50,
                "client_p50_ms": 80,
                "serving_overhead_ms": 30,
                "serving_overhead_pct": 37.5,
            },
            "gpu": {"error": "no GPU"},
        }


class TestReferenceComparison:
    def test_compare_references(self):
        floor = {
            "name": "raw-ort",
            "label": "Floor",
            "throughput_rps": 100,
            "latency_ms": {"p50": 10, "p99": 12},
        }
        mlserver = {
            "name": "mlserver",
            "label": "MLServer",
            "throughput_rps": 80,
            "latency_ms": {"p50": 12, "p99": 15},
        }
        ceiling = {
            "name": "triton",
            "label": "Ceiling",
            "throughput_rps": 90,
            "latency_ms": {"p50": 11, "p99": 13},
        }
        result = compare_references(floor, mlserver, ceiling)
        assert result["mlserver_efficiency_pct"] == 88.9
        assert result["serving_overhead_ms"] == 2.0
        assert result["optimization_headroom_rps"] == 10.0

    def test_reference_report(self, tmp_path):
        floor = {
            "name": "ort",
            "label": "Floor",
            "throughput_rps": 100,
            "latency_ms": {"p50": 10, "p99": 12},
        }
        mlserver = {
            "name": "mlserver",
            "label": "MLServer",
            "throughput_rps": 80,
            "latency_ms": {"p50": 12, "p99": 15},
        }
        ceiling = {
            "name": "triton",
            "label": "Ceiling",
            "throughput_rps": 90,
            "latency_ms": {"p50": 11, "p99": 13},
        }
        comparison = compare_references(floor, mlserver, ceiling)
        report = generate_reference_report(comparison, tmp_path)
        assert report.is_file()
        assert "Reference Comparison" in report.read_text()
