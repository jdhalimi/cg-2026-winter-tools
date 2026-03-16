import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AI_TOOL_PATH = ROOT / "simulator" / "ai-tool.py"


@pytest.fixture
def ai_tool_module():
    module_name = "test_tools_ai_tool_module"
    spec = importlib.util.spec_from_file_location(module_name, AI_TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)


def make_args(**overrides):
    defaults = {
        "candidate": ROOT / "bots" / "wait.py",
        "baseline": ROOT / "bots" / "explorer.py",
        "maps": ROOT / "simulator" / "data" / "maps.txt",
        "generated_maps": None,
        "seed": 0,
        "league_level": 4,
        "max_turns": 20,
        "weights": None,
        "candidate_weights": None,
        "baseline_weights": None,
        "init_timeout_ms": 100,
        "turn_timeout_ms": 100,
        "allow_extra_stdout": False,
        "fail_on_stderr": False,
        "no_seat_swap": True,
        "min_win_rate": 0.0,
        "json_output": None,
        "pretty": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_default_baseline_points_to_public_explorer(ai_tool_module):
    assert ai_tool_module.DEFAULT_BASELINE == ROOT / "bots" / "explorer.py"


def test_validator_metadata_uses_ai_tool_identity(ai_tool_module):
    metadata = ai_tool_module.build_validator_metadata()

    assert metadata["name"] == "ai-tool"
    assert metadata["simulator_module"] == str(ROOT / "simulator" / "simulator.py")


def test_main_emits_structured_json_error_for_missing_candidate(monkeypatch, ai_tool_module, tmp_path):
    captured = {}
    args = make_args(candidate=tmp_path / "missing.py")

    monkeypatch.setattr(ai_tool_module, "parse_args", lambda: args)
    monkeypatch.setattr(
        ai_tool_module,
        "emit_report",
        lambda report, json_output, pretty: captured.setdefault("report", report),
    )

    exit_code = ai_tool_module.main()

    assert exit_code == 2
    assert captured["report"]["schema_version"] == ai_tool_module.REPORT_SCHEMA_VERSION
    assert captured["report"]["status"] == "error"
    assert captured["report"]["error"]["code"] == "candidate_not_found"
    assert captured["report"]["summary"]["pass"] is False
    assert captured["report"]["matches"] == []


def test_run_match_returns_partial_error_record_when_simulator_fails(monkeypatch, ai_tool_module):
    class FailingSimulator:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(ai_tool_module, "AISnakebirdSimulator", FailingSimulator)

    task = ai_tool_module.MatchTask(
        scenario=ai_tool_module.ValidationScenario(name="failing scenario", seed=17),
        candidate_slot=0,
    )

    match = ai_tool_module.run_match(task, make_args(), None, None)

    assert match["status"] == "error"
    assert match["outcome"] == "error"
    assert match["error"]["code"] == "match_execution_failed"
    assert match["error"]["details"]["scenario"] == "failing scenario"
    assert match["candidate"]["protocol_ok"] is False
    assert match["candidate"]["adapter"]["fatal_count"] == 1
    assert match["baseline"]["adapter"]["fatal_count"] == 0


def test_build_report_exposes_success_envelope(ai_tool_module):
    report = ai_tool_module.build_report([], make_args())

    assert report["schema_version"] == ai_tool_module.REPORT_SCHEMA_VERSION
    assert report["status"] == "ok"
    assert report["validator"]["name"] == "ai-tool"
    assert report["baseline"] == str(ROOT / "bots" / "explorer.py")
    assert report["error"] is None
    assert report["summary"]["matches"] == 0
