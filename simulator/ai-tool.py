#!/usr/bin/env python3
"""AI-oriented validator built on top of the local Snakebird simulator.

This validator focuses on deterministic, machine-readable evaluation of a candidate
bot against a baseline bot on a fixed scenario corpus. It adds:

* persistent subprocess isolation for bot execution;
* structured JSON reporting for agents;
* strict protocol checks around stdout/stderr and runtime failures;
* batch evaluation over fixed maps or generated seeds;
* optional seat swapping to reduce first-player bias.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import queue
import select
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NoReturn, Optional, Sequence

SIMULATOR_MODULE_PATH = Path(__file__).with_name("simulator.py")
DEFAULT_MAPS = Path(__file__).parent / "data" / "maps.txt"
DEFAULT_BASELINE = Path(__file__).resolve().parent.parent / "bots" / "explorer.py"
REPORT_SCHEMA_VERSION = "1.0"


if TYPE_CHECKING:
    class SnakebirdSimulatorBase:
        players: List[Any]
        summary: List[str]
        losses: List[int]
        grid: Any

        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

        def serialize_global_info_for(self, player: Any) -> List[str]: ...

        def deactivate_player(self, player: Any, message: str) -> None: ...

        def run(self, record: bool = False) -> Dict[str, Any]: ...


class CLIUsageError(ValueError):
    """Raised when CLI arguments are invalid without emitting argparse text output."""


class JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CLIUsageError(message)


def load_simulator_module():
    spec = importlib.util.spec_from_file_location("snakebird_local_simulator", SIMULATOR_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load simulator module from {SIMULATOR_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SIMULATOR_MODULE = load_simulator_module()
if not TYPE_CHECKING:
    SnakebirdSimulatorBase = _SIMULATOR_MODULE.SnakebirdSimulator
load_map_scenarios = _SIMULATOR_MODULE.load_map_scenarios
load_params = _SIMULATOR_MODULE.load_params


WORKER_CODE = r'''
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import traceback
from pathlib import Path


def make_reader(lines):
    iterator = iter(lines)
    return lambda: next(iterator)


def respond(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


bot_path = Path(sys.argv[1])
params = json.loads(sys.argv[2])
slot = int(sys.argv[3])
game = None

bootstrap_stdout = io.StringIO()
bootstrap_stderr = io.StringIO()
try:
    with contextlib.redirect_stdout(bootstrap_stdout), contextlib.redirect_stderr(bootstrap_stderr):
        digest = hashlib.md5(f"{bot_path}:{slot}".encode(), usedforsecurity=False).hexdigest()
        module_name = f"simulator_ai_bot_{slot}_{digest}"
        spec = importlib.util.spec_from_file_location(module_name, bot_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to import {bot_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "Game"):
            raise RuntimeError(f"{bot_path} does not export a Game class")
        resolved_params = dict(getattr(module, "META_PARAMS", {}))
        if params:
            resolved_params.update(params)
        game = module.Game(resolved_params)
    respond(
        {
            "ok": True,
            "event": "ready",
            "stdout": bootstrap_stdout.getvalue(),
            "stderr": bootstrap_stderr.getvalue(),
        }
    )
except Exception as exc:
    respond(
        {
            "ok": False,
            "event": "ready",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "stdout": bootstrap_stdout.getvalue(),
            "stderr": bootstrap_stderr.getvalue(),
        }
    )


for raw_line in sys.stdin:
    raw_line = raw_line.strip()
    if not raw_line:
        continue

    try:
        request = json.loads(raw_line)
        action = request["action"]

        if action == "shutdown":
            respond({"ok": True, "event": "shutdown"})
            break

        if game is None:
            raise RuntimeError("Bot is unavailable because bootstrap failed")

        if action == "initialize":
            with (
                contextlib.redirect_stdout(io.StringIO()) as stdout_buffer,
                contextlib.redirect_stderr(io.StringIO()) as stderr_buffer,
            ):
                game.load_initial_state(make_reader(request["lines"]))
            respond(
                {
                    "ok": True,
                    "event": "initialize",
                    "stdout": stdout_buffer.getvalue(),
                    "stderr": stderr_buffer.getvalue(),
                }
            )
            continue

        if action == "turn":
            update_stdout = io.StringIO()
            update_stderr = io.StringIO()
            with contextlib.redirect_stdout(update_stdout), contextlib.redirect_stderr(update_stderr):
                game.update(make_reader(request["lines"]))

            play_stdout = io.StringIO()
            play_stderr = io.StringIO()
            with contextlib.redirect_stdout(play_stdout), contextlib.redirect_stderr(play_stderr):
                game.play()

            respond(
                {
                    "ok": True,
                    "event": "turn",
                    "update_stdout": update_stdout.getvalue(),
                    "update_stderr": update_stderr.getvalue(),
                    "play_stdout_lines": play_stdout.getvalue().splitlines(),
                    "play_stderr": play_stderr.getvalue(),
                }
            )
            continue

        raise RuntimeError(f"Unsupported action: {action}")
    except Exception as exc:
        respond(
            {
                "ok": False,
                "event": request.get("action", "unknown") if "request" in locals() else "unknown",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
'''


@dataclass
class DiagnosticEvent:
    severity: str
    kind: str
    stage: str
    message: str
    turn: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "severity": self.severity,
            "kind": self.kind,
            "stage": self.stage,
            "message": self.message,
        }
        if self.turn is not None:
            payload["turn"] = self.turn
        if self.details:
            payload["details"] = self.details
        return payload


@dataclass(frozen=True)
class ValidationScenario:
    name: str
    seed: int
    global_lines: Optional[List[str]] = None
    frame_lines: Optional[List[str]] = None


@dataclass(frozen=True)
class MatchTask:
    scenario: ValidationScenario
    candidate_slot: int


class SubprocessBotAdapter:
    def __init__(
        self,
        path: Path,
        slot: int,
        params: Optional[Dict[str, Any]] = None,
        *,
        role: str,
        init_timeout_ms: int,
        turn_timeout_ms: int,
        strict_stdout: bool,
        fail_on_stderr: bool,
    ):
        self.path = path
        self.slot = slot
        self.role = role
        self.params = params or {}
        self.init_timeout_ms = init_timeout_ms
        self.turn_timeout_ms = turn_timeout_ms
        self.strict_stdout = strict_stdout
        self.fail_on_stderr = fail_on_stderr
        self.turn_index = 0
        self.total_turn_time_ms = 0
        self.max_turn_time_ms = 0
        self.init_time_ms = 0
        self.process: Optional[subprocess.Popen[str]] = None
        self.bootstrap_error: Optional[str] = None
        self.bootstrap_traceback: Optional[str] = None
        self.diagnostics: List[DiagnosticEvent] = []
        self._start_worker()

    def _record(
        self,
        severity: str,
        kind: str,
        stage: str,
        message: str,
        *,
        turn: Optional[int] = None,
        **details: Any,
    ) -> None:
        clean_details = {key: value for key, value in details.items() if value not in (None, "", [], {})}
        self.diagnostics.append(DiagnosticEvent(severity, kind, stage, message, turn=turn, details=clean_details))

    def _start_worker(self) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-c", WORKER_CODE, str(self.path), json.dumps(self.params), str(self.slot)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        started = time.perf_counter()
        try:
            response = self._await_response("bootstrap", self.init_timeout_ms)
        except Exception as exc:
            self.bootstrap_error = str(exc)
            self._record("fatal", "bootstrap-failed", "bootstrap", self.bootstrap_error)
            self._terminate_process()
            return
        finally:
            self.init_time_ms = int((time.perf_counter() - started) * 1000)

        self._handle_io_capture("bootstrap", response.get("stdout", ""), response.get("stderr", ""))
        if not response.get("ok", False):
            self.bootstrap_error = response.get("error", "bootstrap failed")
            self.bootstrap_traceback = response.get("traceback")
            self._record(
                "fatal",
                "bootstrap-failed",
                "bootstrap",
                self.bootstrap_error or "bootstrap failed",
                error_type=response.get("error_type"),
            )
            self._terminate_process()

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self.process is None:
            raise RuntimeError(f"{self.role} process is not available")
        return self.process

    def _read_line_with_timeout(self, stdout, timeout_s: float):
        result_queue: queue.Queue = queue.Queue()

        def _reader():
            try:
                result_queue.put(("line", stdout.readline()))
            except Exception as exc:
                result_queue.put(("error", exc))

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        try:
            kind, value = result_queue.get(timeout=timeout_s)
        except queue.Empty:
            return None, True  # timed out
        if kind == "error":
            raise value
        return value, False

    def _await_response(self, stage: str, timeout_ms: int) -> Dict[str, Any]:
        process = self._ensure_process()
        if process.stdout is None:
            raise RuntimeError(f"{self.role} stdout pipe is unavailable")

        timeout_s = max(timeout_ms, 1) / 1000.0

        # select.select doesn't work on Windows pipes; use threading fallback
        try:
            ready, _, _ = select.select([process.stdout], [], [], timeout_s)
            timed_out = not ready
            line = process.stdout.readline() if not timed_out else ""
        except (select.error, OSError, ValueError):
            line, timed_out = self._read_line_with_timeout(process.stdout, timeout_s)
            if line is None:
                timed_out = True

        if timed_out:
            self._record("fatal", "timeout", stage, f"{self.role} timed out after {timeout_ms} ms")
            self._terminate_process()
            raise RuntimeError(f"{self.role} timed out after {timeout_ms} ms during {stage}")

        if not line:
            stderr_text = ""
            if process.stderr is not None:
                stderr_text = process.stderr.read().strip()
            self._terminate_process()
            raise RuntimeError(f"{self.role} exited unexpectedly during {stage}: {stderr_text or 'no response'}")

        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            self._terminate_process()
            raise RuntimeError(f"Invalid worker response during {stage}: {line.strip()}") from exc

    def _request(self, stage: str, payload: Dict[str, Any], timeout_ms: int) -> Dict[str, Any]:
        process = self._ensure_process()
        if process.stdin is None:
            raise RuntimeError(f"{self.role} stdin pipe is unavailable")
        try:
            process.stdin.write(json.dumps(payload) + "\n")
            process.stdin.flush()
        except BrokenPipeError as exc:
            self._terminate_process()
            raise RuntimeError(f"{self.role} process terminated before handling {stage}") from exc

        return self._await_response(stage, timeout_ms)

    def _handle_io_capture(self, stage: str, stdout_text: str, stderr_text: str, *, turn: Optional[int] = None) -> None:
        if stdout_text.strip():
            self._record(
                "fatal" if self.strict_stdout else "warning",
                "unexpected-stdout",
                stage,
                f"{self.role} wrote to stdout outside the command channel",
                turn=turn,
                stdout=stdout_text.strip(),
            )
        if stderr_text.strip():
            self._record(
                "fatal" if self.fail_on_stderr else "warning",
                "stderr-output",
                stage,
                f"{self.role} wrote to stderr",
                turn=turn,
                stderr=stderr_text.strip(),
            )

    def initialize(self, global_lines: Sequence[str]) -> None:
        if self.bootstrap_error is not None:
            raise RuntimeError(self.bootstrap_error)

        started = time.perf_counter()
        response = self._request(
            "initialize",
            {"action": "initialize", "lines": list(global_lines)},
            self.init_timeout_ms,
        )
        self.init_time_ms += int((time.perf_counter() - started) * 1000)

        self._handle_io_capture("initialize", response.get("stdout", ""), response.get("stderr", ""))
        if not response.get("ok", False):
            self._record(
                "fatal",
                "initialize-failed",
                "initialize",
                response.get("error", "initialize failed"),
                error_type=response.get("error_type"),
            )
            self._terminate_process()
            raise RuntimeError(response.get("error", "initialize failed"))

        self._raise_if_fatal("initialize")

    def play_turn(self, frame_lines: Sequence[str]) -> str:
        if self.bootstrap_error is not None:
            raise RuntimeError(self.bootstrap_error)

        self.turn_index += 1
        started = time.perf_counter()
        response = self._request("turn", {"action": "turn", "lines": list(frame_lines)}, self.turn_timeout_ms)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self.total_turn_time_ms += elapsed_ms
        self.max_turn_time_ms = max(self.max_turn_time_ms, elapsed_ms)

        if not response.get("ok", False):
            self._record(
                "fatal",
                "turn-failed",
                "turn",
                response.get("error", "turn failed"),
                turn=self.turn_index,
                error_type=response.get("error_type"),
            )
            self._terminate_process()
            raise RuntimeError(response.get("error", "turn failed"))

        self._handle_io_capture(
            "update",
            response.get("update_stdout", ""),
            response.get("update_stderr", ""),
            turn=self.turn_index,
        )
        self._handle_io_capture("play", "", response.get("play_stderr", ""), turn=self.turn_index)

        output_lines = response.get("play_stdout_lines", [])
        if not output_lines:
            self._record(
                "fatal",
                "empty-output",
                "play",
                f"{self.role} produced no command line",
                turn=self.turn_index,
            )
            self._raise_if_fatal("play")
            return "WAIT"

        if len(output_lines) > 1:
            self._record(
                "fatal" if self.strict_stdout else "warning",
                "multiple-output-lines",
                "play",
                f"{self.role} produced {len(output_lines)} stdout lines instead of 1",
                turn=self.turn_index,
                lines=output_lines,
            )

        command = str(output_lines[0]).strip()
        if not command:
            self._record(
                "fatal",
                "blank-output",
                "play",
                f"{self.role} produced a blank command line",
                turn=self.turn_index,
            )

        self._raise_if_fatal("play")
        return command or "WAIT"

    def _raise_if_fatal(self, stage: str) -> None:
        for diagnostic in reversed(self.diagnostics):
            if diagnostic.severity == "fatal" and diagnostic.stage == stage:
                raise RuntimeError(diagnostic.message)

    def snapshot(self) -> Dict[str, Any]:
        fatal_count = sum(1 for event in self.diagnostics if event.severity == "fatal")
        warning_count = sum(1 for event in self.diagnostics if event.severity == "warning")
        return {
            "role": self.role,
            "path": str(self.path),
            "slot": self.slot,
            "init_time_ms": self.init_time_ms,
            "turns_seen": self.turn_index,
            "total_turn_time_ms": self.total_turn_time_ms,
            "max_turn_time_ms": self.max_turn_time_ms,
            "fatal_count": fatal_count,
            "warning_count": warning_count,
            "diagnostics": [event.to_dict() for event in self.diagnostics],
        }

    def _terminate_process(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=1)
        except Exception:
            pass
        finally:
            self.process = None

    def shutdown(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            if process.poll() is None:
                self._request("shutdown", {"action": "shutdown"}, 500)
        except Exception:
            pass
        finally:
            self._terminate_process()


class AISnakebirdSimulator(SnakebirdSimulatorBase):
    def __init__(
        self,
        bot_paths: Sequence[Path],
        seed: int,
        league_level: int,
        *,
        adapter_factory: Callable[[Path, int, Optional[Dict[str, Any]]], Any],
        max_turns: int = 200,
        initial_global_lines: Optional[Sequence[str]] = None,
        initial_frame_lines: Optional[Sequence[str]] = None,
        initial_losses: tuple[int, int] = (0, 0),
        bot_params: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ):
        super().__init__(
            bot_paths=bot_paths,
            seed=seed,
            league_level=league_level,
            max_turns=max_turns,
            initial_global_lines=initial_global_lines,
            initial_frame_lines=initial_frame_lines,
            initial_losses=initial_losses,
            bot_params=bot_params,
            adapter_factory=adapter_factory,
        )

    def send_global_info(self) -> None:
        for player in self.players:
            if not player.active:
                continue
            try:
                player.adapter.initialize(self.serialize_global_info_for(player))
            except Exception as exc:
                self.deactivate_player(player, f"Player {player.index} failed during initialization: {exc}")

    def close(self) -> None:
        for player in self.players:
            shutdown = getattr(player.adapter, "shutdown", None)
            if callable(shutdown):
                shutdown()


def parse_args() -> argparse.Namespace:
    parser = JSONArgumentParser(description="Validate a Snakebird bot for AI-agent workflows and emit JSON.")
    parser.add_argument("candidate", type=Path, help="Path to the candidate bot file")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help=f"Baseline bot (default: {DEFAULT_BASELINE})",
    )
    parser.add_argument("--maps", type=Path, help=f"Fixed map corpus to use (default: {DEFAULT_MAPS})")
    parser.add_argument(
        "--generated-maps",
        type=int,
        help="Generate N maps instead of loading a fixed map corpus",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used for generated maps")
    parser.add_argument("--league-level", type=int, default=4, help="League level for generated maps")
    parser.add_argument("--max-turns", type=int, default=200, help="Maximum turns per match")
    parser.add_argument("--weights", type=Path, help="Shared params file applied to both candidate and baseline")
    parser.add_argument("--candidate-weights", type=Path, help="Params file applied only to the candidate")
    parser.add_argument("--baseline-weights", type=Path, help="Params file applied only to the baseline")
    parser.add_argument(
        "--init-timeout-ms",
        type=int,
        default=3000,
        help="Timeout for import/init/global-state loading",
    )
    parser.add_argument("--turn-timeout-ms", type=int, default=1000, help="Timeout per update+play turn")
    parser.add_argument(
        "--allow-extra-stdout",
        action="store_true",
        help="Do not fail when the bot prints extra stdout",
    )
    parser.add_argument("--fail-on-stderr", action="store_true", help="Treat any stderr output as fatal")
    parser.add_argument("--no-seat-swap", action="store_true", help="Run only with the candidate as player 0")
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=0.0,
        help="Minimum candidate win rate required for a passing verdict",
    )
    parser.add_argument("--json-output", type=Path, help="Optional path to write the JSON report")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args()


def load_validation_scenarios(args: argparse.Namespace) -> List[ValidationScenario]:
    if args.generated_maps is not None and args.generated_maps <= 0:
        raise ValueError("--generated-maps must be a positive integer")

    if args.generated_maps is not None:
        return [
            ValidationScenario(name=f"generated map {index + 1}", seed=args.seed + index)
            for index in range(args.generated_maps)
        ]

    maps_path = args.maps or DEFAULT_MAPS
    if not maps_path.is_file():
        raise FileNotFoundError(f"Map file not found: {maps_path}")

    return [
        ValidationScenario(
            name=scenario.name,
            seed=args.seed,
            global_lines=list(scenario.global_lines),
            frame_lines=list(scenario.frame_lines),
        )
        for scenario in load_map_scenarios(maps_path)
    ]


def build_bot_params(args: argparse.Namespace) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    shared = load_params(args.weights) if args.weights is not None else {}
    candidate = {**shared, **(load_params(args.candidate_weights) if args.candidate_weights is not None else {})}
    baseline = {**shared, **(load_params(args.baseline_weights) if args.baseline_weights is not None else {})}
    return candidate or None, baseline or None


def build_tasks(scenarios: Sequence[ValidationScenario], seat_swap: bool) -> List[MatchTask]:
    tasks = [MatchTask(scenario=scenario, candidate_slot=0) for scenario in scenarios]
    if seat_swap:
        tasks.extend(MatchTask(scenario=scenario, candidate_slot=1) for scenario in scenarios)
    return tasks


def outcome_from_winner(winner: Optional[int], candidate_slot: int) -> str:
    if winner is None:
        return "tie"
    return "win" if winner == candidate_slot else "loss"


def extract_summary_issues(summary_lines: Sequence[str], player_index: int) -> List[str]:
    candidates = [
        f"player {player_index}",
        f"Player {player_index}",
    ]
    return [line for line in summary_lines if any(token in line for token in candidates)]


def build_player_report(player: Any, summary_lines: Sequence[str]) -> Dict[str, Any]:
    adapter_snapshot = player.adapter.snapshot()
    summary_issues = extract_summary_issues(summary_lines, player.index)
    protocol_ok = player.deactivate_reason is None and adapter_snapshot["fatal_count"] == 0 and not summary_issues
    return {
        "index": player.index,
        "path": str(player.adapter.path),
        "active": player.active,
        "score": player.score,
        "reason": player.deactivate_reason,
        "last_execution_time_ms": player.last_execution_time_ms,
        "protocol_ok": protocol_ok,
        "summary_issues": summary_issues,
        "adapter": adapter_snapshot,
    }


def build_unavailable_player_report(
    index: int,
    path: Path,
    role: str,
    reason: str,
    *,
    fatal_count: int = 0,
) -> Dict[str, Any]:
    return {
        "index": index,
        "path": str(path),
        "active": False,
        "score": -1,
        "reason": reason,
        "last_execution_time_ms": 0,
        "protocol_ok": False,
        "summary_issues": [reason],
        "adapter": {
            "role": role,
            "path": str(path),
            "slot": index,
            "init_time_ms": 0,
            "turns_seen": 0,
            "total_turn_time_ms": 0,
            "max_turn_time_ms": 0,
            "fatal_count": fatal_count,
            "warning_count": 0,
            "diagnostics": [
                {
                    "severity": "fatal" if fatal_count else "warning",
                    "kind": "match-failed",
                    "stage": "run_match",
                    "message": reason,
                }
            ],
        },
    }


def mark_player_report_failed(report: Dict[str, Any], reason: str, *, fatal_count: int) -> Dict[str, Any]:
    adapter = dict(report["adapter"])
    adapter["fatal_count"] = max(int(adapter.get("fatal_count", 0)), fatal_count)
    diagnostics = list(adapter.get("diagnostics", []))
    diagnostics.append(
        {
            "severity": "fatal" if fatal_count else "warning",
            "kind": "match-failed",
            "stage": "run_match",
            "message": reason,
        }
    )
    adapter["diagnostics"] = diagnostics

    updated_report = dict(report)
    updated_report["active"] = False
    updated_report["protocol_ok"] = False
    updated_report["reason"] = updated_report.get("reason") or reason
    summary_issues = list(updated_report.get("summary_issues", []))
    if reason not in summary_issues:
        summary_issues.append(reason)
    updated_report["summary_issues"] = summary_issues
    updated_report["adapter"] = adapter
    return updated_report


def run_match(
    task: MatchTask,
    args: argparse.Namespace,
    candidate_params: Optional[Dict[str, Any]],
    baseline_params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
    scenario = task.scenario
    bot_paths = [args.baseline, args.baseline]
    bot_params: List[Optional[Dict[str, Any]]] = [baseline_params, baseline_params]
    bot_paths[task.candidate_slot] = args.candidate
    bot_params[task.candidate_slot] = candidate_params
    candidate_path = bot_paths[task.candidate_slot]
    baseline_path = bot_paths[1 - task.candidate_slot]

    simulator: Optional[AISnakebirdSimulator] = None
    started = time.perf_counter()

    def adapter_factory(path: Path, slot: int, params: Optional[Dict[str, Any]]) -> SubprocessBotAdapter:
        role = "candidate" if slot == task.candidate_slot else "baseline"
        return SubprocessBotAdapter(
            path=path,
            slot=slot,
            params=params,
            role=role,
            init_timeout_ms=args.init_timeout_ms,
            turn_timeout_ms=args.turn_timeout_ms,
            strict_stdout=not args.allow_extra_stdout,
            fail_on_stderr=args.fail_on_stderr,
        )

    try:
        simulator = AISnakebirdSimulator(
            bot_paths=bot_paths,
            seed=scenario.seed,
            league_level=args.league_level,
            adapter_factory=adapter_factory,
            max_turns=args.max_turns,
            initial_global_lines=scenario.global_lines,
            initial_frame_lines=scenario.frame_lines,
            bot_params=bot_params,
        )
        result = simulator.run()
        match_duration_ms = int((time.perf_counter() - started) * 1000)
        candidate_player = simulator.players[task.candidate_slot]
        baseline_player = simulator.players[1 - task.candidate_slot]
        candidate_report = build_player_report(candidate_player, result["summary"])
        baseline_report = build_player_report(baseline_player, result["summary"])
        return {
            "status": "ok",
            "scenario": scenario.name,
            "seed": scenario.seed,
            "candidate_slot": task.candidate_slot,
            "baseline_slot": 1 - task.candidate_slot,
            "turns": result["turns"],
            "match_duration_ms": match_duration_ms,
            "winner": result["winner"],
            "outcome": outcome_from_winner(result["winner"], task.candidate_slot),
            "scores": result["scores"],
            "losses": result["losses"],
            "remaining_apples": result["remaining_apples"],
            "summary": result["summary"],
            "candidate": candidate_report,
            "baseline": baseline_report,
            "error": None,
        }
    except Exception as exc:
        match_duration_ms = int((time.perf_counter() - started) * 1000)
        reason = f"Match execution failed: {exc}"
        error = classify_error("run_match", exc, args)
        error["details"] = {
            **error.get("details", {}),
            "scenario": scenario.name,
            "seed": scenario.seed,
            "candidate_slot": task.candidate_slot,
            "baseline_slot": 1 - task.candidate_slot,
        }

        if simulator is not None:
            summary_lines = list(simulator.summary)
            candidate_report = mark_player_report_failed(
                build_player_report(simulator.players[task.candidate_slot], summary_lines),
                reason,
                fatal_count=1,
            )
            baseline_report = mark_player_report_failed(
                build_player_report(simulator.players[1 - task.candidate_slot], summary_lines),
                reason,
                fatal_count=0,
            )
            losses = list(simulator.losses)
            remaining_apples = [apple.to_int_string() for apple in simulator.grid.apples]
        else:
            summary_lines = [reason]
            candidate_report = build_unavailable_player_report(
                task.candidate_slot,
                candidate_path,
                "candidate",
                reason,
                fatal_count=1,
            )
            baseline_report = build_unavailable_player_report(
                1 - task.candidate_slot,
                baseline_path,
                "baseline",
                reason,
                fatal_count=0,
            )
            losses = [0, 0]
            remaining_apples = []

        return {
            "status": "error",
            "scenario": scenario.name,
            "seed": scenario.seed,
            "candidate_slot": task.candidate_slot,
            "baseline_slot": 1 - task.candidate_slot,
            "turns": 0,
            "match_duration_ms": match_duration_ms,
            "winner": None,
            "outcome": "error",
            "scores": [candidate_report["score"], baseline_report["score"]],
            "losses": losses,
            "remaining_apples": remaining_apples,
            "summary": summary_lines,
            "candidate": candidate_report,
            "baseline": baseline_report,
            "error": error,
        }
    finally:
        if simulator is not None:
            simulator.close()


def aggregate_report(matches: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    completed_matches = [match for match in matches if match.get("status", "ok") == "ok"]
    match_failures = len(matches) - len(completed_matches)
    wins = sum(1 for match in completed_matches if match["outcome"] == "win")
    losses = sum(1 for match in completed_matches if match["outcome"] == "loss")
    ties = sum(1 for match in completed_matches if match["outcome"] == "tie")
    protocol_failures = sum(1 for match in matches if not match["candidate"]["protocol_ok"])
    inactive_matches = sum(1 for match in matches if not match["candidate"]["active"])
    fatal_events = sum(match["candidate"]["adapter"]["fatal_count"] for match in matches)
    warning_events = sum(match["candidate"]["adapter"]["warning_count"] for match in matches)
    total_turns = sum(int(match["turns"]) for match in completed_matches)
    total_score = sum(int(match["candidate"]["score"]) for match in completed_matches)
    total_score_against = sum(int(match["baseline"]["score"]) for match in completed_matches)
    win_rate = wins / max(len(completed_matches), 1)
    structural_pass = protocol_failures == 0 and inactive_matches == 0 and fatal_events == 0 and match_failures == 0
    competitive_pass = win_rate >= args.min_win_rate
    return {
        "matches": len(matches),
        "completed_matches": len(completed_matches),
        "match_failures": match_failures,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": win_rate,
        "protocol_failures": protocol_failures,
        "inactive_matches": inactive_matches,
        "fatal_events": fatal_events,
        "warning_events": warning_events,
        "total_turns": total_turns,
        "average_turns": total_turns / max(len(completed_matches), 1),
        "total_score": total_score,
        "total_score_against": total_score_against,
        "score_diff": total_score - total_score_against,
        "structural_pass": structural_pass,
        "competitive_pass": competitive_pass,
        "pass": structural_pass and competitive_pass,
    }


def build_validator_metadata() -> Dict[str, Any]:
    return {
        "name": "ai-tool",
        "simulator_module": str(SIMULATOR_MODULE_PATH),
    }


def build_config_snapshot(args: Optional[argparse.Namespace]) -> Dict[str, Any]:
    if args is None:
        return {
            "maps": None,
            "generated_maps": None,
            "seed": None,
            "league_level": None,
            "max_turns": None,
            "init_timeout_ms": None,
            "turn_timeout_ms": None,
            "seat_swap": None,
            "allow_extra_stdout": None,
            "fail_on_stderr": None,
            "min_win_rate": None,
        }
    return {
        "maps": str(args.maps or DEFAULT_MAPS) if args.generated_maps is None else None,
        "generated_maps": args.generated_maps,
        "seed": args.seed,
        "league_level": args.league_level,
        "max_turns": args.max_turns,
        "init_timeout_ms": args.init_timeout_ms,
        "turn_timeout_ms": args.turn_timeout_ms,
        "seat_swap": not args.no_seat_swap,
        "allow_extra_stdout": args.allow_extra_stdout,
        "fail_on_stderr": args.fail_on_stderr,
        "min_win_rate": args.min_win_rate,
    }


def default_summary(matches_count: int = 0) -> Dict[str, Any]:
    return {
        "matches": matches_count,
        "completed_matches": 0,
        "match_failures": 0,
        "wins": 0,
        "losses": 0,
        "ties": 0,
        "win_rate": 0.0,
        "protocol_failures": 0,
        "inactive_matches": 0,
        "fatal_events": 1,
        "warning_events": 0,
        "total_turns": 0,
        "average_turns": 0.0,
        "total_score": 0,
        "total_score_against": 0,
        "score_diff": 0,
        "structural_pass": False,
        "competitive_pass": False,
        "pass": False,
    }


def classify_error(stage: str, exc: Exception, args: Optional[argparse.Namespace]) -> Dict[str, Any]:
    details: Dict[str, Any] = {}
    retryable = False
    code = "internal_error"

    if isinstance(exc, CLIUsageError):
        code = "argument_parsing_failed"
    elif stage == "preflight" and args is not None and not args.candidate.is_file():
        code = "candidate_not_found"
        details["path"] = str(args.candidate)
    elif stage == "preflight" and args is not None and not args.baseline.is_file():
        code = "baseline_not_found"
        details["path"] = str(args.baseline)
    elif stage == "preflight":
        code = "invalid_arguments"
    elif stage == "load_scenarios":
        code = "scenario_loading_failed"
    elif stage == "build_params":
        code = "params_loading_failed"
    elif stage == "build_tasks":
        code = "task_build_failed"
    elif stage == "run_match":
        code = "match_execution_failed"
        retryable = True
    elif stage == "reporting":
        code = "report_build_failed"
    elif stage == "output":
        code = "report_output_failed"
        retryable = True
    elif isinstance(exc, FileNotFoundError):
        code = "file_not_found"
    elif isinstance(exc, ValueError):
        code = "invalid_arguments"

    return {
        "type": type(exc).__name__,
        "code": code,
        "stage": stage,
        "message": str(exc),
        "details": details,
        "retryable": retryable,
        "traceback": (
            None
            if isinstance(exc, (CLIUsageError, FileNotFoundError, ValueError))
            else traceback.format_exc()
        ),
    }


def build_error_report(
    args: Optional[argparse.Namespace],
    stage: str,
    exc: Exception,
    matches: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    summary = aggregate_report(matches, args) if args is not None and matches else default_summary(len(matches))
    summary["structural_pass"] = False
    summary["competitive_pass"] = False
    summary["pass"] = False
    summary["fatal_events"] = max(int(summary.get("fatal_events", 0)), 1)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "error",
        "validator": build_validator_metadata(),
        "candidate": str(args.candidate) if args is not None else None,
        "baseline": str(args.baseline) if args is not None else None,
        "config": build_config_snapshot(args),
        "summary": summary,
        "matches": list(matches),
        "error": classify_error(stage, exc, args),
    }


def emit_report(report: Dict[str, Any], json_output: Optional[Path], pretty: bool) -> None:
    json_text = json.dumps(report, indent=2 if pretty else None, sort_keys=False)
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json_text + "\n", encoding="utf-8")
    print(json_text)


def build_report(matches: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    summary = aggregate_report(matches, args)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "ok",
        "validator": build_validator_metadata(),
        "candidate": str(args.candidate),
        "baseline": str(args.baseline),
        "config": build_config_snapshot(args),
        "summary": summary,
        "matches": list(matches),
        "error": None,
    }


def main() -> int:
    args: Optional[argparse.Namespace] = None
    matches: List[Dict[str, Any]] = []
    stage = "parse_args"

    try:
        args = parse_args()

        stage = "preflight"
        if not args.candidate.is_file():
            raise FileNotFoundError(f"Candidate bot file not found: {args.candidate}")
        if not args.baseline.is_file():
            raise FileNotFoundError(f"Baseline bot file not found: {args.baseline}")
        if args.generated_maps is not None and args.maps is not None:
            raise ValueError("Use either --maps or --generated-maps, not both")
        if not 0.0 <= args.min_win_rate <= 1.0:
            raise ValueError("--min-win-rate must be between 0.0 and 1.0")

        stage = "load_scenarios"
        scenarios = load_validation_scenarios(args)

        stage = "build_params"
        candidate_params, baseline_params = build_bot_params(args)

        stage = "build_tasks"
        tasks = build_tasks(scenarios, seat_swap=not args.no_seat_swap)

        stage = "run_match"
        for task in tasks:
            matches.append(run_match(task, args, candidate_params, baseline_params))

        stage = "reporting"
        report = build_report(matches, args)

        stage = "output"
        emit_report(report, args.json_output, args.pretty)
        return 0 if report["summary"]["pass"] else 1
    except SystemExit:
        raise
    except Exception as exc:
        report = build_error_report(args, stage, exc, matches)
        emit_report(
            report,
            None if stage == "output" else getattr(args, "json_output", None),
            bool(getattr(args, "pretty", False)),
        )
        if isinstance(exc, (CLIUsageError, FileNotFoundError, ValueError)):
            return 2
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
