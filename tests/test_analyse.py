#!/usr/bin/env python3
"""Unit tests for analyse.py — replay parsing, events, metrics, and strategy."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.analyse import (
    GameLog,
    SnakeDebug,
    TurnData,
    _bar,
    _classify_strategy_p1,
    _classify_strategy_p2,
    _collect_segments,
    _extract_player_header,
    _infer_strategy_label,
    _is_initial_state,
    _parse_body_from_list,
    _parse_initial_state,
    _parse_moves,
    _parse_snake_debug,
    _parse_stderr_segment,
    _parse_stdout_segment,
    _parse_turn_state,
    _Segment,
    build_events,
    build_optimization_hints,
    manhattan,
    nearest_fruit_dist,
)

# ─────────────────────────────────────────────────────────────
# SnakeDebug
# ─────────────────────────────────────────────────────────────


class TestSnakeDebug:
    def test_head(self):
        sd = SnakeDebug(id=0, current_dir="UP", chosen_dir="RIGHT",
                        score=100, fall_distance=0, ate=False, collided=False,
                        projected_body=[(5, 3), (5, 4), (5, 5)])
        assert sd.head() == (5, 3)

    def test_length(self):
        sd = SnakeDebug(id=1, current_dir="UP", chosen_dir="UP",
                        score=0, fall_distance=0, ate=False, collided=False,
                        projected_body=[(1, 1), (1, 2)])
        assert sd.length() == 2

    def test_is_doomed_true(self):
        sd = SnakeDebug(id=0, current_dir="UP", chosen_dir="UP",
                        score=-(10**9), fall_distance=0, ate=False, collided=False,
                        projected_body=[])
        assert sd.is_doomed()

    def test_is_doomed_false(self):
        sd = SnakeDebug(id=0, current_dir="UP", chosen_dir="UP",
                        score=-100, fall_distance=0, ate=False, collided=False,
                        projected_body=[])
        assert not sd.is_doomed()


# ─────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────


class TestParseBodyFromList:
    def test_normal(self):
        result = _parse_body_from_list("(4, 7), (5, 8)")
        assert result == [(4, 7), (5, 8)]

    def test_no_spaces(self):
        result = _parse_body_from_list("(1,2),(3,4)")
        assert result == [(1, 2), (3, 4)]

    def test_invalid(self):
        result = _parse_body_from_list("not a body")
        assert result == []


class TestParseSnakeDebug:
    def test_valid_line(self):
        line = "0 UP RIGHT 150 0 True False [(5, 3), (5, 4), (5, 5)]"
        sd = _parse_snake_debug(line)
        assert sd is not None
        assert sd.id == 0
        assert sd.current_dir == "UP"
        assert sd.chosen_dir == "RIGHT"
        assert sd.score == 150
        assert sd.fall_distance == 0
        assert sd.ate is True
        assert sd.collided is False
        assert sd.head() == (5, 3)

    def test_negative_score(self):
        line = "2 DOWN LEFT -500 3 False True [(1, 1)]"
        sd = _parse_snake_debug(line)
        assert sd is not None
        assert sd.score == -500
        assert sd.fall_distance == 3
        assert sd.collided is True

    def test_invalid_line(self):
        assert _parse_snake_debug("not a debug line") is None
        assert _parse_snake_debug("") is None
        assert _parse_snake_debug("0 BADDIR UP 0 0 True False [(0,0)]") is None


class TestParseMoves:
    def test_single_move(self):
        result = _parse_moves("0 LEFT")
        assert result == {0: "LEFT"}

    def test_multiple_moves(self):
        result = _parse_moves("0 RIGHT;1 DOWN;2 UP")
        assert result == {0: "RIGHT", 1: "DOWN", 2: "UP"}

    def test_no_moves(self):
        result = _parse_moves("WAIT")
        assert result == {}


class TestParseTurnState:
    def test_fruits_and_snakes(self):
        state = ["2", "5 3", "8 7", "2", "0 1,2:1,3:1,4", "1 6,6:6,7"]
        fruits, snakes = _parse_turn_state(state)
        assert fruits == {(5, 3), (8, 7)}
        assert 0 in snakes
        assert snakes[0] == [(1, 2), (1, 3), (1, 4)]
        assert snakes[1] == [(6, 6), (6, 7)]

    def test_no_fruits(self):
        state = ["0", "1", "0 3,3:3,4"]
        fruits, snakes = _parse_turn_state(state)
        assert fruits == set()
        assert len(snakes) == 1

    def test_empty(self):
        fruits, snakes = _parse_turn_state([])
        assert fruits == set()
        assert snakes == {}


class TestIsInitialState:
    def test_true_for_grid(self):
        state = ["0", "10", "5", "..........", "..........", "..........", "..........", "##########"]
        assert _is_initial_state(state)

    def test_false_for_turn(self):
        state = ["2", "5 3", "8 7", "1", "0 1,2:1,3"]
        assert not _is_initial_state(state)

    def test_short_list(self):
        assert not _is_initial_state(["0", "10"])


class TestParseInitialState:
    def test_normal(self):
        state = [
            "0", "10", "5",
            "..........", "..........", "..........", "....##....", "##########",
            "2", "0", "1", "2", "3",
        ]
        pid, w, h, walls, spp, my_ids, opp_ids = _parse_initial_state(state)
        assert pid == 0
        assert w == 10
        assert h == 5
        assert (4, 3) in walls
        assert (5, 3) in walls
        assert spp == 2
        assert my_ids == [0, 1]
        assert opp_ids == [2, 3]


# ─────────────────────────────────────────────────────────────
# Segment parsing
# ─────────────────────────────────────────────────────────────


class TestCollectSegments:
    def test_stdout_stderr(self):
        lines = [
            "Sortie d'erreur :",
            "some debug",
            "Sortie standard :",
            "0 RIGHT",
            "5",
            "100",
        ]
        segments = _collect_segments(lines)
        assert len(segments) == 2
        assert segments[0].kind == "stderr"
        assert segments[1].kind == "stdout"

    def test_empty(self):
        assert _collect_segments([]) == []

    def test_no_markers(self):
        assert _collect_segments(["random line", "another"]) == []


class TestParseStdoutSegment:
    def test_with_moves_and_turn(self):
        seg = _Segment("stdout", ["0 RIGHT;1 UP", "5", "100"])
        moves, turn, total = _parse_stdout_segment(seg)
        assert moves == {0: "RIGHT", 1: "UP"}
        assert turn == 5
        assert total == 100

    def test_moves_only(self):
        seg = _Segment("stdout", ["0 LEFT"])
        moves, turn, total = _parse_stdout_segment(seg)
        assert moves == {0: "LEFT"}
        assert turn == 0
        assert total == 0

    def test_empty(self):
        seg = _Segment("stdout", [])
        moves, turn, total = _parse_stdout_segment(seg)
        assert moves == {}


class TestParseStderrSegment:
    def test_state_list(self):
        seg = _Segment("stderr", ["['2', '5 3', '8 7', '1', '0 1,2:1,3']"])
        state_lists, debug = _parse_stderr_segment(seg)
        assert len(state_lists) == 1
        assert state_lists[0] == ["2", "5 3", "8 7", "1", "0 1,2:1,3"]
        assert debug == {}

    def test_debug_lines(self):
        seg = _Segment("stderr", ["0 UP RIGHT 150 0 True False [(5, 3), (5, 4)]"])
        state_lists, debug = _parse_stderr_segment(seg)
        assert len(state_lists) == 0
        assert 0 in debug
        assert debug[0].score == 150


class TestExtractPlayerHeader:
    def test_with_header(self):
        lines = [
            "Rang Suffixe Pseudo",
            "1",
            "er",
            "Player_Alpha",
            "2",
            "eme",
            "Player_Beta",
            "Sortie d'erreur :",
        ]
        p1, p2, r1, r2 = _extract_player_header(lines)
        assert p1 == "Player_Alpha"
        assert p2 == "Player_Beta"
        assert r1 == 1
        assert r2 == 2

    def test_no_header(self):
        lines = ["Sortie d'erreur :", "data"]
        p1, p2, r1, r2 = _extract_player_header(lines)
        assert p1 == "Bot-1"
        assert p2 == "Bot-2"


# ─────────────────────────────────────────────────────────────
# Manhattan distance
# ─────────────────────────────────────────────────────────────


class TestManhattan:
    def test_same_point(self):
        assert manhattan((3, 5), (3, 5)) == 0

    def test_horizontal(self):
        assert manhattan((0, 0), (5, 0)) == 5

    def test_diagonal(self):
        assert manhattan((1, 2), (4, 6)) == 7


class TestNearestFruitDist:
    def test_with_fruits(self):
        assert nearest_fruit_dist((5, 5), {(3, 5), (10, 10)}) == 2

    def test_empty_fruits(self):
        assert nearest_fruit_dist((0, 0), set()) is None


# ─────────────────────────────────────────────────────────────
# Build events
# ─────────────────────────────────────────────────────────────


class TestBuildEvents:
    def _make_game(self) -> GameLog:
        return GameLog(
            p1_name="A", p2_name="B", p1_rank=1, p2_rank=2,
            width=10, height=5, walls=set(),
            p1_snake_ids=[0, 1], p2_snake_ids=[2, 3],
        )

    def test_death_event(self):
        game = self._make_game()
        # Turn 1: snake 0 alive; Turn 2: snake 0 gone
        game.turns = [
            TurnData(turn=1, total_turns=10, fruits=set(),
                     snakes_actual={0: [(1, 1)], 1: [(2, 2)], 2: [(3, 3)], 3: [(4, 4)]},
                     debug={}, p1_moves={}, p2_moves={}),
            TurnData(turn=2, total_turns=10, fruits=set(),
                     snakes_actual={1: [(2, 2)], 2: [(3, 3)], 3: [(4, 4)]},
                     debug={}, p1_moves={}, p2_moves={}),
        ]
        events = build_events(game)
        death_events = [e for e in events if e.kind == "death"]
        assert len(death_events) == 1
        assert death_events[0].snake_id == 0
        assert death_events[0].player == 1

    def test_eat_event_via_debug(self):
        game = self._make_game()
        sd = SnakeDebug(id=0, current_dir="UP", chosen_dir="RIGHT",
                        score=100, fall_distance=0, ate=True, collided=False,
                        projected_body=[(5, 3)])
        game.turns = [
            TurnData(turn=1, total_turns=10, fruits={(5, 3)},
                     snakes_actual={0: [(4, 3)], 2: [(8, 8)]},
                     debug={0: sd}, p1_moves={0: "RIGHT"}, p2_moves={}),
        ]
        events = build_events(game)
        eat_events = [e for e in events if e.kind == "eat" and e.player == 1]
        assert len(eat_events) == 1

    def test_collision_event(self):
        game = self._make_game()
        sd = SnakeDebug(id=0, current_dir="UP", chosen_dir="UP",
                        score=-50, fall_distance=0, ate=False, collided=True,
                        projected_body=[(5, 3)])
        game.turns = [
            TurnData(turn=3, total_turns=10, fruits=set(),
                     snakes_actual={0: [(5, 4)]},
                     debug={0: sd}, p1_moves={}, p2_moves={}),
        ]
        events = build_events(game)
        coll = [e for e in events if e.kind == "collision"]
        assert len(coll) == 1

    def test_doomed_event(self):
        game = self._make_game()
        sd = SnakeDebug(id=0, current_dir="UP", chosen_dir="LEFT",
                        score=-(10**9), fall_distance=0, ate=False, collided=False,
                        projected_body=[(2, 2)])
        game.turns = [
            TurnData(turn=5, total_turns=10, fruits=set(),
                     snakes_actual={0: [(3, 2)]},
                     debug={0: sd}, p1_moves={}, p2_moves={}),
        ]
        events = build_events(game)
        doomed = [e for e in events if e.kind == "doomed"]
        assert len(doomed) == 1

    def test_fall_event(self):
        game = self._make_game()
        sd = SnakeDebug(id=0, current_dir="UP", chosen_dir="UP",
                        score=50, fall_distance=4, ate=False, collided=False,
                        projected_body=[(5, 1)])
        game.turns = [
            TurnData(turn=1, total_turns=10, fruits=set(),
                     snakes_actual={0: [(5, 2)]},
                     debug={0: sd}, p1_moves={}, p2_moves={}),
        ]
        events = build_events(game)
        falls = [e for e in events if e.kind == "fall"]
        assert len(falls) == 1

    def test_no_events_empty_game(self):
        game = self._make_game()
        game.turns = []
        assert build_events(game) == []


# ─────────────────────────────────────────────────────────────
# Strategy classification
# ─────────────────────────────────────────────────────────────


class TestStrategyClassification:
    def _make_game_with_turns(self, n_turns: int, fruit_chase_rate: float = 0.5) -> GameLog:
        """Create a game with synthetic turns for strategy testing."""
        game = GameLog(
            p1_name="A", p2_name="B", p1_rank=1, p2_rank=2,
            width=20, height=10, walls=set(),
            p1_snake_ids=[0], p2_snake_ids=[1],
        )
        for t in range(1, n_turns + 1):
            # P1 debug: alternating directions, configurable fruit chase
            sd = SnakeDebug(
                id=0, current_dir="RIGHT", chosen_dir="RIGHT",
                score=100, fall_distance=0, ate=(t % 5 == 0), collided=False,
                projected_body=[(t + 1, 5), (t, 5), (t - 1, 5)],
            )
            fruits = {(t + 2, 5)} if fruit_chase_rate > 0.5 else {(0, 0)}
            game.turns.append(TurnData(
                turn=t, total_turns=n_turns,
                fruits=fruits,
                snakes_actual={0: [(t, 5), (t - 1, 5), (t - 2, 5)],
                               1: [(15, 5), (15, 6)]},
                debug={0: sd},
                p1_moves={0: "RIGHT"},
                p2_moves={1: "LEFT"},
            ))
        return game

    def test_classify_p1_returns_dict(self):
        game = self._make_game_with_turns(20)
        stats = _classify_strategy_p1(game)
        assert "eat_turns" in stats
        assert "avg_score" in stats
        assert "direction_counts" in stats

    def test_classify_p2_returns_dict(self):
        game = self._make_game_with_turns(20)
        stats = _classify_strategy_p2(game)
        assert "direction_counts" in stats
        assert "avg_fruit_chase_rate" in stats
        assert "avg_aggression_rate" in stats

    def test_infer_strategy_label_p1_greedy(self):
        stats = {"avg_fruit_chase_rate": 0.8, "avg_score": 200,
                 "doomed_turns": [], "fall_events": []}
        label = _infer_strategy_label(stats, True)
        assert "fruits" in label.lower() or "greedy" in label.lower()

    def test_infer_strategy_label_p1_tactical(self):
        stats = {"avg_fruit_chase_rate": 0.2, "avg_score": 50,
                 "doomed_turns": [], "fall_events": []}
        label = _infer_strategy_label(stats, True)
        assert "tactique" in label.lower() or "territoire" in label.lower()

    def test_infer_strategy_label_p2_aggressive(self):
        stats = {"avg_fruit_chase_rate": 0.3, "avg_aggression_rate": 0.7,
                 "direction_change_rate": 0.3}
        label = _infer_strategy_label(stats, False)
        assert "agressif" in label.lower()


# ─────────────────────────────────────────────────────────────
# Optimization hints
# ─────────────────────────────────────────────────────────────


class TestOptimizationHints:
    def _make_game(self) -> GameLog:
        game = GameLog(
            p1_name="A", p2_name="B", p1_rank=1, p2_rank=2,
            width=10, height=5, walls=set(),
            p1_snake_ids=[0], p2_snake_ids=[1],
        )
        game.turns = [
            TurnData(turn=1, total_turns=100, fruits={(5, 3)},
                     snakes_actual={0: [(4, 3)], 1: [(8, 3)]},
                     debug={}, p1_moves={}, p2_moves={}),
        ]
        return game

    def test_no_anomalies(self):
        game = self._make_game()
        events = build_events(game)
        stats = {
            "doomed_turns": [],
            "collision_turns": [],
            "fall_events": [],
            "eat_turns": [1, 2, 3],
            "avg_score": 150,
            "avg_fruit_chase_rate": 0.6,
            "score_series": [100, 120, 140, 160],
        }
        hints = build_optimization_hints(game, events, stats)
        # Should have lookahead hint (no doomed + positive score)
        assert any("LOOKAHEAD" in h for h in hints)

    def test_doomed_hint(self):
        game = self._make_game()
        events = build_events(game)
        stats = {
            "doomed_turns": [5, 10],
            "collision_turns": [],
            "fall_events": [],
            "eat_turns": [],
            "avg_score": -100,
            "avg_fruit_chase_rate": 0.5,
            "score_series": [-100, -200],
        }
        hints = build_optimization_hints(game, events, stats)
        assert any("PIEGE" in h for h in hints)

    def test_low_fruit_chase_hint(self):
        game = self._make_game()
        events = build_events(game)
        stats = {
            "doomed_turns": [],
            "collision_turns": [],
            "fall_events": [],
            "eat_turns": [],
            "avg_score": 50,
            "avg_fruit_chase_rate": 0.3,
            "score_series": [],
        }
        hints = build_optimization_hints(game, events, stats)
        assert any("FRUITS" in h for h in hints)


# ─────────────────────────────────────────────────────────────
# Bar helper
# ─────────────────────────────────────────────────────────────


class TestBar:
    def test_full(self):
        assert _bar(10, 10, 10) == "#" * 10

    def test_empty(self):
        assert _bar(0, 10, 10) == "." * 10

    def test_half(self):
        result = _bar(5, 10, 10)
        assert len(result) == 10
        assert result.count("#") == 5

    def test_zero_max(self):
        assert _bar(5, 0, 10) == "-" * 10
