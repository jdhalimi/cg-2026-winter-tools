#!/usr/bin/env python3
"""Unit tests for display.py — game file parsing and terminal rendering."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import simulator.display as display
from simulator.display import (
    CYAN,
    RESET,
    col,
    load_games,
    parse_frame,
    parse_global,
    parse_list_line,
    render,
)

BOTS_DIR = Path(__file__).resolve().parent.parent / "bots"
MAPS_FILE = Path(__file__).resolve().parent.parent / "simulator" / "data" / "maps.txt"


# ─────────────────────────────────────────────────────────────
# col() — ANSI color helper
# ─────────────────────────────────────────────────────────────


class TestCol:
    def test_with_color(self):
        display._USE_COLOR = True
        result = col(CYAN, "hello")
        assert result.startswith(CYAN)
        assert result.endswith(RESET)
        assert "hello" in result

    def test_without_color(self):
        display._USE_COLOR = False
        result = col(CYAN, "hello")
        assert result == "hello"
        display._USE_COLOR = True  # restore


# ─────────────────────────────────────────────────────────────
# parse_list_line
# ─────────────────────────────────────────────────────────────


class TestParseListLine:
    def test_string_list(self):
        result = parse_list_line("['a', 'b', 'c']")
        assert result == ["a", "b", "c"]

    def test_mixed_types(self):
        result = parse_list_line("['0', '10', '5', '..........']")
        assert result[0] == "0"
        assert result[3] == ".........."


# ─────────────────────────────────────────────────────────────
# parse_global
# ─────────────────────────────────────────────────────────────


class TestParseGlobal:
    def test_normal(self):
        gl = [
            "0", "10", "5",
            "..........", "..........", "..........", "..........", "##########",
            "2", "0", "1", "2", "3",
        ]
        player_idx, width, height, grid_rows, my_ids, opp_ids = parse_global(gl)
        assert player_idx == 0
        assert width == 10
        assert height == 5
        assert len(grid_rows) == 5
        assert grid_rows[4] == "##########"
        assert my_ids == [0, 1]
        assert opp_ids == [2, 3]

    def test_player_1(self):
        gl = [
            "1", "8", "3",
            "........", "........", "########",
            "1", "3", "0",
        ]
        player_idx, width, height, grid_rows, my_ids, opp_ids = parse_global(gl)
        assert player_idx == 1
        assert my_ids == [3]
        assert opp_ids == [0]


# ─────────────────────────────────────────────────────────────
# parse_frame
# ─────────────────────────────────────────────────────────────


class TestParseFrame:
    def test_normal(self):
        fl = [
            "3",             # 3 apples
            "5 3", "8 7", "1 1",
            "2",             # 2 birds
            "0 5,2:5,3:5,4",
            "1 8,2:8,3",
        ]
        apples, birds = parse_frame(fl)
        assert apples == {(5, 3), (8, 7), (1, 1)}
        assert len(birds) == 2
        assert birds[0] == [(5, 2), (5, 3), (5, 4)]
        assert birds[1] == [(8, 2), (8, 3)]

    def test_no_apples(self):
        fl = ["0", "1", "0 3,3:3,4:3,5"]
        apples, birds = parse_frame(fl)
        assert apples == set()
        assert len(birds) == 1
        assert birds[0][0] == (3, 3)  # head

    def test_no_birds(self):
        fl = ["1", "5 5", "0"]
        apples, birds = parse_frame(fl)
        assert apples == {(5, 5)}
        assert len(birds) == 0


# ─────────────────────────────────────────────────────────────
# load_games — file parsing
# ─────────────────────────────────────────────────────────────


class TestLoadGames:
    def test_single_game(self, tmp_path):
        content = (
            "# test game\n"
            "['0', '4', '3', '....', '....', '####', '1', '0', '1']\n"
            "['1', '1 1', '1', '0 2,0:2,1:2,2']\n"
            "['1', '1 1', '1', '0 2,1:2,1:2,2']\n"
        )
        f = tmp_path / "game.txt"
        f.write_text(content)
        games = load_games(f)
        assert len(games) == 1
        name, global_lines, frames = games[0]
        assert name == "test game"
        assert len(frames) == 2

    def test_multiple_games(self, tmp_path):
        content = (
            "# game 1\n"
            "['0', '4', '3', '....', '....', '####', '1', '0', '1']\n"
            "['1', '1 1', '1', '0 2,0:2,1:2,2']\n"
            "\n"
            "# game 2\n"
            "['0', '4', '3', '....', '....', '####', '1', '0', '1']\n"
            "['1', '1 1', '1', '0 2,0:2,1:2,2']\n"
            "['0', '1', '0 2,1:2,2:2,3']\n"
        )
        f = tmp_path / "games.txt"
        f.write_text(content)
        games = load_games(f)
        assert len(games) == 2
        assert games[0][0] == "game 1"
        assert len(games[0][2]) == 1  # 1 frame
        assert games[1][0] == "game 2"
        assert len(games[1][2]) == 2  # 2 frames

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        games = load_games(f)
        assert games == []


# ─────────────────────────────────────────────────────────────
# render
# ─────────────────────────────────────────────────────────────


class TestRender:
    @pytest.fixture
    def setup(self):
        grid_rows = [
            "....",
            "....",
            "####",
        ]
        return grid_rows, 4, 3, [0], [1]

    def test_basic_render(self, setup):
        grid_rows, width, height, my_ids, opp_ids = setup
        apples: Set[Tuple[int, int]] = {(1, 0)}
        birds: Dict[int, List[Tuple[int, int]]] = {
            0: [(2, 0), (2, 1)],
            1: [(3, 0), (3, 1)],
        }
        result = render(grid_rows, width, height, my_ids, opp_ids, apples, birds, 1, 10)
        assert "Turn 1/10" in result
        assert "apples=1" in result
        assert "live birds=2" in result

    def test_render_no_birds(self, setup):
        grid_rows, width, height, my_ids, opp_ids = setup
        result = render(grid_rows, width, height, my_ids, opp_ids, set(), {}, 5, 20)
        assert "Turn 5/20" in result
        assert "live birds=0" in result

    def test_render_no_color(self, setup):
        display._USE_COLOR = False
        grid_rows, width, height, my_ids, opp_ids = setup
        result = render(grid_rows, width, height, my_ids, opp_ids, set(), {}, 1, 1)
        assert "\033" not in result
        display._USE_COLOR = True  # restore

    def test_render_with_apples_and_walls(self, setup):
        grid_rows, width, height, my_ids, opp_ids = setup
        apples: Set[Tuple[int, int]] = {(0, 0), (1, 1)}
        result = render(grid_rows, width, height, my_ids, opp_ids, apples, {}, 1, 1)
        # Apples rendered as '*'
        assert "*" in result


# ─────────────────────────────────────────────────────────────
# Integration: record + load + render
# ─────────────────────────────────────────────────────────────


class TestDisplayIntegration:
    def test_record_then_load(self, tmp_path):
        """Record a game with the simulator, then load it with display."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from simulator.simulator import SnakebirdSimulator, load_map_scenarios

        scenarios = load_map_scenarios(MAPS_FILE)
        s = scenarios[0]
        sim = SnakebirdSimulator(
            [BOTS_DIR / "wait.py", BOTS_DIR / "wait.py"],
            seed=0, league_level=4, max_turns=5,
            initial_global_lines=s.global_lines,
            initial_frame_lines=s.frame_lines,
        )
        result = sim.run(record=True)

        # Write recorded game to file
        gl = sim.serialize_global_info_for(sim.players[0])
        out = tmp_path / "recorded.txt"
        lines = [repr(gl)]
        for frame in result["recorded_frames"]:
            lines.append(repr(frame))
        lines.append("")
        out.write_text("\n".join(lines))

        # Load with display
        games = load_games(out)
        assert len(games) == 1
        name, global_lines, frames = games[0]
        assert len(frames) == len(result["recorded_frames"])

        # Parse and render first frame
        pi, w, h, grid_rows, my_ids, opp_ids = parse_global(global_lines)
        apples, birds = parse_frame(frames[0])
        rendered = render(grid_rows, w, h, my_ids, opp_ids, apples, birds, 1, len(frames))
        assert "Turn 1/" in rendered
        assert len(birds) > 0

    def test_render_all_frames(self, tmp_path):
        """Ensure every frame in a recorded game renders without error."""
        from simulator.simulator import SnakebirdSimulator

        sim = SnakebirdSimulator(
            [BOTS_DIR / "skeleton.py", BOTS_DIR / "wait.py"],
            seed=0, league_level=4, max_turns=10,
        )
        result = sim.run(record=True)

        gl = sim.serialize_global_info_for(sim.players[0])
        out = tmp_path / "game.txt"
        lines = [repr(gl)]
        for frame in result["recorded_frames"]:
            lines.append(repr(frame))
        lines.append("")
        out.write_text("\n".join(lines))

        games = load_games(out)
        name, global_lines, frames = games[0]
        pi, w, h, grid_rows, my_ids, opp_ids = parse_global(global_lines)

        for i, frame in enumerate(frames):
            apples, birds = parse_frame(frame)
            rendered = render(grid_rows, w, h, my_ids, opp_ids, apples, birds, i + 1, len(frames))
            assert "Turn" in rendered
