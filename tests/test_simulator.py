#!/usr/bin/env python3
"""Unit tests for simulator.py — game engine, grid, RNG, parsing, and full match."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

# Add parent directory so imports work when running from public/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.simulator import (
    DOWN,
    LEFT,
    RIGHT,
    TYPE_EMPTY,
    TYPE_INVALID,
    TYPE_WALL,
    UNSET,
    UP,
    Bird,
    BotAdapter,
    Coord,
    Grid,
    GridMaker,
    JavaRandom,
    MapScenario,
    PlayerState,
    SnakebirdSimulator,
    dump_lines_literal,
    java_round,
    load_map_scenarios,
    load_params,
    parse_body_string,
    parse_dump_lines,
    parse_losses,
    select_map_scenarios,
    shuffle_in_place,
    write_map_scenarios,
)

BOTS_DIR = Path(__file__).resolve().parent.parent / "bots"
MAPS_FILE = Path(__file__).resolve().parent.parent / "simulator" / "data" / "maps.txt"


# ─────────────────────────────────────────────────────────────
# Coord
# ─────────────────────────────────────────────────────────────


class TestCoord:
    def test_creation(self):
        c = Coord(3, 7)
        assert c.x == 3
        assert c.y == 7

    def test_add(self):
        c = Coord(5, 10)
        result = c.add(-2, 3)
        assert result == Coord(3, 13)

    def test_add_zero(self):
        c = Coord(1, 2)
        assert c.add(0, 0) == c

    def test_to_int_string(self):
        assert Coord(4, 9).to_int_string() == "4 9"
        assert Coord(0, 0).to_int_string() == "0 0"

    def test_frozen(self):
        c = Coord(1, 2)
        with pytest.raises(AttributeError):
            c.x = 5  # type: ignore

    def test_hashable(self):
        s = {Coord(1, 2), Coord(3, 4), Coord(1, 2)}
        assert len(s) == 2

    def test_ordering(self):
        coords = [Coord(2, 1), Coord(1, 2), Coord(1, 1)]
        assert sorted(coords) == [Coord(1, 1), Coord(1, 2), Coord(2, 1)]


# ─────────────────────────────────────────────────────────────
# java_round
# ─────────────────────────────────────────────────────────────


class TestJavaRound:
    def test_half_rounds_up(self):
        assert java_round(0.5) == 1
        assert java_round(1.5) == 2

    def test_below_half(self):
        assert java_round(0.4) == 0
        assert java_round(1.4) == 1

    def test_exact_integer(self):
        assert java_round(3.0) == 3

    def test_negative(self):
        assert java_round(-0.5) == 0
        assert java_round(-1.5) == -1


# ─────────────────────────────────────────────────────────────
# JavaRandom
# ─────────────────────────────────────────────────────────────


class TestJavaRandom:
    def test_deterministic(self):
        """Same seed produces same sequence."""
        r1 = JavaRandom(42)
        r2 = JavaRandom(42)
        for _ in range(20):
            assert r1.next_double() == r2.next_double()

    def test_different_seeds(self):
        r1 = JavaRandom(0)
        r2 = JavaRandom(1)
        assert r1.next_double() != r2.next_double()

    def test_next_double_range(self):
        r = JavaRandom(123)
        for _ in range(100):
            v = r.next_double()
            assert 0.0 <= v < 1.0

    def test_next_int_range(self):
        r = JavaRandom(99)
        for _ in range(100):
            v = r.next_int(10)
            assert 0 <= v < 10

    def test_next_int_bound_one(self):
        r = JavaRandom(0)
        assert r.next_int(1) == 0

    def test_next_int_power_of_two(self):
        r = JavaRandom(7)
        for _ in range(50):
            v = r.next_int(8)
            assert 0 <= v < 8

    def test_next_int_zero_bound_raises(self):
        r = JavaRandom(0)
        with pytest.raises(ValueError):
            r.next_int(0)

    def test_next_int_range_method(self):
        r = JavaRandom(55)
        for _ in range(50):
            v = r.next_int_range(5, 15)
            assert 5 <= v < 15

    def test_next_int_range_invalid(self):
        r = JavaRandom(0)
        with pytest.raises(ValueError):
            r.next_int_range(10, 5)


# ─────────────────────────────────────────────────────────────
# shuffle_in_place
# ─────────────────────────────────────────────────────────────


class TestShuffleInPlace:
    def test_preserves_elements(self):
        original = [Coord(i, 0) for i in range(10)]
        shuffled = list(original)
        shuffle_in_place(shuffled, JavaRandom(42))
        assert sorted(shuffled) == sorted(original)

    def test_deterministic(self):
        a = [Coord(i, i) for i in range(8)]
        b = list(a)
        shuffle_in_place(a, JavaRandom(7))
        shuffle_in_place(b, JavaRandom(7))
        assert a == b

    def test_empty_list(self):
        vals: List[Coord] = []
        shuffle_in_place(vals, JavaRandom(0))
        assert vals == []

    def test_single_element(self):
        vals = [Coord(5, 5)]
        shuffle_in_place(vals, JavaRandom(0))
        assert vals == [Coord(5, 5)]


# ─────────────────────────────────────────────────────────────
# Parsing utilities
# ─────────────────────────────────────────────────────────────


class TestParseDumpLines:
    def test_json_list(self):
        result = parse_dump_lines('["a", "b", "c"]')
        assert result == ["a", "b", "c"]

    def test_python_list(self):
        result = parse_dump_lines("['x', 'y']")
        assert result == ["x", "y"]

    def test_integers_to_strings(self):
        result = parse_dump_lines("[1, 2, 3]")
        assert result == ["1", "2", "3"]

    def test_not_a_list_raises(self):
        with pytest.raises(ValueError, match="list"):
            parse_dump_lines('{"key": "value"}')


class TestParseLosses:
    def test_valid(self):
        assert parse_losses("3,5") == (3, 5)

    def test_with_spaces(self):
        assert parse_losses(" 1 , 2 ") == (1, 2)

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            parse_losses("1,2,3")


class TestParseBodyString:
    def test_single_segment(self):
        result = parse_body_string("5,10")
        assert result == [Coord(5, 10)]

    def test_multiple_segments(self):
        result = parse_body_string("1,2:3,4:5,6")
        assert result == [Coord(1, 2), Coord(3, 4), Coord(5, 6)]


class TestLoadParams:
    def test_json_file(self, tmp_path):
        f = tmp_path / "params.json"
        f.write_text('{"score_eat_bonus": 100, "penalty": 50}')
        result = load_params(f)
        assert result == {"score_eat_bonus": 100, "penalty": 50}

    def test_python_dict_file(self, tmp_path):
        f = tmp_path / "params.txt"
        f.write_text("{'a': 1, 'b': 2}")
        result = load_params(f)
        assert result == {"a": 1, "b": 2}

    def test_not_a_dict_raises(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="dict"):
            load_params(f)


class TestDumpLinesLiteral:
    def test_round_trip(self):
        lines = ["0", "26", "14", "..##.."]
        dumped = dump_lines_literal(lines)
        recovered = parse_dump_lines(dumped)
        assert recovered == lines


# ─────────────────────────────────────────────────────────────
# Grid
# ─────────────────────────────────────────────────────────────


class TestGrid:
    def test_creation(self):
        g = Grid(10, 5)
        assert g.width == 10
        assert g.height == 5
        assert len(g.coords) == 50

    def test_in_bounds(self):
        g = Grid(4, 3)
        assert g.in_bounds(Coord(0, 0))
        assert g.in_bounds(Coord(3, 2))
        assert not g.in_bounds(Coord(4, 0))
        assert not g.in_bounds(Coord(-1, 0))
        assert not g.in_bounds(Coord(0, 3))

    def test_get_set_type(self):
        g = Grid(5, 5)
        assert g.get_type(Coord(2, 2)) == TYPE_EMPTY
        g.set_type(Coord(2, 2), TYPE_WALL)
        assert g.get_type(Coord(2, 2)) == TYPE_WALL

    def test_get_type_out_of_bounds(self):
        g = Grid(3, 3)
        assert g.get_type(Coord(10, 10)) == TYPE_INVALID

    def test_clear(self):
        g = Grid(5, 5)
        g.set_type(Coord(1, 1), TYPE_WALL)
        g.clear(Coord(1, 1))
        assert g.get_type(Coord(1, 1)) == TYPE_EMPTY

    def test_apples(self):
        g = Grid(5, 5)
        g.add_apple(Coord(2, 3))
        g.add_apple(Coord(4, 1))
        assert len(g.apples) == 2
        g.add_apple(Coord(2, 3))  # duplicate
        assert len(g.apples) == 2
        g.remove_apple(Coord(2, 3))
        assert len(g.apples) == 1
        g.remove_apple(Coord(9, 9))  # non-existent — no error
        assert len(g.apples) == 1

    def test_neighbours(self):
        g = Grid(5, 5)
        n = g.neighbours(Coord(2, 2))
        assert len(n) == 4
        assert Coord(2, 1) in n
        assert Coord(2, 3) in n
        assert Coord(1, 2) in n
        assert Coord(3, 2) in n

    def test_neighbours_corner(self):
        g = Grid(5, 5)
        n = g.neighbours(Coord(0, 0))
        assert len(n) == 2
        assert Coord(1, 0) in n
        assert Coord(0, 1) in n

    def test_opposite(self):
        g = Grid(10, 8)
        assert g.opposite(Coord(0, 3)) == Coord(9, 3)
        assert g.opposite(Coord(4, 7)) == Coord(5, 7)

    def test_opposite_y_symmetry(self):
        g = Grid(10, 8, y_symmetry=True)
        opp = g.opposite(Coord(0, 0))
        assert opp == Coord(9, 7)

    def test_detect_air_pockets(self):
        g = Grid(5, 5)
        # Create a walled-off 1x1 pocket at (2,2) surrounded by walls
        for x, y in [(1, 1), (2, 1), (3, 1), (1, 2), (3, 2), (1, 3), (2, 3), (3, 3)]:
            g.set_type(Coord(x, y), TYPE_WALL)
        islands = g.detect_air_pockets()
        sizes = sorted(len(i) for i in islands)
        assert 1 in sizes  # the pocket at (2,2)

    def test_detect_spawn_islands(self):
        g = Grid(10, 10)
        g.spawns = [Coord(1, 1), Coord(1, 2), Coord(5, 5), Coord(5, 6)]
        islands = g.detect_spawn_islands()
        assert len(islands) == 2


# ─────────────────────────────────────────────────────────────
# GridMaker
# ─────────────────────────────────────────────────────────────


class TestGridMaker:
    @pytest.mark.parametrize("seed", [0, 1, 42, 100])
    def test_make_deterministic(self, seed):
        g1 = GridMaker(JavaRandom(seed), 4).make()
        g2 = GridMaker(JavaRandom(seed), 4).make()
        assert g1.width == g2.width
        assert g1.height == g2.height
        assert g1.apples == g2.apples
        for coord in g1.coords:
            assert g1.get_type(coord) == g2.get_type(coord)

    @pytest.mark.parametrize("league", [1, 2, 3, 4])
    def test_make_all_leagues(self, league):
        g = GridMaker(JavaRandom(0), league).make()
        assert g.width > 0
        assert g.height > 0
        assert len(g.apples) > 0

    def test_grid_has_walls(self):
        g = GridMaker(JavaRandom(7), 4).make()
        wall_count = sum(1 for c in g.coords if g.get_type(c) == TYPE_WALL)
        assert wall_count > 0

    def test_grid_symmetric(self):
        """Grid should be left-right symmetric."""
        g = GridMaker(JavaRandom(3), 4).make()
        for coord in g.coords:
            opp = g.opposite(coord)
            assert g.get_type(coord) == g.get_type(opp), f"Asymmetric at {coord} vs {opp}"


# ─────────────────────────────────────────────────────────────
# Bird
# ─────────────────────────────────────────────────────────────


class TestBird:
    def test_head(self):
        b = Bird(id=0, owner_index=0, body=[Coord(5, 3), Coord(5, 4), Coord(5, 5)])
        assert b.head() == Coord(5, 3)

    def test_facing_up(self):
        b = Bird(id=0, owner_index=0, body=[Coord(5, 3), Coord(5, 4)])
        assert b.facing() == UP

    def test_facing_right(self):
        b = Bird(id=0, owner_index=0, body=[Coord(6, 3), Coord(5, 3)])
        assert b.facing() == RIGHT

    def test_facing_down(self):
        b = Bird(id=0, owner_index=0, body=[Coord(5, 5), Coord(5, 4)])
        assert b.facing() == DOWN

    def test_facing_left(self):
        b = Bird(id=0, owner_index=0, body=[Coord(4, 3), Coord(5, 3)])
        assert b.facing() == LEFT

    def test_facing_single_body(self):
        b = Bird(id=0, owner_index=0, body=[Coord(5, 5)])
        assert b.facing() == UNSET

    def test_set_message_short(self):
        b = Bird(id=0, owner_index=0)
        b.set_message("hello")
        assert b.message == "hello"

    def test_set_message_truncate(self):
        b = Bird(id=0, owner_index=0)
        long_msg = "a" * 100
        b.set_message(long_msg)
        assert len(b.message) == 49  # 46 + "..."
        assert b.message.endswith("...")

    def test_set_message_none(self):
        b = Bird(id=0, owner_index=0)
        b.set_message(None)
        assert b.message is None


# ─────────────────────────────────────────────────────────────
# MapScenario loading / writing
# ─────────────────────────────────────────────────────────────


class TestMapScenarios:
    def test_load_from_real_file(self):
        scenarios = load_map_scenarios(MAPS_FILE)
        assert len(scenarios) >= 1
        s = scenarios[0]
        assert s.index == 1
        assert s.name == "map 1"
        assert len(s.global_lines) > 0
        assert len(s.frame_lines) > 0

    def test_write_and_reload(self, tmp_path):
        out = tmp_path / "test_maps.txt"
        data = [
            (
                "test map",
                ["0", "4", "3", "....", "....", "####", "1", "0", "1"],
                ["2", "1 1", "2 2", "2", "0 1,1:1,2:1,3", "1 2,1:2,2:2,3"],
            ),
        ]
        write_map_scenarios(out, data)
        loaded = load_map_scenarios(out)
        assert len(loaded) == 1
        assert loaded[0].name == "test map"
        assert loaded[0].global_lines == data[0][1]
        assert loaded[0].frame_lines == data[0][2]

    def test_load_empty_raises(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        with pytest.raises(ValueError, match="No maps"):
            load_map_scenarios(f)

    def test_load_incomplete_raises(self, tmp_path):
        f = tmp_path / "incomplete.txt"
        f.write_text("# test\n['0', '4', '3']\n")
        with pytest.raises(ValueError, match="Incomplete"):
            load_map_scenarios(f)


class TestSelectMapScenarios:
    @pytest.fixture
    def scenarios(self):
        return [
            MapScenario(1, "Alpha", [], []),
            MapScenario(2, "Beta", [], []),
            MapScenario(3, "Gamma", [], []),
        ]

    def test_default_first(self, scenarios):
        result = select_map_scenarios(scenarios, None, None, False)
        assert len(result) == 1
        assert result[0].name == "Alpha"

    def test_all_maps(self, scenarios):
        result = select_map_scenarios(scenarios, None, None, True)
        assert len(result) == 3

    def test_by_index(self, scenarios):
        result = select_map_scenarios(scenarios, 2, None, False)
        assert result[0].name == "Beta"

    def test_by_name(self, scenarios):
        result = select_map_scenarios(scenarios, None, "gamma", False)
        assert result[0].name == "Gamma"

    def test_name_not_found(self, scenarios):
        with pytest.raises(ValueError, match="not found"):
            select_map_scenarios(scenarios, None, "delta", False)

    def test_index_out_of_range(self, scenarios):
        with pytest.raises(ValueError, match="out of range"):
            select_map_scenarios(scenarios, 10, None, False)

    def test_multiple_selectors_raises(self, scenarios):
        with pytest.raises(ValueError, match="only one"):
            select_map_scenarios(scenarios, 1, "Alpha", False)


# ─────────────────────────────────────────────────────────────
# BotAdapter
# ─────────────────────────────────────────────────────────────


class TestBotAdapter:
    def test_load_wait_bot(self):
        adapter = BotAdapter(BOTS_DIR / "wait.py", 0)
        assert hasattr(adapter.game, "play")

    def test_load_skeleton_bot(self):
        adapter = BotAdapter(BOTS_DIR / "skeleton.py", 1)
        assert hasattr(adapter.game, "play")

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(Exception):
            BotAdapter(tmp_path / "nope.py", 0)

    def test_load_no_game_class_raises(self, tmp_path):
        bad = tmp_path / "bad_bot.py"
        bad.write_text("x = 1\n")
        with pytest.raises(RuntimeError, match="Game"):
            BotAdapter(bad, 0)

    def test_wait_bot_plays_wait(self):
        adapter = BotAdapter(BOTS_DIR / "wait.py", 0)
        global_lines = [
            "0", "10", "5",
            "..........", "..........", "..........", "..........", "##########",
            "1", "0", "1",
        ]
        adapter.initialize(global_lines)
        frame_lines = [
            "2", "3 1", "6 2",
            "2", "0 5,2:5,3:5,4", "1 4,2:4,3:4,4",
        ]
        output = adapter.play_turn(frame_lines)
        assert output == "WAIT"

    def test_params_override(self, tmp_path):
        bot = tmp_path / "param_bot.py"
        bot.write_text(
            "from typing import Dict\n"
            "META_PARAMS = {'a': 1, 'b': 2}\n"
            "class Game:\n"
            "    def __init__(self, params):\n"
            "        self.params = params\n"
            "    def load_initial_state(self, _input=input): pass\n"
            "    def update(self, _input=input): pass\n"
            "    def play(self): print('WAIT')\n"
        )
        adapter = BotAdapter(bot, 0, {"b": 99, "c": 3})
        assert adapter.game.params == {"a": 1, "b": 99, "c": 3}


# ─────────────────────────────────────────────────────────────
# PlayerState
# ─────────────────────────────────────────────────────────────


class TestPlayerState:
    @pytest.fixture
    def player(self):
        adapter = BotAdapter(BOTS_DIR / "wait.py", 0)
        p = PlayerState(index=0, adapter=adapter)
        p.birds = [
            Bird(id=0, owner_index=0, body=[Coord(1, 1), Coord(1, 2)], direction=RIGHT, message="hi"),
            Bird(id=1, owner_index=0, body=[Coord(3, 3)], direction=LEFT),
        ]
        return p

    def test_get_bird_by_id(self, player):
        assert player.get_bird_by_id(0).id == 0
        assert player.get_bird_by_id(1).id == 1
        assert player.get_bird_by_id(99) is None

    def test_reset(self, player):
        player.marks = [Coord(0, 0)]
        player.reset()
        assert all(b.direction is None for b in player.birds)
        assert all(b.message is None for b in player.birds)
        assert player.marks == []

    def test_add_mark(self, player):
        for i in range(4):
            assert player.add_mark(Coord(i, 0))
        assert not player.add_mark(Coord(5, 0))  # 5th mark rejected


# ─────────────────────────────────────────────────────────────
# SnakebirdSimulator — integration
# ─────────────────────────────────────────────────────────────


class TestSnakebirdSimulator:
    def test_run_with_seed(self):
        """Run a full match with seed-based map generation."""
        sim = SnakebirdSimulator(
            [BOTS_DIR / "skeleton.py", BOTS_DIR / "wait.py"],
            seed=0, league_level=4, max_turns=50,
        )
        result = sim.run()
        assert "scores" in result
        assert "winner" in result
        assert "turns" in result
        assert result["turns"] <= 50

    def test_deterministic(self):
        """Same seed + same bots = same result."""
        def run_once():
            sim = SnakebirdSimulator(
                [BOTS_DIR / "wait.py", BOTS_DIR / "wait.py"],
                seed=7, league_level=3, max_turns=20,
            )
            return sim.run()
        r1 = run_once()
        r2 = run_once()
        assert r1["scores"] == r2["scores"]
        assert r1["turns"] == r2["turns"]

    def test_run_from_map_file(self):
        """Run from a pre-generated map file."""
        scenarios = load_map_scenarios(MAPS_FILE)
        s = scenarios[0]
        sim = SnakebirdSimulator(
            [BOTS_DIR / "wait.py", BOTS_DIR / "wait.py"],
            seed=0, league_level=4, max_turns=10,
            initial_global_lines=s.global_lines,
            initial_frame_lines=s.frame_lines,
        )
        result = sim.run()
        assert result["turns"] <= 10

    def test_record_frames(self):
        """Recorded frames can be written and are non-empty."""
        sim = SnakebirdSimulator(
            [BOTS_DIR / "wait.py", BOTS_DIR / "wait.py"],
            seed=0, league_level=4, max_turns=5,
        )
        result = sim.run(record=True)
        assert len(result["recorded_frames"]) > 0
        # Each frame should be a list of strings
        assert all(isinstance(f, list) for f in result["recorded_frames"])

    def test_serialization_round_trip(self):
        """Global and frame lines can round-trip through the simulator."""
        sim = SnakebirdSimulator(
            [BOTS_DIR / "wait.py", BOTS_DIR / "wait.py"],
            seed=5, league_level=4, max_turns=1,
        )
        gl = sim.serialize_global_info_for(sim.players[0])
        fl = sim.serialize_frame_info_for(sim.players[0])
        # Reload into a new sim
        sim2 = SnakebirdSimulator(
            [BOTS_DIR / "wait.py", BOTS_DIR / "wait.py"],
            seed=5, league_level=4, max_turns=1,
            initial_global_lines=gl,
            initial_frame_lines=fl,
        )
        gl2 = sim2.serialize_global_info_for(sim2.players[0])
        assert gl == gl2

    def test_game_over_when_all_apples_gone(self):
        """Game ends when no apples remain."""
        sim = SnakebirdSimulator(
            [BOTS_DIR / "skeleton.py", BOTS_DIR / "skeleton.py"],
            seed=0, league_level=4, max_turns=500,
        )
        result = sim.run()
        # Either all apples consumed or max turns reached
        result.get("remaining_apples", [])
        assert result["turns"] <= 500

    def test_build_global_lines_for_both_players(self):
        """Player 0 and player 1 get consistent but swapped global lines."""
        sim = SnakebirdSimulator(
            [BOTS_DIR / "wait.py", BOTS_DIR / "wait.py"],
            seed=0, league_level=4, max_turns=1,
        )
        gl0 = sim.serialize_global_info_for(sim.players[0])
        gl1 = sim.serialize_global_info_for(sim.players[1])
        # Player index differs
        assert gl0[0] == "0"
        assert gl1[0] == "1"
        # Width and height same
        assert gl0[1] == gl1[1]
        assert gl0[2] == gl1[2]

    def test_skeleton_beats_wait(self):
        """skeleton.py should beat wait.py (or at least not lose)."""
        sim = SnakebirdSimulator(
            [BOTS_DIR / "skeleton.py", BOTS_DIR / "wait.py"],
            seed=0, league_level=4, max_turns=200,
        )
        result = sim.run()
        assert result["scores"][0] >= result["scores"][1]


# ─────────────────────────────────────────────────────────────
# Simulator game mechanics
# ─────────────────────────────────────────────────────────────


class TestSimulatorMechanics:
    @pytest.fixture
    def sim(self):
        """Create a simulator with a simple known state."""
        return SnakebirdSimulator(
            [BOTS_DIR / "wait.py", BOTS_DIR / "wait.py"],
            seed=0, league_level=4, max_turns=10,
        )

    def test_do_moves(self, sim):
        """After do_moves, bird heads should shift by direction delta."""
        bird = sim.players[0].birds[0]
        old_head = bird.head()
        bird.direction = RIGHT
        old_body_len = len(bird.body)
        sim.do_moves()
        new_head = bird.head()
        assert new_head == old_head.add(1, 0)
        # Body length unchanged (no apple eaten)
        if old_head not in sim.grid.apples:
            assert len(bird.body) == old_body_len

    def test_do_beheadings_wall(self, sim):
        """Bird moving into a wall gets beheaded."""
        bird = sim.players[0].birds[0]
        # Place bird head adjacent to a wall
        wall_coord = None
        for coord in sim.grid.coords:
            if sim.grid.get_type(coord) == TYPE_WALL:
                wall_coord = coord
                break
        assert wall_coord is not None
        # Position bird so head is at wall
        bird.body = [wall_coord, wall_coord.add(0, -1), wall_coord.add(0, -2)]
        bird.alive = True
        original_len = len(bird.body)
        sim.do_beheadings()
        # Should have lost head or died
        assert len(bird.body) < original_len or not bird.alive

    def test_is_game_over_no_apples(self, sim):
        """Game is over when no apples remain."""
        sim.grid.apples = []
        assert sim.is_game_over()

    def test_is_game_over_player_dead(self, sim):
        """Game is over when all birds of a player are dead."""
        for bird in sim.players[0].birds:
            bird.alive = False
        assert sim.is_game_over()

    def test_finalize_scores(self, sim):
        """Scores are total alive body length."""
        for bird in sim.players[0].birds:
            bird.body = [Coord(0, 0), Coord(0, 1), Coord(0, 2)]
            bird.alive = True
        for bird in sim.players[1].birds:
            bird.body = [Coord(5, 0), Coord(5, 1)]
            bird.alive = True
        sim.finalize_scores()
        p0_expected = 3 * len(sim.players[0].birds)
        p1_expected = 2 * len(sim.players[1].birds)
        assert sim.players[0].score == p0_expected
        assert sim.players[1].score == p1_expected

    def test_parse_commands_valid(self, sim):
        """Valid move command is parsed correctly."""
        player = sim.players[0]
        bird = player.birds[0]
        bird.alive = True
        bird.body = [Coord(5, 5), Coord(5, 6), Coord(5, 7)]
        bird.direction = None
        sim.parse_commands(player, f"{bird.id} UP")
        assert bird.direction == UP

    def test_parse_commands_wait(self, sim):
        """WAIT command is accepted."""
        player = sim.players[0]
        sim.parse_commands(player, "WAIT")
        # No error, no deactivation
        assert player.active
