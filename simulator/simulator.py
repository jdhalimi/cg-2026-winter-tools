"""Exact-ish local simulator for the Snakebird game engine.

This file ports the runtime rules from
[`src/main/java/com/codingame/game/Game.java`](src/main/java/com/codingame/game/Game.java),
[`src/main/java/com/codingame/game/Serializer.java`](src/main/java/com/codingame/game/Serializer.java),
[`src/main/java/com/codingame/game/CommandManager.java`](src/main/java/com/codingame/game/CommandManager.java),
and [`src/main/java/com/codingame/game/grid/GridMaker.java`](src/main/java/com/codingame/game/grid/GridMaker.java)
to Python so two Python solutions can be imported and played against each other locally.

Expected bot interface:

1. the module exports a [`Game`](bots/main.py:125)-like class;
2. the instance implements [`load_initial_state()`](bots/main.py:217),
   [`update()`](bots/main.py:233), and [`play()`](bots/main.py:736).

Usage:

```bash
python simulator.py bots/main.py bots/best.py --seed 0 --league-level 4
python simulator.py bots/main.py bots/best.py \
  --seed 0 --league-level 4 \
  --map-output simulator/generated_maps.txt \
  --nb-maps 10
python simulator.py bots/main.py bots/best.py \
  --global-lines '["0","26","14",...]' \
  --frame-lines '["30","7 2",...]'
```
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

EMPTY = "."
WALL = "#"

UP = "UP"
DOWN = "DOWN"
LEFT = "LEFT"
RIGHT = "RIGHT"
UNSET = "UNSET"

DIRECTION_DELTAS: Dict[str, Tuple[int, int]] = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
    UNSET: (0, 0),
}

OPPOSITE: Dict[str, str] = {
    UP: DOWN,
    DOWN: UP,
    LEFT: RIGHT,
    RIGHT: LEFT,
    UNSET: UNSET,
}

TURN_ORDER = [UP, RIGHT, DOWN, LEFT]
ADJ4 = [(0, -1), (1, 0), (0, 1), (-1, 0)]
ADJ8 = ADJ4 + [(-1, -1), (1, 1), (1, -1), (-1, 1)]

TYPE_EMPTY = 0
TYPE_WALL = 1
TYPE_INVALID = -1

MOVE_PATTERNS = {
    UP: re.compile(r"^(?P<bird_id>\d+) UP( (?P<message>[^;]*))?$", re.IGNORECASE),
    DOWN: re.compile(r"^(?P<bird_id>\d+) DOWN( (?P<message>[^;]*))?$", re.IGNORECASE),
    LEFT: re.compile(r"^(?P<bird_id>\d+) LEFT( (?P<message>[^;]*))?$", re.IGNORECASE),
    RIGHT: re.compile(r"^(?P<bird_id>\d+) RIGHT( (?P<message>[^;]*))?$", re.IGNORECASE),
}
MARK_PATTERN = re.compile(r"^MARK (?P<x>\d+) (?P<y>\d+)$", re.IGNORECASE)
WAIT_PATTERN = re.compile(r"^WAIT$", re.IGNORECASE)


def java_round(value: float) -> int:
    return math.floor(value + 0.5)


def parse_dump_lines(raw: str) -> List[str]:
    text = Path(raw[1:]).read_text(encoding="utf-8") if raw.startswith("@") else raw
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = ast.literal_eval(text)

    if not isinstance(data, list):
        raise ValueError("dump lines must be provided as a JSON/Python list")
    return [str(item) for item in data]


def parse_losses(raw: str) -> Tuple[int, int]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("losses must be in the form 'a,b'")
    return int(parts[0]), int(parts[1])


def load_params(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = ast.literal_eval(text)

    if not isinstance(data, dict):
        raise ValueError(f"Params file must contain an object/dict: {path}")
    return dict(data)


def parse_body_string(raw_body: str) -> List["Coord"]:
    body: List[Coord] = []
    for raw_part in raw_body.split(":"):
        x_str, y_str = raw_part.split(",")
        body.append(Coord(int(x_str), int(y_str)))
    return body


@dataclass(frozen=True, order=True)
class Coord:
    x: int
    y: int

    def add(self, dx: int, dy: int) -> "Coord":
        return Coord(self.x + dx, self.y + dy)

    def to_int_string(self) -> str:
        return f"{self.x} {self.y}"


@dataclass(frozen=True)
class MapScenario:
    index: int
    name: str
    global_lines: List[str]
    frame_lines: List[str]


def load_map_scenarios(path: Path) -> List[MapScenario]:
    scenarios: List[MapScenario] = []
    current_name: Optional[str] = None
    pending_lists: List[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current_name = line[1:].strip() or f"map {len(scenarios) + 1}"
            continue

        pending_lists.append(line)
        if len(pending_lists) == 2:
            scenarios.append(
                MapScenario(
                    index=len(scenarios) + 1,
                    name=current_name or f"map {len(scenarios) + 1}",
                    global_lines=parse_dump_lines(pending_lists[0]),
                    frame_lines=parse_dump_lines(pending_lists[1]),
                )
            )
            current_name = None
            pending_lists = []
        elif len(pending_lists) > 2:
            raise ValueError(f"Invalid map block in {path}: expected 2 serialized lists per map")

    if pending_lists:
        raise ValueError(f"Incomplete map block in {path}: missing global or frame lines")
    if not scenarios:
        raise ValueError(f"No maps found in {path}")
    return scenarios


def dump_lines_literal(lines: Sequence[str]) -> str:
    return repr([str(line) for line in lines])


def write_map_scenarios(
    path: Path,
    scenarios: Sequence[Tuple[Optional[str], Sequence[str], Sequence[str]]],
) -> None:
    output_lines: List[str] = []
    for index, (name, global_lines, frame_lines) in enumerate(scenarios, start=1):
        if index > 1:
            output_lines.append("")
        output_lines.append(f"# {name or f'map {index}'}")
        output_lines.append(dump_lines_literal(global_lines))
        output_lines.append(dump_lines_literal(frame_lines))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def select_map_scenarios(
    scenarios: Sequence[MapScenario],
    map_index: Optional[int],
    map_name: Optional[str],
    all_maps: bool,
) -> List[MapScenario]:
    selector_count = int(map_index is not None) + int(map_name is not None) + int(all_maps)
    if selector_count > 1:
        raise ValueError("Use only one of --map, --map-name, or --all-maps")

    if all_maps:
        return list(scenarios)

    if map_name is not None:
        normalized = map_name.casefold()
        selected = [scenario for scenario in scenarios if scenario.name.casefold() == normalized]
        if not selected:
            raise ValueError(f"Map name not found: {map_name}")
        return selected

    if map_index is not None:
        if map_index < 1 or map_index > len(scenarios):
            raise ValueError(f"Map index out of range: {map_index}")
        return [scenarios[map_index - 1]]

    return [scenarios[0]]


class JavaRandom:
    """Port of [`java.util.Random`](src/main/java/com/codingame/game/Game.java:47)."""

    _multiplier = 0x5DEECE66D
    _addend = 0xB
    _mask = (1 << 48) - 1

    def __init__(self, seed: int):
        self.seed = (seed ^ self._multiplier) & self._mask

    def next(self, bits: int) -> int:
        self.seed = (self.seed * self._multiplier + self._addend) & self._mask
        return self.seed >> (48 - bits)

    def next_double(self) -> float:
        return ((self.next(26) << 27) + self.next(27)) / float(1 << 53)

    def next_int(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        if (bound & -bound) == bound:
            return (bound * self.next(31)) >> 31
        while True:
            bits = self.next(31)
            value = bits % bound
            if bits - value + (bound - 1) >= 0:
                return value

    def next_int_range(self, origin: int, bound: int) -> int:
        if origin >= bound:
            raise ValueError("origin must be < bound")
        return origin + self.next_int(bound - origin)


def shuffle_in_place(values: List[Coord], rnd: JavaRandom) -> None:
    for i in range(len(values), 1, -1):
        j = rnd.next_int(i)
        values[i - 1], values[j] = values[j], values[i - 1]


class Grid:
    def __init__(self, width: int, height: int, y_symmetry: bool = False):
        self.width = width
        self.height = height
        self.y_symmetry = y_symmetry
        self.cells: Dict[Coord, int] = {}
        self.coords: List[Coord] = []
        for y in range(height):
            for x in range(width):
                coord = Coord(x, y)
                self.coords.append(coord)
                self.cells[coord] = TYPE_EMPTY
        self.spawns: List[Coord] = []
        self.apples: List[Coord] = []

    def in_bounds(self, coord: Coord) -> bool:
        return 0 <= coord.x < self.width and 0 <= coord.y < self.height

    def get_type(self, coord: Coord) -> int:
        return self.cells.get(coord, TYPE_INVALID)

    def set_type(self, coord: Coord, tile_type: int) -> None:
        if not self.in_bounds(coord):
            return
        self.cells[coord] = tile_type

    def clear(self, coord: Coord) -> None:
        self.set_type(coord, TYPE_EMPTY)

    def opposite(self, coord: Coord) -> Coord:
        y = self.height - coord.y - 1 if self.y_symmetry else coord.y
        return Coord(self.width - coord.x - 1, y)

    def neighbours(self, coord: Coord, adjacency: Sequence[Tuple[int, int]] = ADJ4) -> List[Coord]:
        result: List[Coord] = []
        for dx, dy in adjacency:
            nxt = coord.add(dx, dy)
            if self.in_bounds(nxt):
                result.append(nxt)
        return result

    def add_apple(self, coord: Coord) -> None:
        if coord not in self.apples:
            self.apples.append(coord)

    def remove_apple(self, coord: Coord) -> None:
        try:
            self.apples.remove(coord)
        except ValueError:
            pass

    def detect_air_pockets(self) -> List[List[Coord]]:
        islands: List[List[Coord]] = []
        computed: set[Coord] = set()
        for start in self.coords:
            if start in computed or self.get_type(start) == TYPE_WALL:
                computed.add(start)
                continue
            queue = deque([start])
            computed.add(start)
            island: List[Coord] = []
            while queue:
                cur = queue.popleft()
                island.append(cur)
                for nxt in self.neighbours(cur):
                    if nxt not in computed and self.get_type(nxt) != TYPE_WALL:
                        computed.add(nxt)
                        queue.append(nxt)
            islands.append(island)
        return islands

    def detect_spawn_islands(self) -> List[List[Coord]]:
        islands: List[List[Coord]] = []
        spawn_set = set(self.spawns)
        computed: set[Coord] = set()
        for start in self.spawns:
            if start in computed:
                continue
            queue = deque([start])
            computed.add(start)
            island: List[Coord] = []
            while queue:
                cur = queue.popleft()
                island.append(cur)
                for nxt in self.neighbours(cur):
                    if nxt in spawn_set and nxt not in computed:
                        computed.add(nxt)
                        queue.append(nxt)
            islands.append(island)
        return islands

    def detect_lowest_island(self) -> List[Coord]:
        start = Coord(0, self.height - 1)
        if self.get_type(start) != TYPE_WALL:
            return []
        computed = {start}
        queue = deque([start])
        lowest = [start]
        while queue:
            cur = queue.popleft()
            for nxt in self.neighbours(cur):
                if nxt not in computed and self.get_type(nxt) == TYPE_WALL:
                    computed.add(nxt)
                    queue.append(nxt)
                    lowest.append(nxt)
        return lowest


class GridMaker:
    MIN_GRID_HEIGHT = 10
    MAX_GRID_HEIGHT = 24
    ASPECT_RATIO = 1.8
    SPAWN_HEIGHT = 3
    DESIRED_SPAWNS = 4

    def __init__(self, rnd: JavaRandom, league_level: int):
        self.random = rnd
        self.league_level = league_level

    def get_free_above(self, grid: Grid, coord: Coord, by: int) -> List[Coord]:
        result: List[Coord] = []
        for i in range(1, by + 1):
            above = Coord(coord.x, coord.y - i)
            if grid.in_bounds(above) and grid.get_type(above) == TYPE_EMPTY:
                result.append(above)
            else:
                break
        return result

    def check_grid(self, grid: Grid) -> None:
        if len(grid.apples) != len(set(grid.apples)):
            raise RuntimeError("Duplicate apples")
        for apple in grid.apples:
            if grid.get_type(apple) != TYPE_EMPTY:
                raise RuntimeError(f"Apple on wall at {apple}")

    def make(self) -> Grid:
        if self.league_level == 1:
            skew = 2.0
        elif self.league_level == 2:
            skew = 1.0
        elif self.league_level == 3:
            skew = 0.8
        else:
            skew = 0.3

        rand = self.random.next_double()
        height = self.MIN_GRID_HEIGHT + java_round((rand**skew) * (self.MAX_GRID_HEIGHT - self.MIN_GRID_HEIGHT))
        width = java_round(height * self.ASPECT_RATIO)
        if width % 2 != 0:
            width += 1

        grid = Grid(width, height)
        b = 5.0 + self.random.next_double() * 10.0

        for x in range(width):
            grid.set_type(Coord(x, height - 1), TYPE_WALL)

        for y in range(height - 2, -1, -1):
            y_norm = (height - 1 - y) / (height - 1)
            block_chance = 1.0 / (y_norm + 0.1) / b
            for x in range(width):
                if self.random.next_double() < block_chance:
                    grid.set_type(Coord(x, y), TYPE_WALL)

        for coord in list(grid.coords):
            grid.set_type(grid.opposite(coord), grid.get_type(coord))

        for island in grid.detect_air_pockets():
            if len(island) < 10:
                for coord in island:
                    grid.set_type(coord, TYPE_WALL)

        something_destroyed = True
        while something_destroyed:
            something_destroyed = False
            for coord in list(grid.coords):
                if grid.get_type(coord) == TYPE_WALL:
                    continue
                neighbour_walls = [n for n in grid.neighbours(coord) if grid.get_type(n) == TYPE_WALL]
                if len(neighbour_walls) >= 3:
                    destroyable = [n for n in neighbour_walls if n.y <= coord.y]
                    shuffle_in_place(destroyable, self.random)
                    doomed = destroyable[0]
                    grid.clear(doomed)
                    grid.clear(grid.opposite(doomed))
                    something_destroyed = True

        island = grid.detect_lowest_island()
        lower_by = 0
        can_lower = True
        while can_lower:
            for x in range(width):
                coord = Coord(x, height - 1 - (lower_by + 1))
                if coord not in island:
                    can_lower = False
                    break
            if can_lower:
                lower_by += 1

        if lower_by >= 2:
            lower_by = self.random.next_int_range(2, lower_by + 1)

        for coord in island:
            grid.clear(coord)
            grid.clear(grid.opposite(coord))
        for coord in island:
            lowered = Coord(coord.x, coord.y + lower_by)
            if grid.in_bounds(lowered):
                grid.set_type(lowered, TYPE_WALL)
                grid.set_type(grid.opposite(lowered), TYPE_WALL)

        for y in range(height):
            for x in range(width // 2):
                coord = Coord(x, y)
                if grid.get_type(coord) == TYPE_EMPTY and self.random.next_double() < 0.025:
                    grid.add_apple(coord)
                    grid.add_apple(grid.opposite(coord))

        if len(grid.apples) < 8:
            grid.apples.clear()
            free_tiles = [coord for coord in grid.coords if grid.get_type(coord) == TYPE_EMPTY]
            shuffle_in_place(free_tiles, self.random)
            min_apple_coords = max(4, int(0.025 * len(free_tiles)))
            while len(grid.apples) < min_apple_coords * 2 and free_tiles:
                coord = free_tiles.pop(0)
                opp = grid.opposite(coord)
                grid.add_apple(coord)
                grid.add_apple(opp)
                try:
                    free_tiles.remove(opp)
                except ValueError:
                    pass

        for coord in list(grid.coords):
            if grid.get_type(coord) == TYPE_EMPTY:
                continue
            neighbour_wall_count = sum(1 for n in grid.neighbours(coord, ADJ8) if grid.get_type(n) == TYPE_WALL)
            if neighbour_wall_count == 0:
                grid.clear(coord)
                grid.clear(grid.opposite(coord))
                grid.add_apple(coord)
                grid.add_apple(grid.opposite(coord))

        potential_spawns = [
            coord
            for coord in grid.coords
            if grid.get_type(coord) == TYPE_WALL
            and len(self.get_free_above(grid, coord, self.SPAWN_HEIGHT)) >= self.SPAWN_HEIGHT
        ]
        shuffle_in_place(potential_spawns, self.random)

        desired_spawns = self.DESIRED_SPAWNS
        if height <= 15:
            desired_spawns -= 1
        if height <= 10:
            desired_spawns -= 1

        spawn_set: set[Coord] = set()
        while desired_spawns > 0 and potential_spawns:
            spawn = potential_spawns.pop(0)
            spawn_loc = self.get_free_above(grid, spawn, self.SPAWN_HEIGHT)
            too_close = False
            for coord in spawn_loc:
                if coord.x == width // 2 - 1 or coord.x == width // 2:
                    too_close = True
                    break
                for neighbour in grid.neighbours(coord, ADJ8):
                    if neighbour in spawn_set or grid.opposite(neighbour) in spawn_set:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                continue

            for coord in spawn_loc:
                if coord not in spawn_set:
                    spawn_set.add(coord)
                    grid.spawns.append(coord)
                grid.remove_apple(coord)
                grid.remove_apple(grid.opposite(coord))
            desired_spawns -= 1

        self.check_grid(grid)
        return grid


@dataclass
class Bird:
    id: int
    owner_index: int
    body: List[Coord] = field(default_factory=list)
    alive: bool = True
    direction: Optional[str] = None
    message: Optional[str] = None

    def head(self) -> Coord:
        return self.body[0]

    def facing(self) -> str:
        if len(self.body) < 2:
            return UNSET
        dx = self.body[0].x - self.body[1].x
        dy = self.body[0].y - self.body[1].y
        for direction, delta in DIRECTION_DELTAS.items():
            if delta == (dx, dy):
                return direction
        return UNSET

    def set_message(self, message: Optional[str]) -> None:
        self.message = message
        if self.message is not None and len(self.message) > 48:
            self.message = self.message[:46] + "..."


class BotAdapter:
    def __init__(self, path: Path, slot: int, params: Optional[Dict[str, Any]] = None):
        self.path = path
        digest = hashlib.md5(f"{path}:{slot}".encode(), usedforsecurity=False).hexdigest()
        module_name = f"simulator_bot_{slot}_{digest}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to import {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module
        if not hasattr(module, "Game"):
            raise RuntimeError(f"{path} does not export a Game class")
        resolved_params = dict(getattr(module, "META_PARAMS", {}))
        if params:
            resolved_params.update(params)
        self.game = module.Game(resolved_params)

    @staticmethod
    def _reader(lines: Iterable[str]) -> Callable[[], str]:
        iterator = iter(lines)
        return lambda: next(iterator)

    def initialize(self, global_lines: Sequence[str]) -> None:
        self.game.load_initial_state(self._reader(global_lines))

    def play_turn(self, frame_lines: Sequence[str]) -> str:
        self.game.update(self._reader(frame_lines))
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            self.game.play()
        output_lines = stdout_buffer.getvalue().splitlines()
        return output_lines[0] if output_lines else ""


@dataclass
class PlayerState:
    index: int
    adapter: BotAdapter
    birds: List[Bird] = field(default_factory=list)
    marks: List[Coord] = field(default_factory=list)
    active: bool = True
    score: int = 0
    last_execution_time_ms: int = 0
    pending_output: str = "WAIT"
    deactivate_reason: Optional[str] = None

    def reset(self) -> None:
        for bird in self.birds:
            bird.direction = None
            bird.message = None
        self.marks.clear()

    def get_bird_by_id(self, bird_id: int) -> Optional[Bird]:
        for bird in self.birds:
            if bird.id == bird_id:
                return bird
        return None

    def add_mark(self, coord: Coord) -> bool:
        if len(self.marks) < 4:
            self.marks.append(coord)
            return True
        return False


class SnakebirdSimulator:
    def __init__(
        self,
        bot_paths: Sequence[Path],
        seed: int,
        league_level: int,
        max_turns: int = 200,
        initial_global_lines: Optional[Sequence[str]] = None,
        initial_frame_lines: Optional[Sequence[str]] = None,
        initial_losses: Tuple[int, int] = (0, 0),
        bot_params: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        adapter_factory: Optional[Callable[[Path, int, Optional[Dict[str, Any]]], Any]] = None,
    ):
        if len(bot_paths) != 2:
            raise ValueError("Exactly two bot paths are required")
        self.seed = seed
        self.league_level = league_level
        self.max_turns = max_turns
        self.turn = 0
        self.random = JavaRandom(seed)
        self.grid = Grid(0, 0)
        if bot_params is None:
            bot_params = [None, None]
        if adapter_factory is None:
            adapter_factory = BotAdapter
        self.players = [PlayerState(i, adapter_factory(bot_paths[i], i, bot_params[i])) for i in range(2)]
        self.losses = [initial_losses[0], initial_losses[1]]
        self.summary: List[str] = []

        if initial_global_lines is not None and initial_frame_lines is not None:
            self.load_state_from_serialized_lines(initial_global_lines, initial_frame_lines)
        else:
            self.grid = GridMaker(self.random, league_level).make()
            self.init_players()

        self.send_global_info()

    def build_global_lines_for_player(self, base_global_lines: Sequence[str], player_index: int) -> List[str]:
        cursor = 0
        base_player_index = int(base_global_lines[cursor])
        cursor += 1
        width = int(base_global_lines[cursor])
        cursor += 1
        height = int(base_global_lines[cursor])
        cursor += 1
        rows = list(base_global_lines[cursor : cursor + height])
        cursor += height
        birds_per_player = int(base_global_lines[cursor])
        cursor += 1
        first_ids = [int(value) for value in base_global_lines[cursor : cursor + birds_per_player]]
        cursor += birds_per_player
        second_ids = [int(value) for value in base_global_lines[cursor : cursor + birds_per_player]]

        if base_player_index == 0:
            player0_ids, player1_ids = first_ids, second_ids
        else:
            player0_ids, player1_ids = second_ids, first_ids

        my_ids = player0_ids if player_index == 0 else player1_ids
        opp_ids = player1_ids if player_index == 0 else player0_ids
        return [
            str(player_index),
            str(width),
            str(height),
            *rows,
            str(birds_per_player),
            *[str(bird_id) for bird_id in my_ids],
            *[str(bird_id) for bird_id in opp_ids],
        ]

    def load_state_from_serialized_lines(
        self,
        global_lines: Sequence[str],
        frame_lines: Sequence[str],
    ) -> None:
        p0_global_lines = self.build_global_lines_for_player(global_lines, 0)
        cursor = 0
        cursor += 1  # my_id
        width = int(p0_global_lines[cursor])
        cursor += 1
        height = int(p0_global_lines[cursor])
        cursor += 1

        self.grid = Grid(width, height)
        rows = p0_global_lines[cursor : cursor + height]
        cursor += height
        for y, row in enumerate(rows):
            for x, cell in enumerate(row):
                if cell == WALL:
                    self.grid.set_type(Coord(x, y), TYPE_WALL)

        birds_per_player = int(p0_global_lines[cursor])
        cursor += 1
        player0_ids = [int(value) for value in p0_global_lines[cursor : cursor + birds_per_player]]
        cursor += birds_per_player
        player1_ids = [int(value) for value in p0_global_lines[cursor : cursor + birds_per_player]]

        self.players[0].birds = [Bird(id=bird_id, owner_index=0) for bird_id in player0_ids]
        self.players[1].birds = [Bird(id=bird_id, owner_index=1) for bird_id in player1_ids]
        for player in self.players:
            player.marks = []
            player.active = True
            player.score = 0
            player.pending_output = "WAIT"
            player.deactivate_reason = None

        birds_by_id = {bird.id: bird for bird in self.get_all_birds()}
        for bird in birds_by_id.values():
            bird.body = []
            bird.alive = False
            bird.direction = None
            bird.message = None

        frame_cursor = 0
        apple_count = int(frame_lines[frame_cursor])
        frame_cursor += 1
        self.grid.apples = [
            Coord(*map(int, frame_lines[index].split()))
            for index in range(frame_cursor, frame_cursor + apple_count)
        ]
        frame_cursor += apple_count

        snake_count = int(frame_lines[frame_cursor])
        frame_cursor += 1
        for raw_line in frame_lines[frame_cursor : frame_cursor + snake_count]:
            bird_id_str, raw_body = raw_line.split()
            bird = birds_by_id[int(bird_id_str)]
            bird.body = parse_body_string(raw_body)
            bird.alive = True

        for player in self.players:
            for bird in player.birds:
                if bird.alive and len(bird.body) >= 2:
                    bird.direction = bird.facing()

    def init_players(self) -> None:
        bird_id = 0
        spawn_locations = [sorted(island) for island in self.grid.detect_spawn_islands()]
        for player in self.players:
            player.birds = []
            player.marks = []
            for spawn in spawn_locations:
                bird = Bird(id=bird_id, owner_index=player.index)
                bird_id += 1
                player.birds.append(bird)
                for coord in spawn:
                    actual = self.grid.opposite(coord) if player.index == 1 else coord
                    bird.body.append(actual)
                    if len(bird.body) == 1:
                        left = actual.add(-1, 0)
                        right = actual.add(1, 0)
                        if self.grid.get_type(left) == TYPE_WALL and self.grid.get_type(right) == TYPE_WALL:
                            self.grid.clear(left)
                            self.grid.clear(self.grid.opposite(left))

    def deactivate_player(self, player: PlayerState, message: str) -> None:
        player.active = False
        player.score = -1
        player.deactivate_reason = message
        self.summary.append(message)

    def get_all_birds(self) -> List[Bird]:
        return [bird for player in self.players for bird in player.birds]

    def get_live_birds(self) -> List[Bird]:
        return [bird for bird in self.get_all_birds() if bird.alive]

    def serialize_global_info_for(self, player: PlayerState) -> List[str]:
        lines: List[str] = [str(player.index), str(self.grid.width), str(self.grid.height)]
        for y in range(self.grid.height):
            row = []
            for x in range(self.grid.width):
                row.append(WALL if self.grid.get_type(Coord(x, y)) == TYPE_WALL else EMPTY)
            lines.append("".join(row))
        lines.append(str(len(self.players[0].birds)))
        lines.extend(str(bird.id) for bird in player.birds)
        lines.extend(str(bird.id) for bird in self.players[1 - player.index].birds)
        return lines

    def serialize_frame_info_for(self, _player: PlayerState) -> List[str]:
        lines: List[str] = [str(len(self.grid.apples))]
        lines.extend(coord.to_int_string() for coord in self.grid.apples)
        live_birds = self.get_live_birds()
        lines.append(str(len(live_birds)))
        for bird in live_birds:
            body = ":".join(f"{coord.x},{coord.y}" for coord in bird.body)
            lines.append(f"{bird.id} {body}")
        return lines

    def send_global_info(self) -> None:
        for player in self.players:
            if player.active:
                player.adapter.initialize(self.serialize_global_info_for(player))

    def reset_game_turn_data(self) -> None:
        for player in self.players:
            player.reset()

    def parse_commands(self, player: PlayerState, line: str) -> None:
        errors: List[str] = []
        commands = line.split(";")
        reasonable_limit = 30

        for command in commands:
            if reasonable_limit <= 0:
                return
            reasonable_limit -= 1

            found = False
            for direction, pattern in MOVE_PATTERNS.items():
                match = pattern.match(command)
                if not match:
                    continue
                found = True
                bird_id = int(match.group("bird_id"))
                bird = player.get_bird_by_id(bird_id)
                if bird is None:
                    errors.append(f"Player {player.index}: Bird not found for id {bird_id}")
                    break
                if not bird.alive:
                    errors.append(f"Player {player.index}: Bird with id {bird_id} is dead")
                    break
                if bird.direction is not None:
                    errors.append(f"Player {player.index}: Bird id {bird_id} has already been given a move")
                    break
                if OPPOSITE[bird.facing()] == direction:
                    errors.append(f"Player {player.index}: Bird id {bird_id} cannot move backwards")
                    break
                bird.direction = direction
                bird.set_message(match.group("message"))
                break

            if found:
                continue

            mark_match = MARK_PATTERN.match(command)
            if mark_match:
                found = True
                coord = Coord(int(mark_match.group("x")), int(mark_match.group("y")))
                if not player.add_mark(coord):
                    errors.append(f"Player {player.index}: Too many MARK actions this turn")
                continue

            if WAIT_PATTERN.match(command):
                continue

            self.deactivate_player(
                player,
                f"Invalid input from player {player.index}: expected MESSAGE text, got {command!r}",
            )
            return

        self.summary.extend(errors[:4])
        if len(errors) > 4:
            self.summary.append(f"...and {len(errors) - 4} more errors.")

    def handle_player_commands(self) -> None:
        for player in list(self.players):
            if not player.active:
                continue
            self.parse_commands(player, player.pending_output or "WAIT")

    def execute_players(self) -> None:
        for player in self.players:
            if not player.active:
                continue
            frame_lines = self.serialize_frame_info_for(player)
            start = time.perf_counter()
            try:
                player.pending_output = player.adapter.play_turn(frame_lines)
            except Exception as exc:  # pragma: no cover - defensive runtime behavior
                self.deactivate_player(player, f"Player {player.index} crashed: {exc}")
                player.pending_output = "WAIT"
            finally:
                player.last_execution_time_ms = int((time.perf_counter() - start) * 1000)

    def do_moves(self) -> None:
        for player in self.players:
            for bird in player.birds:
                if not bird.alive:
                    continue
                if bird.direction in (None, UNSET):
                    bird.direction = bird.facing()
                dx, dy = DIRECTION_DELTAS[bird.direction or UNSET]
                new_head = bird.head().add(dx, dy)
                will_eat_apple = new_head in self.grid.apples
                if not will_eat_apple:
                    bird.body.pop()
                bird.body.insert(0, new_head)

    def do_eats(self) -> None:
        apples_eaten = {
            bird.head()
            for player in self.players
            for bird in player.birds
            if bird.alive and bird.head() in self.grid.apples
        }
        if apples_eaten:
            self.grid.apples = [apple for apple in self.grid.apples if apple not in apples_eaten]

    def do_beheadings(self) -> None:
        birds_to_behead: List[Bird] = []
        live_birds = self.get_live_birds()
        for bird in live_birds:
            head = bird.head()
            is_in_wall = self.grid.get_type(head) == TYPE_WALL
            intersecting = [other for other in live_birds if head in other.body]
            is_in_bird = any(other.id != bird.id or head in other.body[1:] for other in intersecting)
            if is_in_wall or is_in_bird:
                birds_to_behead.append(bird)

        for bird in birds_to_behead:
            if len(bird.body) <= 3:
                bird.alive = False
                self.losses[bird.owner_index] += len(bird.body)
            else:
                bird.body.pop(0)
                self.losses[bird.owner_index] += 1

    def has_tile_or_apple_under(self, coord: Coord) -> bool:
        below = coord.add(0, 1)
        return self.grid.get_type(below) == TYPE_WALL or below in self.grid.apples

    def is_grounded_coord(self, coord: Coord, grounded_birds: Sequence[Bird]) -> bool:
        below = coord.add(0, 1)
        if self.has_tile_or_apple_under(coord):
            return True
        return any(below in bird.body for bird in grounded_birds)

    def do_falls(self) -> None:
        something_fell = True
        airborne_birds: List[Bird] = self.get_live_birds()
        grounded_birds: List[Bird] = []
        grounded_ids: set[int] = set()
        out_of_bounds_ids: set[int] = set()

        while something_fell:
            something_fell = False
            something_got_grounded = True

            while something_got_grounded:
                something_got_grounded = False
                newly_grounded: List[Bird] = []
                for bird in airborne_birds:
                    if any(self.is_grounded_coord(coord, grounded_birds) for coord in bird.body):
                        newly_grounded.append(bird)
                        something_got_grounded = True
                if newly_grounded:
                    for bird in newly_grounded:
                        if bird.id not in grounded_ids:
                            grounded_ids.add(bird.id)
                            grounded_birds.append(bird)
                    airborne_birds = [bird for bird in airborne_birds if bird.id not in grounded_ids]

            for bird in airborne_birds:
                something_fell = True
                bird.body = [coord.add(0, 1) for coord in bird.body]
                if all(part.y >= self.grid.height + 1 for part in bird.body):
                    bird.alive = False
                    out_of_bounds_ids.add(bird.id)

            if out_of_bounds_ids:
                airborne_birds = [bird for bird in airborne_birds if bird.id not in out_of_bounds_ids]

    def perform_game_update(self, turn: int) -> None:
        self.turn = turn
        self.do_moves()
        self.do_eats()
        self.do_beheadings()
        self.do_falls()

    def is_game_over(self) -> bool:
        no_apples = not self.grid.apples
        player_dead = any(not any(bird.alive for bird in player.birds) for player in self.players)
        return no_apples or player_dead

    def finalize_scores(self) -> None:
        for player in self.players:
            if not player.active:
                player.score = -1
            else:
                player.score = sum(len(bird.body) for bird in player.birds if bird.alive)

        if self.players[0].score == self.players[1].score and self.players[0].score != -1:
            for player in self.players:
                player.score -= self.losses[player.index]

    def run(self, record: bool = False) -> Dict[str, Any]:
        turns_played = 0
        recorded_frames: List[List[str]] = []
        for turn in range(1, self.max_turns + 1):
            self.reset_game_turn_data()
            if record:
                recorded_frames.append(self.serialize_frame_info_for(self.players[0]))
            self.execute_players()
            self.handle_player_commands()
            self.perform_game_update(turn)
            turns_played = turn
            if self.is_game_over() or sum(1 for player in self.players if player.active) < 2:
                break

        self.finalize_scores()
        scores = [player.score for player in self.players]
        best_score = max(scores)
        winners = [player.index for player in self.players if player.score == best_score]
        return {
            "seed": self.seed,
            "league_level": self.league_level,
            "turns": turns_played,
            "scores": scores,
            "losses": list(self.losses),
            "winner": winners[0] if len(winners) == 1 else None,
            "tie": len(winners) != 1,
            "players": [
                {
                    "index": player.index,
                    "path": str(player.adapter.path),
                    "active": player.active,
                    "score": player.score,
                    "reason": player.deactivate_reason,
                }
                for player in self.players
            ],
            "remaining_apples": [apple.to_int_string() for apple in self.grid.apples],
            "summary": list(self.summary),
            "recorded_frames": recorded_frames,
        }


@dataclass
class ScenarioTask:
    name: Optional[str]
    seed: int
    global_lines: Optional[List[str]]
    frame_lines: Optional[List[str]]
    bot_a: Path
    bot_b: Path
    league_level: int
    max_turns: int
    initial_losses: Tuple[int, int]
    bot_params: List[Optional[Dict[str, Any]]]
    record: bool
    need_map_output: bool


def _run_scenario_worker(
    task: ScenarioTask,
) -> Tuple[Optional[str], Optional[str], List[str], List[str], Dict[str, Any]]:
    sim = SnakebirdSimulator(
        [task.bot_a, task.bot_b],
        seed=task.seed,
        league_level=task.league_level,
        max_turns=task.max_turns,
        initial_global_lines=task.global_lines,
        initial_frame_lines=task.frame_lines,
        initial_losses=task.initial_losses,
        bot_params=task.bot_params,
    )
    effective_global = (
        list(task.global_lines)
        if task.global_lines is not None
        else sim.serialize_global_info_for(sim.players[0])
    )
    effective_frame = (
        list(task.frame_lines)
        if task.frame_lines is not None
        else sim.serialize_frame_info_for(sim.players[0])
    )
    effective_name = task.name
    if effective_name is None and task.need_map_output:
        effective_name = f"generated map seed={task.seed} league={task.league_level}"
    result = sim.run(record=task.record)
    return task.name, effective_name, effective_global, effective_frame, result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import two Python Snakebird bots and simulate a full match.")
    parser.add_argument("bot_a", type=Path, help="Path to player 0 Python solution")
    parser.add_argument("bot_b", type=Path, help="Path to player 1 Python solution")
    parser.add_argument("--seed", type=int, default=0, help="Java Random seed used by the grid generator")
    parser.add_argument("--league-level", type=int, default=4, help="League level used by the Java grid generator")
    parser.add_argument("--max-turns", type=int, default=200, help="Maximum number of turns")
    parser.add_argument("--maps", type=Path, help="Path to a map file containing dumped initial states")
    parser.add_argument("--map-output", type=Path, help="Write the selected/generated initial map states to this file")
    parser.add_argument(
        "--nb-maps",
        type=int,
        default=1,
        help="Generate this many maps when no explicit map input is provided",
    )
    parser.add_argument("--map", dest="map_index", type=int, help="1-based map index to run from --maps")
    parser.add_argument("--map-name", type=str, help="Map name to run from --maps")
    parser.add_argument("--all-maps", action="store_true", help="Run all maps from --maps")
    parser.add_argument("--weights", type=Path, help="JSON file applied to both bots as META_PARAMS overrides")
    parser.add_argument("--weights-a", type=Path, help="JSON file applied only to player 0 bot")
    parser.add_argument("--weights-b", type=Path, help="JSON file applied only to player 1 bot")
    parser.add_argument(
        "--global-lines",
        type=str,
        help="Serialized global input lines as a JSON/Python list, or @path to a file containing that list",
    )
    parser.add_argument(
        "--frame-lines",
        type=str,
        help="Serialized frame input lines as a JSON/Python list, or @path to a file containing that list",
    )
    parser.add_argument(
        "--losses",
        type=str,
        default="0,0",
        help="Initial losses as 'a,b' when resuming from dumped lines",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write game results as flat text (initial state + all turns); resumable via --global-lines/--frame-lines",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers when running multiple maps (0=auto, 1=sequential, default 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.global_lines is None) != (args.frame_lines is None):
        raise SystemExit("Both --global-lines and --frame-lines must be provided together")

    if args.nb_maps <= 0:
        raise SystemExit("--nb-maps must be a positive integer")

    if args.maps is not None and (args.global_lines is not None or args.frame_lines is not None):
        raise SystemExit("Use either --maps or the pair --global-lines/--frame-lines, not both")

    if args.maps is None and (args.map_index is not None or args.map_name is not None or args.all_maps):
        raise SystemExit("--map, --map-name, and --all-maps require --maps")

    if args.nb_maps != 1 and args.maps is not None:
        raise SystemExit("--nb-maps cannot be used with --maps")

    if args.nb_maps != 1 and (args.global_lines is not None or args.frame_lines is not None):
        raise SystemExit("--nb-maps cannot be used with --global-lines/--frame-lines")

    initial_losses = parse_losses(args.losses)
    shared_params = load_params(args.weights) if args.weights is not None else {}
    bot_a_params = {**shared_params, **(load_params(args.weights_a) if args.weights_a is not None else {})}
    bot_b_params = {**shared_params, **(load_params(args.weights_b) if args.weights_b is not None else {})}
    bot_params: List[Optional[Dict[str, Any]]] = [bot_a_params or None, bot_b_params or None]
    output_maps: List[Tuple[Optional[str], List[str], List[str]]] = []

    scenarios: List[Tuple[Optional[str], int, Optional[List[str]], Optional[List[str]]]]
    if args.maps is not None:
        loaded_maps = load_map_scenarios(args.maps)
        selected_maps = select_map_scenarios(loaded_maps, args.map_index, args.map_name, args.all_maps)
        scenarios = [
            (scenario.name, args.seed, scenario.global_lines, scenario.frame_lines)
            for scenario in selected_maps
        ]
    else:
        initial_global_lines = parse_dump_lines(args.global_lines) if args.global_lines is not None else None
        initial_frame_lines = parse_dump_lines(args.frame_lines) if args.frame_lines is not None else None
        if initial_global_lines is not None:
            scenarios = [(None, args.seed, initial_global_lines, initial_frame_lines)]
        else:
            scenarios = []
            for index in range(args.nb_maps):
                scenario_seed = args.seed + index
                scenario_name = None
                if args.nb_maps > 1 or args.map_output is not None:
                    scenario_name = f"generated map {index + 1} seed={scenario_seed} league={args.league_level}"
                scenarios.append((scenario_name, scenario_seed, None, None))

    do_record = args.output is not None
    need_map_output = args.map_output is not None

    tasks = [
        ScenarioTask(
            name=name,
            seed=seed,
            global_lines=gl,
            frame_lines=fl,
            bot_a=args.bot_a,
            bot_b=args.bot_b,
            league_level=args.league_level,
            max_turns=args.max_turns,
            initial_losses=initial_losses,
            bot_params=bot_params,
            record=do_record,
            need_map_output=need_map_output,
        )
        for name, seed, gl, fl in scenarios
    ]

    parallel = len(tasks) > 1 and args.workers != 1

    if parallel:
        import concurrent.futures
        workers = args.workers if args.workers > 0 else None
        print(f"Running {len(tasks)} maps in parallel (workers={workers or 'auto'})...")
        interrupted = False
        futures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_scenario_worker, t) for t in tasks]
            try:
                concurrent.futures.wait(futures)
            except KeyboardInterrupt:
                interrupted = True
                print("\nInterrupted — cancelling pending jobs...")
                for f in futures:
                    f.cancel()
        ordered = []
        for f in futures:
            if f.done() and not f.cancelled() and f.exception() is None:
                ordered.append(f.result())
        if interrupted:
            done = len(ordered)
            total = len(tasks)
            print(f"Showing {done}/{total} completed game(s).\n")
    else:
        ordered = []
        try:
            for t in tasks:
                ordered.append(_run_scenario_worker(t))
        except KeyboardInterrupt:
            print(f"\nInterrupted — showing {len(ordered)}/{len(tasks)} completed game(s).\n")

    game_results: List[Tuple[Optional[str], List[str], List[str], Dict[str, Any]]] = []

    for scenario_name, effective_name, effective_global_lines, effective_frame_lines, result in ordered:
        if need_map_output:
            output_maps.append((effective_name, effective_global_lines, effective_frame_lines))
        game_results.append((scenario_name, effective_global_lines, effective_frame_lines, result))

        if scenario_name is not None:
            print(f"=== {scenario_name} ===")
        winner = result["winner"]
        if winner is None:
            print(f"Tie after {result['turns']} turns | scores={result['scores']} | losses={result['losses']}")
        else:
            print(
                f"Winner: player {winner} | turns={result['turns']} | "
                f"scores={result['scores']} | losses={result['losses']}"
            )
        for player in result["players"]:
            status = "active" if player["active"] else f"inactive ({player['reason']})"
            print(f"P{player['index']}: {player['path']} -> {player['score']} [{status}]")
        if result["summary"]:
            print("Summary:")
            for line in result["summary"]:
                print(f"- {line}")
        if len(scenarios) > 1:
            print()

    if len(game_results) > 1:
        path_a = game_results[0][3]["players"][0]["path"]
        path_b = game_results[0][3]["players"][1]["path"]
        wins_a = wins_b = ties = 0
        col_w = max(len(s or f"game {i+1}") for i, (s, _, __, ___) in enumerate(game_results))
        col_w = max(col_w, 4)
        header = f"{'#':>4}  {'Map':<{col_w}}  {'P0 score':>8}  {'P1 score':>8}  Result"
        print("=" * len(header))
        print(f"RESULTS SUMMARY ({len(game_results)} games)")
        print(f"  P0: {path_a}")
        print(f"  P1: {path_b}")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for i, (name, _gl, _fl, res) in enumerate(game_results, start=1):
            label = name or f"game {i}"
            s0, s1 = res["scores"]
            w = res["winner"]
            if w is None:
                outcome = "Tie"
                ties += 1
            elif w == 0:
                outcome = "P0 wins"
                wins_a += 1
            else:
                outcome = "P1 wins"
                wins_b += 1
            print(f"{i:>4}  {label:<{col_w}}  {s0:>8}  {s1:>8}  {outcome}")
        print("-" * len(header))
        print(f"{'Totals':<{col_w+6}}  P0 wins: {wins_a}  |  P1 wins: {wins_b}  |  Ties: {ties}")
        print("=" * len(header))

    if args.map_output is not None:
        write_map_scenarios(args.map_output, output_maps)
        print(f"Saved {len(output_maps)} map(s) to {args.map_output}")

    if args.output is not None:
        lines: List[str] = []
        for i, (name, gl, _fl, res) in enumerate(game_results):
            if name:
                lines.append(f"# {name}")
            elif len(game_results) > 1:
                lines.append(f"# game {i + 1}")
            lines.append(repr(gl))
            for frame in res["recorded_frames"]:
                lines.append(repr(frame))
            lines.append("")
        args.output.write_text("\n".join(lines), encoding="utf-8")
        total_turns = sum(len(res["recorded_frames"]) for _, _, _, res in game_results)
        print(f"Saved {len(game_results)} game(s) / {total_turns} turns to {args.output}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
