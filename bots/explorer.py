"""
explorer.py — Territory-control bot using Voronoi flood fill.

Core idea: instead of purely chasing power sources (like best.py), this bot
evaluates moves primarily by how many grid cells it can reach before the enemy
(Voronoi partition). Dead-end avoidance and power-racing are secondary.
"""
import sys
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

EMPTY = "."
PLATFORM = "#"

UP = "UP"
DOWN = "DOWN"
LEFT = "LEFT"
RIGHT = "RIGHT"

DIRECTIONS: Dict[str, Tuple[int, int]] = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
}

OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
TURN_ORDER = [UP, RIGHT, DOWN, LEFT]
SCORE_NEG_INF = -(10**9)

META_PARAMS: Dict[str, int] = {
    # --- Territory (Voronoi) ---
    "territory_weight": 9,           # pts per cell I reach before enemy
    "enemy_territory_penalty": 4,    # pts penalty per cell enemy reaches first
    "territory_limit": 100,          # max BFS nodes per voronoi evaluation
    # --- Power sources ---
    "power_distance_penalty": 30,    # penalty per manhattan unit to nearest power
    "eat_bonus": 800,                # bonus for eating a power source
    "no_power_bonus": 160,           # bonus when no power left (just survive)
    # --- Contest power (manhattan comparison vs enemy) ---
    "contest_closer_bonus": 45,
    "contest_tie_bonus": 22,
    # --- Aggression ---
    "gravity_kill_bonus": 650,       # bonus for triggering enemy gravity-kill
    "head_collision_min_length": 6,  # min length to attempt head collision
    "head_collision_short_enemy_max_length": 2,
    "head_collision_short_enemy_bonus": 280,
    "head_collision_size_margin": 3,
    "head_collision_larger_bonus": 100,
    # --- Safety ---
    "danger_penalty": 190,           # penalty for moving into enemy danger zone
    "body_danger_penalty": 40,       # penalty per body cell in danger zone
    "no_exit_penalty": 520,          # penalty for having 0 forward exits
    "one_exit_penalty": 290,         # penalty for having only 1 forward exit
    "repeat_state_penalty": 180,     # penalty for revisiting a body state
    "momentum_bonus": 8,             # small bonus for continuing same direction
    # --- Loop escape ---
    "score_loop_escape_progress_bonus": 290,
    "score_loop_escape_directional_bonus": 95,
    # --- Lookahead ---
    "lookahead_future_score_divisor": 2,
}

Coord = Tuple[int, int]


def debug(*args):
    print(*args, file=sys.stderr, flush=True)


def add_coord(coord: Coord, delta: Tuple[int, int]) -> Coord:
    return coord[0] + delta[0], coord[1] + delta[1]


def manhattan_distance(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def parse_body(raw_body: str) -> List[Coord]:
    body = []
    for raw_part in raw_body.split(":"):
        x, y = raw_part.split(",")
        body.append((int(x), int(y)))
    return body


def infer_direction(body: List[Coord], fallback: str = UP) -> str:
    if len(body) >= 2:
        dx = body[0][0] - body[1][0]
        dy = body[0][1] - body[1][1]
        for direction, delta in DIRECTIONS.items():
            if delta == (dx, dy):
                return direction
    return fallback


@dataclass
class SnakeBot:
    id: int
    body: List[Coord]
    direction: str

    def head(self) -> Coord:
        return self.body[0]


@dataclass
class SimulationResult:
    direction: str
    body: List[Coord]
    alive: bool
    ate: bool
    collided: bool
    fall_distance: int
    remaining_power_sources: Set[Coord]
    score: int


class Game:
    def __init__(self, params: Dict[str, int] = META_PARAMS):
        self.params = params
        self.my_id = 0
        self.width = 0
        self.height = 0
        self.walls: Set[Coord] = set()
        self.snakebots_per_player = 0
        self.my_snakebot_ids: List[int] = []
        self.opp_snakebot_ids: List[int] = []
        self.power_sources: Set[Coord] = set()
        self.snakebots: List[SnakeBot] = []
        self.position_history: Dict[int, List[Tuple[Coord, ...]]] = {}
        self.last_directions: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load_initial_state(self, _input=input):
        self.my_id = int(_input())
        self.width = int(_input())
        self.height = int(_input())
        self.walls.clear()
        for y in range(self.height):
            row = _input()
            for x, cell in enumerate(row):
                if cell == PLATFORM:
                    self.walls.add((x, y))
        self.snakebots_per_player = int(_input())
        self.my_snakebot_ids = [int(_input()) for _ in range(self.snakebots_per_player)]
        self.opp_snakebot_ids = [int(_input()) for _ in range(self.snakebots_per_player)]

    def update(self, _input=input):
        self.power_sources.clear()
        self.snakebots = []
        power_source_count = int(_input())
        for _ in range(power_source_count):
            x, y = map(int, _input().split())
            self.power_sources.add((x, y))
        snakebot_count = int(_input())
        for _ in range(snakebot_count):
            snakebot_id_str, raw_body = _input().split()
            snakebot_id = int(snakebot_id_str)
            body = parse_body(raw_body)
            direction = infer_direction(body, self.last_directions.get(snakebot_id, UP))
            self.last_directions[snakebot_id] = direction
            self.snakebots.append(SnakeBot(snakebot_id, body, direction))
            history = self.position_history.setdefault(snakebot_id, [])
            state = tuple(body)
            if not history or history[-1] != state:
                history.append(state)
                del history[:-12]

    # ------------------------------------------------------------------
    # Game mechanics (gravity, movement)
    # ------------------------------------------------------------------

    def my_snakebots(self) -> List[SnakeBot]:
        return [s for s in self.snakebots if s.id in self.my_snakebot_ids]

    def opp_snakebots(self) -> List[SnakeBot]:
        return [s for s in self.snakebots if s.id in self.opp_snakebot_ids]

    def is_viable_cell(self, coord: Coord) -> bool:
        x, y = coord
        return 0 <= x < self.width and y < self.height

    def current_occupied(self, exclude_id: Optional[int] = None) -> Set[Coord]:
        occupied: Set[Coord] = set()
        for snake in self.snakebots:
            if snake.id == exclude_id:
                continue
            occupied.update(snake.body)
        return occupied

    def has_support(self, body: List[Coord], solids: Set[Coord]) -> bool:
        body_cells = set(body)
        for x, y in body:
            below = (x, y + 1)
            if below in body_cells:
                continue
            if below in solids:
                return True
        return False

    def gravity_land(self, x: int, y: int, solids: Set[Coord]) -> Tuple[Coord, bool]:
        """Drop a single point until it lands on a solid or falls off the map."""
        while True:
            if (x, y + 1) in solids:
                return (x, y), True
            y += 1
            if y >= self.height + 1:
                return (x, y), False

    def apply_gravity(
        self,
        body: List[Coord],
        snake_id: int,
        remaining_power_sources: Set[Coord],
        reserved: Set[Coord],
    ) -> Tuple[List[Coord], bool, int]:
        fall_distance = 0
        solids = set(self.walls)
        solids.update(remaining_power_sources)
        solids.update(self.current_occupied(exclude_id=snake_id))
        solids.update(reserved)
        while not self.has_support(body, solids):
            body = [(x, y + 1) for x, y in body]
            fall_distance += 1
            if all(y >= self.height + 1 for _, y in body):
                return body, False, fall_distance
        return body, True, fall_distance

    def candidate_directions(self, snake: SnakeBot) -> List[str]:
        forbidden = OPPOSITE.get(snake.direction)
        return [d for d in TURN_ORDER if d != forbidden]

    def move_body(self, snake: SnakeBot, direction: str, grow: bool) -> List[Coord]:
        new_head = add_coord(snake.head(), DIRECTIONS[direction])
        if grow:
            return [new_head] + snake.body[:]
        return [new_head] + snake.body[:-1]

    def move_blockers(self, snake: SnakeBot, grow: bool, reserved: Set[Coord]) -> Set[Coord]:
        blockers = set(self.walls)
        blockers.update(self.current_occupied(exclude_id=snake.id))
        blockers.update(reserved)
        if grow:
            blockers.update(snake.body[1:])
        else:
            blockers.update(snake.body[1:-1])
        return blockers

    def enemy_viable_moves(self, opp: SnakeBot) -> List[Coord]:
        opp_dir = self.last_directions.get(opp.id, UP)
        occupied = self.current_occupied(exclude_id=opp.id)
        blockers = self.walls | occupied
        viable = []
        for d in TURN_ORDER:
            if d == OPPOSITE.get(opp_dir):
                continue
            target = add_coord(opp.head(), DIRECTIONS[d])
            if not self.is_viable_cell(target):
                continue
            if target in blockers and target not in self.power_sources:
                continue
            viable.append(target)
        return viable

    def enemy_danger_zones(self) -> Set[Coord]:
        zones: Set[Coord] = set()
        for opp in self.opp_snakebots():
            zones.update(self.enemy_viable_moves(opp))
        return zones

    # ------------------------------------------------------------------
    # Strategic helpers
    # ------------------------------------------------------------------

    def gravity_kills_enemy(self, eaten_power: Optional[Coord]) -> List[int]:
        if eaten_power is None:
            return []
        killed = []
        for opp in self.opp_snakebots():
            solids_after = set(self.walls) | (self.power_sources - {eaten_power})
            for s in self.snakebots:
                if s.id != opp.id:
                    solids_after.update(s.body)
            if not self.has_support(opp.body, solids_after):
                test_body = opp.body[:]
                while True:
                    test_body = [(x, y + 1) for x, y in test_body]
                    if all(y >= self.height + 1 for _, y in test_body):
                        killed.append(opp.id)
                        break
                    if self.has_support(test_body, solids_after):
                        break
        return killed

    def contest_power_bonus_manhattan(
        self, head: Coord, remaining_power_sources: Set[Coord]
    ) -> int:
        """Fast contest check: am I closer (Manhattan) to any power than all enemies?"""
        if not remaining_power_sources:
            return 0
        my_dist = min(manhattan_distance(head, ps) for ps in remaining_power_sources)
        bonus = 0
        for opp in self.opp_snakebots():
            opp_dist = min(manhattan_distance(opp.head(), ps) for ps in remaining_power_sources)
            if my_dist < opp_dist:
                bonus += self.params["contest_closer_bonus"]
            elif my_dist == opp_dist:
                bonus += self.params["contest_tie_bonus"]
        return bonus

    # ------------------------------------------------------------------
    # Loop escape detection (same as best.py)
    # ------------------------------------------------------------------

    def recent_heads(self, snake_id: int, limit: int = 8) -> List[Coord]:
        return [state[0] for state in self.position_history.get(snake_id, [])[-limit:] if state]

    def loop_escape_target_x(self, snake_id: int) -> Optional[int]:
        recent = self.recent_heads(snake_id)
        if len(recent) < 6:
            return None
        if len(set(recent)) > max(3, len(recent) // 2):
            return None
        midpoint = self.width // 2
        min_x = min(x for x, _ in recent)
        max_x = max(x for x, _ in recent)
        if max_x < midpoint:
            return self.width - 1
        if min_x >= midpoint:
            return 0
        return None

    def head_collision_value(self, head: Coord, snake_len: int) -> int:
        if snake_len < self.params["head_collision_min_length"]:
            return 0
        for opp in self.opp_snakebots():
            viable = self.enemy_viable_moves(opp)
            if head in viable:
                if len(opp.body) <= self.params["head_collision_short_enemy_max_length"]:
                    return self.params["head_collision_short_enemy_bonus"]
                if snake_len > len(opp.body) + self.params["head_collision_size_margin"]:
                    return self.params["head_collision_larger_bonus"]
        return 0

    def voronoi_territory(
        self,
        my_head: Coord,
        opp_heads: List[Coord],
        movement_blockers: Set[Coord],
        power_sources: Set[Coord],
        limit: int,
    ) -> Tuple[int, int]:
        """
        Simultaneous BFS from my_head and all opp_heads respecting gravity.
        Returns (my_cells, enemy_cells) — cells closer to me vs closer to enemy.
        Power sources provide gravity support but are NOT movement blockers.
        """
        # solids = walls + occupied + power_sources (for gravity_land support check)
        solids = movement_blockers | power_sources

        # owner: 0 = me, 1 = enemy
        dist_owner: Dict[Coord, int] = {}
        queue: deque = deque()  # (coord, owner)

        def try_enqueue(head: Coord, owner: int) -> None:
            landed, alive = self.gravity_land(head[0], head[1], solids)
            if alive and landed not in dist_owner:
                dist_owner[landed] = owner
                queue.append((landed, owner))

        try_enqueue(my_head, 0)
        for h in opp_heads:
            try_enqueue(h, 1)

        my_cells = 0
        enemy_cells = 0
        total = 0

        while queue and total < limit:
            (px, py), owner = queue.popleft()
            total += 1
            if owner == 0:
                my_cells += 1
            else:
                enemy_cells += 1

            for dx, dy in DIRECTIONS.values():
                nx, ny = px + dx, py + dy
                if nx < 0 or nx >= self.width or ny < 0:
                    continue
                # movement blocked by walls or occupied cells (power sources passable)
                if (nx, ny) in movement_blockers:
                    continue
                landed, alive = self.gravity_land(nx, ny, solids)
                if not alive:
                    continue
                if landed not in dist_owner:
                    dist_owner[landed] = owner
                    queue.append((landed, owner))

        return my_cells, enemy_cells

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_move(
        self,
        snake: SnakeBot,
        direction: str,
        body: List[Coord],
        alive: bool,
        ate: bool,
        collided: bool,
        fall_distance: int,  # noqa: ARG002 — kept for API symmetry
        remaining_power_sources: Set[Coord],
    ) -> int:
        if not alive:
            return SCORE_NEG_INF

        head = body[0]
        score = 0

        if ate:
            score += self.params["eat_bonus"]
        if collided:
            score -= 500

        if direction == snake.direction:
            score += self.params["momentum_bonus"]

        # Repeat state penalty
        if tuple(body) in self.position_history.get(snake.id, []):
            score -= self.params["repeat_state_penalty"]

        # Danger zones + head collision
        danger_zones = self.enemy_danger_zones()
        collision_val = self.head_collision_value(head, len(body))
        if collision_val > 0:
            score += collision_val
        elif head in danger_zones:
            score -= self.params["danger_penalty"]
        for bx, by in body[1:]:
            if (bx, by) in danger_zones:
                score -= self.params["body_danger_penalty"]

        # Gravity kill
        if ate:
            new_head = add_coord(snake.head(), DIRECTIONS[direction])
            killed = self.gravity_kills_enemy(new_head)
            if killed:
                score += self.params["gravity_kill_bonus"] * len(killed)

        # Exit count (dead-end avoidance)
        body_set = set(body)
        blockers_self = self.walls | self.current_occupied(exclude_id=snake.id) | body_set
        my_exits = 0
        for d in TURN_ORDER:
            if d == OPPOSITE.get(direction):
                continue
            target = add_coord(head, DIRECTIONS[d])
            if not self.is_viable_cell(target):
                continue
            if target not in blockers_self:
                my_exits += 1
        if my_exits == 0:
            score -= self.params["no_exit_penalty"]
        elif my_exits == 1:
            score -= self.params["one_exit_penalty"]

        # *** Voronoi territory — primary differentiator ***
        # movement_blockers = walls + occupied (power sources intentionally excluded)
        movement_blockers = self.walls | self.current_occupied(exclude_id=snake.id) | body_set
        opp_heads = [opp.head() for opp in self.opp_snakebots()]
        my_cells, enemy_cells = self.voronoi_territory(
            head, opp_heads, movement_blockers, remaining_power_sources,
            self.params["territory_limit"],
        )
        score += self.params["territory_weight"] * my_cells
        score -= self.params["enemy_territory_penalty"] * enemy_cells

        # Loop escape (copy from best.py — avoids endless circling)
        loop_target_x = self.loop_escape_target_x(snake.id)
        if loop_target_x is not None:
            current_dist = abs(snake.head()[0] - loop_target_x)
            next_dist = abs(head[0] - loop_target_x)
            progress = current_dist - next_dist
            score += self.params["score_loop_escape_progress_bonus"] * progress
            if progress > 0:
                score += self.params["score_loop_escape_directional_bonus"]
            elif progress < 0:
                score -= self.params["score_loop_escape_directional_bonus"]

        # Power distance (Manhattan — fast, Voronoi already handles proximity)
        if remaining_power_sources:
            min_dist = min(manhattan_distance(head, ps) for ps in remaining_power_sources)
            score -= self.params["power_distance_penalty"] * min_dist
            score += self.contest_power_bonus_manhattan(head, remaining_power_sources)
        else:
            score += self.params["no_power_bonus"]

        return score

    # ------------------------------------------------------------------
    # Simulation + lookahead
    # ------------------------------------------------------------------

    def simulate_move(self, snake: SnakeBot, direction: str, reserved: Set[Coord]) -> SimulationResult:
        new_head = add_coord(snake.head(), DIRECTIONS[direction])
        apple_at_target = new_head in self.power_sources
        moved_body = self.move_body(snake, direction, apple_at_target)

        collided = new_head in self.move_blockers(snake, apple_at_target, reserved)
        if collided:
            if len(moved_body) <= 3:
                return SimulationResult(
                    direction, moved_body, False, False, True, 0,
                    set(self.power_sources), SCORE_NEG_INF,
                )
            moved_body = moved_body[1:]

        remaining_power_sources = set(self.power_sources)
        if apple_at_target:
            remaining_power_sources.discard(new_head)

        grew = apple_at_target and not collided
        stable_body, alive, fall_distance = self.apply_gravity(
            moved_body, snake.id, remaining_power_sources, reserved,
        )
        score = self.score_move(
            snake, direction, stable_body, alive, grew, collided,
            fall_distance, remaining_power_sources,
        )
        return SimulationResult(
            direction, stable_body, alive, grew, collided,
            fall_distance, remaining_power_sources, score,
        )

    def evaluate_at_depth(
        self, snake: SnakeBot, reserved: Set[Coord], depth: int
    ) -> Tuple[int, str]:
        best_score = SCORE_NEG_INF
        best_direction = snake.direction

        for direction in self.candidate_directions(snake):
            result = self.simulate_move(snake, direction, reserved)

            if not result.alive or depth <= 1:
                if result.score > best_score:
                    best_score = result.score
                    best_direction = direction
                continue

            temp_snake = SnakeBot(snake.id, result.body, result.direction)
            saved_power = self.power_sources
            self.power_sources = result.remaining_power_sources
            future_score, _ = self.evaluate_at_depth(temp_snake, reserved, depth - 1)
            self.power_sources = saved_power

            combined = result.score + future_score // self.params["lookahead_future_score_divisor"]
            if combined > best_score:
                best_score = combined
                best_direction = direction

        return best_score, best_direction

    def choose_action(self, snake: SnakeBot, reserved: Set[Coord]) -> SimulationResult:
        candidates = self.candidate_directions(snake)
        alive_count = sum(
            1 for d in candidates if self.simulate_move(snake, d, reserved).alive
        )
        depth = 3 if alive_count <= 2 else 2
        _, best_direction = self.evaluate_at_depth(snake, reserved, depth=depth)
        result = self.simulate_move(snake, best_direction, reserved)
        self.last_directions[snake.id] = result.direction
        return result

    def play(self):
        actions = []
        reserved: Set[Coord] = set()
        my_snakes = sorted(self.my_snakebots(), key=lambda s: (-len(s.body), s.id))
        for snake in my_snakes:
            result = self.choose_action(snake, reserved)
            if result.alive:
                reserved.update(result.body)
            actions.append(f"{snake.id} {result.direction}")
            debug(
                snake.id, snake.direction, "->", result.direction,
                "score:", result.score, "ate:", result.ate, "alive:", result.alive,
            )
        print(";".join(actions) if actions else "WAIT")


def main():
    game = Game(META_PARAMS)
    game.load_initial_state()
    while True:
        game.update()
        game.play()


if __name__ == "__main__":
    main()
