#!/usr/bin/env python3
"""Skeleton bot — minimal template for a Snakebird bot.

This bot reads the game state, picks the direction that moves each snake's head
closest to the nearest power source (Manhattan distance), and avoids walls.

Use this as a starting point for your own bot: add lookahead, territory
evaluation, collision avoidance, gravity handling, etc.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Set, Tuple

EMPTY = "."
WALL = "#"

DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


def debug(*args):
    """Print debug info to stderr (stdout is reserved for commands)."""
    print(*args, file=sys.stderr, flush=True)


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class SnakeBot:
    def __init__(self, snake_id: int, body: List[Tuple[int, int]]):
        self.id = snake_id
        self.body = body

    def head(self) -> Tuple[int, int]:
        return self.body[0]


META_PARAMS: Dict[str, int] = {}


class Game:
    def __init__(self, params: Dict[str, int]):
        self.params = params
        self.my_id = 0
        self.width = 0
        self.height = 0
        self.walls: Set[Tuple[int, int]] = set()
        self.my_snake_ids: List[int] = []
        self.opp_snake_ids: List[int] = []
        self.power_sources: List[Tuple[int, int]] = []
        self.snakes: List[SnakeBot] = []

    def load_initial_state(self, _input=input):
        self.my_id = int(_input())
        self.width = int(_input())
        self.height = int(_input())
        for y in range(self.height):
            row = _input()
            for x, cell in enumerate(row):
                if cell == WALL:
                    self.walls.add((x, y))
        snakes_per_player = int(_input())
        self.my_snake_ids = [int(_input()) for _ in range(snakes_per_player)]
        self.opp_snake_ids = [int(_input()) for _ in range(snakes_per_player)]

    def update(self, _input=input):
        self.power_sources = []
        self.snakes = []
        power_count = int(_input())
        for _ in range(power_count):
            x, y = map(int, _input().split())
            self.power_sources.append((x, y))
        snake_count = int(_input())
        for _ in range(snake_count):
            parts = _input().split()
            snake_id = int(parts[0])
            body = []
            for raw in parts[1].split(":"):
                cx, cy = raw.split(",")
                body.append((int(cx), int(cy)))
            self.snakes.append(SnakeBot(snake_id, body))

    def my_snakes(self) -> List[SnakeBot]:
        return [s for s in self.snakes if s.id in self.my_snake_ids]

    def available_moves(self, snake: SnakeBot) -> List[str]:
        """Return directions that don't immediately hit a wall or go off-grid."""
        moves = []
        hx, hy = snake.head()
        occupied = set()
        for s in self.snakes:
            occupied.update(s.body)
        for direction, (dx, dy) in DIRECTIONS.items():
            nx, ny = hx + dx, hy + dy
            if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                continue
            if (nx, ny) in self.walls:
                continue
            if (nx, ny) in occupied:
                continue
            moves.append(direction)
        return moves

    def best_move(self, snake: SnakeBot) -> str:
        """Pick the move that gets closest to the nearest power source."""
        moves = self.available_moves(snake)
        if not moves:
            return "UP"  # fallback — doomed anyway
        if not self.power_sources:
            return moves[0]

        best_dir = moves[0]
        best_dist = float("inf")
        hx, hy = snake.head()
        for direction in moves:
            dx, dy = DIRECTIONS[direction]
            nx, ny = hx + dx, hy + dy
            for ps in self.power_sources:
                d = manhattan((nx, ny), ps)
                if d < best_dist:
                    best_dist = d
                    best_dir = direction
        return best_dir

    def play(self):
        actions = []
        for snake in self.my_snakes():
            direction = self.best_move(snake)
            actions.append(f"{snake.id} {direction}")
        print(";".join(actions) if actions else "WAIT")


def main():
    game = Game({})
    game.load_initial_state()
    while True:
        game.update()
        game.play()


if __name__ == "__main__":
    main()
