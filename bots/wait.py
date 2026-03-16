#!/usr/bin/env python3

from __future__ import annotations

from typing import Dict


class Game:
    def __init__(self, params: Dict[str, int]):
        self.params = params
        self.my_id = 0
        self.width = 0
        self.height = 0
        self.my_bird_ids: list[int] = []
        self.opp_bird_ids: list[int] = []

    def load_initial_state(self, _input=input):
        self.my_id = int(_input())
        self.width = int(_input())
        self.height = int(_input())
        for _ in range(self.height):
            _input()

        birds_per_player = int(_input())
        self.my_bird_ids = [int(_input()) for _ in range(birds_per_player)]
        self.opp_bird_ids = [int(_input()) for _ in range(birds_per_player)]

    def update(self, _input=input):
        apple_count = int(_input())
        for _ in range(apple_count):
            _input()

        bird_count = int(_input())
        for _ in range(bird_count):
            _input()

    def play(self):
        print("WAIT")


def main():
    game = Game({})
    game.load_initial_state()

    while True:
        game.update()
        game.play()


if __name__ == "__main__":
    main()
