#!/usr/bin/env python3
"""Display a Snakebird game recorded with simulator.py --output in terminal text mode.

Usage:
    python simulator/display.py a.txt
    python simulator/display.py a.txt --delay 0.1
    python simulator/display.py a.txt -i          # interactive, press Enter to advance
    python simulator/display.py a.txt --game 2    # second game in file
    python simulator/display.py a.txt --no-color
"""

from __future__ import annotations

import argparse
import ast

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
import platform
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, cast

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_GREEN = "\033[92m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BRIGHT_RED = "\033[91m"
BRIGHT_YELLOW = "\033[93m"
YELLOW = "\033[33m"
WHITE = "\033[37m"
DIM = "\033[2m"

_USE_COLOR = True


def col(code: str, text: str) -> str:
    return f"{code}{text}{RESET}" if _USE_COLOR else text


def clear() -> None:
    print("\033[H\033[2J", end="", flush=True)


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def parse_list_line(line: str) -> List[str]:
    return ast.literal_eval(line.strip())


def load_games(path: Path) -> List[Tuple[Optional[str], List[str], List[List[str]]]]:
    """Return list of (name, global_lines, [frame_lines_per_turn])."""
    games: List[Tuple[Optional[str], List[str], List[List[str]]]] = []
    current_name: Optional[str] = None
    current_global: Optional[List[str]] = None
    current_frames: List[List[str]] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("[") or raw.startswith("#") or raw.strip() == "":
            line = raw.strip()
            if not line:
                if current_global is not None:
                    games.append((current_name, current_global, current_frames))
                current_name = None
                current_global = None
                current_frames = []
                continue
            if line.startswith("#"):
                current_name = line[1:].strip()
                continue
            data = parse_list_line(line)
            if current_global is None:
                current_global = data
            else:
                current_frames.append(data)

    if current_global is not None:
        games.append((current_name, current_global, current_frames))

    return games


# ---------------------------------------------------------------------------
# Game state parsing
# ---------------------------------------------------------------------------

def parse_global(gl: List[str]):
    """Return (player_index, width, height, grid_rows, my_bird_ids, opp_bird_ids)."""
    player_index = int(gl[0])
    width = int(gl[1])
    height = int(gl[2])
    grid_rows = gl[3:3 + height]
    nb_birds = int(gl[3 + height])
    my_ids = [int(gl[3 + height + 1 + i]) for i in range(nb_birds)]
    opp_ids = [int(gl[3 + height + 1 + nb_birds + i]) for i in range(nb_birds)]
    return player_index, width, height, grid_rows, my_ids, opp_ids


def parse_frame(fl: List[str]) -> Tuple[Set[Tuple[int, int]], Dict[int, List[Tuple[int, int]]]]:
    """Return (apples, {bird_id: [(x,y), ...] head-first})."""
    nb_apples = int(fl[0])
    apples: Set[Tuple[int, int]] = set()
    for i in range(nb_apples):
        x, y = fl[1 + i].split()
        apples.add((int(x), int(y)))
    nb_birds = int(fl[1 + nb_apples])
    birds: Dict[int, List[Tuple[int, int]]] = {}
    for i in range(nb_birds):
        bid_str, body_str = fl[1 + nb_apples + 1 + i].split(" ", 1)
        bid = int(bid_str)
        body = [
            (int(c.split(",")[0]), int(c.split(",")[1]))
            for c in body_str.split(":")
        ]
        birds[bid] = body
    return apples, birds


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

COLORS_P0 = [BRIGHT_CYAN, BRIGHT_GREEN, CYAN, GREEN]
COLORS_P1 = [BRIGHT_RED, MAGENTA, RED, BRIGHT_YELLOW]

# One head char and one body char per bird index within a team
HEAD_CHARS = "@ABCDE"
BODY_CHARS = "oabcde"


def render(
    grid_rows: List[str],
    width: int,
    height: int,
    my_ids: List[int],
    opp_ids: List[int],
    apples: Set[Tuple[int, int]],
    birds: Dict[int, List[Tuple[int, int]]],
    turn: int,
    total: int,
) -> str:
    my_set = set(my_ids)

    # Build cell map
    cells: Dict[Tuple[int, int], str] = {}

    for bird_id, body in birds.items():
        if bird_id in my_set:
            idx = my_ids.index(bird_id)
            color = COLORS_P0[idx % len(COLORS_P0)]
        else:
            idx = opp_ids.index(bird_id)
            color = COLORS_P1[idx % len(COLORS_P1)]
        head_ch = HEAD_CHARS[min(idx, len(HEAD_CHARS) - 1)]
        body_ch = BODY_CHARS[min(idx, len(BODY_CHARS) - 1)]
        for pos in reversed(body):
            cells[pos] = col(color, body_ch)
        cells[body[0]] = col(color + BOLD, head_ch)

    for ax, ay in apples:
        cells[(ax, ay)] = col(YELLOW, "*")

    out: List[str] = []

    # Header
    out.append(
        col(BOLD, f"Turn {turn}/{total}")
        + f"  apples={len(apples)}"
        + f"  live birds={len(birds)}"
    )

    # Column ruler (tens digit every 10)
    ruler = "  " + "".join(str(x // 10) if x % 10 == 0 else " " for x in range(width))
    out.append(col(DIM, ruler))
    ruler2 = "  " + "".join(str(x % 10) for x in range(width))
    out.append(col(DIM, ruler2))

    for y, row in enumerate(grid_rows):
        line_chars: List[str] = []
        for x, ch in enumerate(row):
            pos = (x, y)
            if pos in cells:
                line_chars.append(cells[pos])
            elif ch == "#":
                line_chars.append(col(WHITE, "#"))
            else:
                line_chars.append(col(DIM, "."))
        row_label = col(DIM, f"{y:2}")
        out.append(row_label + "".join(line_chars))

    # Legend
    out.append("")
    p0_parts = [
        col(COLORS_P0[i % len(COLORS_P0)] + BOLD, f"P0[{bid}]")
        + (f"({len(birds[bid])})" if bid in birds else col(DIM, "(dead)"))
        for i, bid in enumerate(my_ids)
    ]
    p1_parts = [
        col(COLORS_P1[i % len(COLORS_P1)] + BOLD, f"P1[{bid}]")
        + (f"({len(birds[bid])})" if bid in birds else col(DIM, "(dead)"))
        for i, bid in enumerate(opp_ids)
    ]
    out.append("  ".join(p0_parts) + "   |   " + "  ".join(p1_parts))

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Raw keypress (cross-platform)
# ---------------------------------------------------------------------------

def _read_key() -> str:
    """Read a single keypress. Returns 'right', 'left', 'q', 'enter', or the char."""
    if platform.system() == "Windows":
        import msvcrt

        getch = cast(Callable[[], bytes], getattr(msvcrt, "getch"))
        win_ch = getch()
        if win_ch in (b"\xe0", b"\x00"):  # special key prefix
            special = getch()
            return {b"M": "right", b"K": "left", b"H": "up", b"P": "down"}.get(special, "")
        if win_ch == b"\r":
            return "enter"
        return win_ch.decode("utf-8", errors="replace")
    else:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            char = sys.stdin.read(1)
            if char == "\x1b":
                next_char = sys.stdin.read(1)
                if next_char == "[":
                    arrow_char = sys.stdin.read(1)
                    return {"C": "right", "D": "left", "A": "up", "B": "down"}.get(arrow_char, "")
                return ""
            if char in ("\r", "\n"):
                return "enter"
            return char
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def display_game(
    name: Optional[str],
    global_lines: List[str],
    frames: List[List[str]],
    delay: float,
    interactive: bool,
) -> None:
    _player_index, width, height, grid_rows, my_ids, opp_ids = parse_global(global_lines)
    total = len(frames)
    title = name or "game"
    turn = 0  # 0-based index

    while 0 <= turn < total:
        apples, birds = parse_frame(frames[turn])
        clear()
        print(col(BOLD, f"=== {title} ==="))
        print(render(grid_rows, width, height, my_ids, opp_ids, apples, birds, turn + 1, total))

        if interactive:
            print()
            print(col(DIM, "[→/Enter] next  [←] prev  [q] quit"))
            try:
                key = _read_key()
            except (EOFError, KeyboardInterrupt):
                break
            if key == "q":
                break
            elif key in ("left", "up"):
                turn = max(0, turn - 1)
            else:
                turn = min(total - 1, turn + 1) if turn == total - 1 else turn + 1
        else:
            if turn < total - 1:
                time.sleep(delay)
                turn += 1
            else:
                # Last frame: wait for a key
                print()
                print(col(DIM, "[→/Enter] restart  [←] prev  [q] quit"))
                try:
                    key = _read_key()
                except (EOFError, KeyboardInterrupt):
                    break
                if key == "q":
                    break
                elif key in ("left", "up"):
                    turn = max(0, turn - 1)
                else:
                    turn = 0  # restart

    clear()
    apples, birds = parse_frame(frames[-1])
    print(col(BOLD, f"=== {title} — final state ==="))
    print(render(grid_rows, width, height, my_ids, opp_ids, apples, birds, total, total))
    print()
    print(col(BOLD, "Game over."))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global _USE_COLOR

    parser = argparse.ArgumentParser(description="Display a recorded Snakebird game in terminal text mode")
    parser.add_argument("input", type=Path, help="Game file written by simulator --output")
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds between frames in auto mode (default 0.2)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Step frame by frame with arrow keys")
    parser.add_argument(
        "--game",
        type=int,
        default=1,
        help="Which game to display if file has multiple (1-based, default 1)",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    args = parser.parse_args()

    if args.no_color:
        _USE_COLOR = False

    if not args.input.is_file():
        print(f"File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    games = load_games(args.input)
    if not games:
        print("No games found in file", file=sys.stderr)
        sys.exit(1)

    idx = args.game - 1
    if idx < 0 or idx >= len(games):
        print(f"Game {args.game} not found (file has {len(games)} game(s))", file=sys.stderr)
        sys.exit(1)

    name, global_lines, frames = games[idx]
    if not frames:
        print("No turns recorded in this game", file=sys.stderr)
        sys.exit(1)

    if len(games) > 1:
        print(f"File contains {len(games)} game(s). Displaying game {args.game}: {name or '(unnamed)'}")
        print("Use --game N to select another.\n")
        time.sleep(0.5)

    display_game(name, global_lines, frames, args.delay, args.interactive)


if __name__ == "__main__":
    main()
