"""Generate map initial states from seed/league parameters.

Outputs maps in the same serialized format as maps.txt, without
running any bot — only the grid generation and spawn placement.

Usage:

    # Generate 10 maps starting from seed 0, league 4
    python generate.py --seed 0 --count 10 --league-level 4

    # Generate maps for seeds 42,43,44 at league 3, append to existing file
    python generate.py --seed 42 --count 3 --league-level 3 --append

    # Generate a single map and print to stdout
    python generate.py --seed 7 --league-level 2

    # Output to a specific file
    python generate.py --seed 0 --count 20 --league-level 4 -o my_maps.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from simulator import SnakebirdSimulator, dump_lines_literal


class _NoOpAdapter:
    """Minimal adapter that satisfies the simulator without loading any bot."""

    def __init__(self, path: Path, slot: int, params: Optional[Dict[str, Any]] = None):
        self.path = path

    def initialize(self, lines: Sequence[str]) -> None:
        pass

    def send_frame(self, lines: Sequence[str]) -> None:
        pass

    def execute(self) -> str:
        return "WAIT"


_DUMMY_PATHS = [Path("__generate_dummy__"), Path("__generate_dummy__")]


def generate_map(seed: int, league_level: int) -> Tuple[List[str], List[str]]:
    """Generate a single map and return (global_lines, frame_lines)."""
    sim = SnakebirdSimulator(
        _DUMMY_PATHS,
        seed=seed,
        league_level=league_level,
        adapter_factory=_NoOpAdapter,
    )
    return (
        sim.serialize_global_info_for(sim.players[0]),
        sim.serialize_frame_info_for(sim.players[0]),
    )


def generate_maps(
    seed: int,
    count: int,
    league_level: int,
) -> List[Tuple[str, List[str], List[str]]]:
    """Generate multiple maps and return list of (name, global_lines, frame_lines)."""
    results: List[Tuple[str, List[str], List[str]]] = []
    for i in range(count):
        current_seed = seed + i
        gl, fl = generate_map(current_seed, league_level)
        name = f"generated seed={current_seed} league={league_level}"
        results.append((name, gl, fl))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Snakebird map initial states from seed/league parameters."
    )
    parser.add_argument("--seed", type=int, default=0, help="Starting seed (default: 0)")
    parser.add_argument("--count", type=int, default=1, help="Number of maps to generate (default: 1)")
    parser.add_argument("--league-level", type=int, default=4, choices=[1, 2, 3, 4], help="League level (default: 4)")
    parser.add_argument("-o", "--output", type=Path, help="Output file path (default: stdout)")
    parser.add_argument("--append", action="store_true", help="Append to output file instead of overwriting")
    return parser.parse_args()


def format_output(maps: Sequence[Tuple[str, List[str], List[str]]]) -> str:
    """Format maps in the same text format as maps.txt."""
    output_lines: List[str] = []
    for index, (name, global_lines, frame_lines) in enumerate(maps):
        if index > 0:
            output_lines.append("")
        output_lines.append(f"# {name}")
        output_lines.append(dump_lines_literal(global_lines))
        output_lines.append(dump_lines_literal(frame_lines))
    return "\n".join(output_lines) + "\n"


def main() -> None:
    args = parse_args()
    maps = generate_maps(args.seed, args.count, args.league_level)
    text = format_output(maps)

    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.append else "w"
        with open(args.output, mode, encoding="utf-8") as f:
            if args.append:
                f.write("\n")
            f.write(text)
        print(
            f"Wrote {len(maps)} map(s) to {args.output}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
