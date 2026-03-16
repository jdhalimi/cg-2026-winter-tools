# Snakebird Simulation Toolkit — CG Winter Challenge 2026

A local simulation toolkit for the [CodinGame Winter Challenge 2026](https://www.codingame.com/) (Snakebird / "Robot-Serpents"). Run bot-vs-bot matches locally, replay games in the terminal, and analyse replays — all without needing the CodinGame platform.

## Contents

```
public/
├── bots/
│   ├── skeleton.py       # Minimal bot template — start here
│   ├── wait.py            # No-op bot (outputs WAIT every turn)
│   └── explorer.py        # Territory-control bot (Voronoi flood-fill, ~600 lines)
├── simulator/
│   ├── simulator.py       # Game engine — runs bot-vs-bot matches locally
│   ├── analyse.py         # Replay analyser — events, metrics, optimization hints
│   ├── display.py         # Terminal replay viewer with ANSI colours
│   └── data/
│       └── maps.txt       # 10 pre-generated test maps
├── LICENSE                # MIT
└── README.md              # This file
```

## Requirements

- **Python 3.10+** (no third-party dependencies — pure stdlib)

## Quick Start

All commands below assume you are inside the `public/` directory.

### 1. Run a match between two bots

```bash
# Random-seed map generation
python simulator/simulator.py bots/skeleton.py bots/wait.py --seed 0 --league-level 4

# Using a pre-generated map file
python simulator/simulator.py bots/skeleton.py bots/explorer.py \
  --maps simulator/data/maps.txt --all-maps
```

Output shows per-game results: winner, scores, losses, and a summary table when running multiple maps.

### 2. Record a match and replay it

```bash
# Record
python simulator/simulator.py bots/explorer.py bots/skeleton.py \
  --maps simulator/data/maps.txt --map 1 --output game.txt

# Replay in terminal (auto-advance)
python simulator/display.py game.txt --delay 0.15

# Replay interactively (arrow keys to step, q to quit)
python simulator/display.py game.txt -i
```

### 3. Generate maps

```bash
python simulator/simulator.py bots/wait.py bots/wait.py \
  --seed 42 --league-level 4 \
  --nb-maps 20 \
  --map-output my_maps.txt
```

### 4. Analyse a CodinGame replay

Save a game log from the CodinGame platform as a text file, then:

```bash
python simulator/analyse.py replay.txt
python simulator/analyse.py replay.txt --verbose   # score evolution per turn
```

The analyser outputs:
- Event timeline (fruits eaten, collisions, deaths)
- Per-player metrics (direction distribution, fruit chase rate, aggression)
- Strategy classification (greedy, tactical, aggressive, defensive)
- Optimization hints tied to bot heuristic parameters

---

## Game Rules (summary)

| Rule | Detail |
|------|--------|
| **Grid** | 2D grid of `.` (empty) and `#` (wall) cells |
| **Players** | 2 players, each controlling 1–4 snake-like robots |
| **Turns** | Simultaneous — both players move at the same time |
| **Movement** | Each snake moves 1 cell in a cardinal direction (UP/DOWN/LEFT/RIGHT). Cannot reverse. |
| **Eating** | If the head lands on a power source, the snake grows (tail stays). Otherwise it slides (tail removed). |
| **Collision** | Head hits wall, own body, or another snake → beheading (lose head cell). Snake with ≤ 3 cells dies. |
| **Gravity** | Unsupported snakes fall until they land on a wall, a power source, or another grounded snake. Falling off the bottom kills the snake. |
| **Win condition** | After all power sources are eaten or a team dies: longest total snake length wins. Tie-breaker: fewest lost cells. |

---

## Bot Interface

Every bot must be a standalone Python file exporting a `Game` class. The simulator imports and drives it automatically.

### Required class structure

```python
from typing import Dict

META_PARAMS: Dict[str, int] = {}   # optional tunable parameters

class Game:
    def __init__(self, params: Dict[str, int]):
        """Called once. `params` = META_PARAMS merged with any weight overrides."""
        ...

    def load_initial_state(self, _input=input):
        """Called once at game start. Read initial state (see protocol below)."""
        ...

    def update(self, _input=input):
        """Called every turn before play(). Read per-turn state."""
        ...

    def play(self):
        """Called every turn after update(). Print exactly one line to stdout."""
        ...

def main():
    game = Game({})
    game.load_initial_state()
    while True:
        game.update()
        game.play()

if __name__ == "__main__":
    main()
```

### Input protocol

**`load_initial_state()`** reads (one value per `_input()` call):

```
player_index            # int: 0 or 1
width                   # int
height                  # int
<row_0>                 # string of length width: '.' = empty, '#' = wall
...
<row_{height-1}>
birds_per_player        # int
<my_bird_id_0>          # int
...
<my_bird_id_{n-1}>
<opp_bird_id_0>         # int
...
<opp_bird_id_{n-1}>
```

**`update()`** reads:

```
apple_count             # int
<x> <y>                 # space-separated apple position × apple_count
bird_count              # int: total live birds (both teams)
<id> <x0,y0:x1,y1:…>   # bird id + colon-separated body coords (head first) × bird_count
```

### Output protocol

**`play()`** must print exactly **one line** of semicolon-separated commands:

```
<bird_id> <DIRECTION>[;<bird_id> <DIRECTION>]*
```

Example: `0 RIGHT;1 DOWN`

Use `WAIT` when all your snakes are dead.

### Important rules

- **stdout** is reserved for commands. All debug output must go to **stderr** (`print(..., file=sys.stderr)`).
- The `_input` parameter lets the simulator inject lines without subprocess I/O. Always use `_input()` instead of `input()` directly.
- Snakes cannot reverse direction (e.g., if facing RIGHT, cannot go LEFT).

---

## Simulator CLI Reference

```
python simulator/simulator.py BOT_A BOT_B [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--seed N` | Java Random seed for map generation (default: 0) |
| `--league-level N` | Grid complexity 1–4 (default: 4) |
| `--max-turns N` | Maximum turns before game ends (default: 200) |
| `--maps FILE` | Load maps from a pre-generated map file |
| `--map N` | Run only map N (1-based) from `--maps` |
| `--map-name NAME` | Run the named map from `--maps` |
| `--all-maps` | Run all maps from `--maps` |
| `--nb-maps N` | Generate N random maps (default: 1) |
| `--map-output FILE` | Save generated/selected maps to file |
| `--output FILE` | Record game frames for replay with `display.py` |
| `--weights FILE` | JSON weight overrides applied to both bots |
| `--weights-a FILE` | JSON weight overrides for player 0 only |
| `--weights-b FILE` | JSON weight overrides for player 1 only |
| `--workers N` | Parallel workers for multi-map runs (0=auto) |

### Display CLI

```
python simulator/display.py GAME_FILE [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--delay N` | Seconds between frames (default: 0.2) |
| `-i` / `--interactive` | Step with arrow keys, q to quit |
| `--game N` | Which game to show if file has multiple (1-based) |
| `--no-color` | Disable ANSI colour codes |

### Analyser CLI

```
python simulator/analyse.py LOG_FILE [--verbose]
```

---

## Included Bots

### `skeleton.py` — Start here

Minimal template (~120 lines). Reads the game state, avoids walls, and greedily moves toward the nearest power source using Manhattan distance. No lookahead, no gravity handling — designed to be extended.

### `wait.py` — Passive baseline

Outputs `WAIT` every turn. Useful as a punching bag to test that your bot actually collects power sources and stays alive.

### `explorer.py` — Territory control

A more advanced bot (~600 lines) that evaluates moves using Voronoi flood-fill (cells reachable before the enemy). Includes:
- Territory scoring (BFS-based)
- Power source contest evaluation
- Head collision detection
- Gravity simulation
- Dead-end avoidance
- Loop escape heuristics
- 2–3 depth lookahead

Good reference for building a competitive bot. Exposes 30+ tunable `META_PARAMS`.

---

## Weight Files

Bots that define `META_PARAMS` can be tuned with JSON weight files:

```json
{
    "eat_bonus": 900,
    "territory_weight": 12,
    "no_exit_penalty": 600
}
```

Pass them to the simulator:

```bash
python simulator/simulator.py bots/explorer.py bots/skeleton.py \
  --weights-a my_weights.json
```

Values in the JSON override the bot's `META_PARAMS` defaults.

---

## Tips for Building a Competitive Bot

1. **Start from `skeleton.py`** — get the I/O protocol right first.
2. **Add gravity simulation** — snakes fall when unsupported. This is critical.
3. **Add lookahead** — simulate 2–3 turns ahead and score each outcome.
4. **Avoid dead-ends** — count exits after each move. Penalise 0 or 1 exits heavily.
5. **Contest power sources** — if you're closer to a fruit than the enemy, prioritise it.
6. **Handle collisions** — sometimes a head-on collision is worth it if you're longer.
7. **Use `display.py`** — watching replays reveals bugs that logs don't.
8. **Test against multiple bots** — run `--all-maps` to find weaknesses across diverse maps.

---

## License

MIT — see [LICENSE](LICENSE).
