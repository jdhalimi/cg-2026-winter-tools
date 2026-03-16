#!/usr/bin/env python3
"""Analyseur de parties Snakebird (CG 2026 Winter)

Parse un fichier de log CodinGame (format replay textuel) et produit :
  - une chronologie des événements (fruits mangés, collisions, morts)
  - des métriques par snake / joueur
  - une tentative de classification de stratégie pour les deux bots
  - des pistes d'optimisation pour le joueur analysé

Format du log attendu (game.txt) :
  Ligne 1  : en-tête de colonnes CodinGame
  Lignes 2-7 : rang+suffixe+nom pour chaque joueur
  Puis N blocs par tour :
    Sortie d'erreur :
    [liste Python de l'état reçu par le bot P1]
    <lignes de debug bot> : id dir_actuelle dir_choisie score fall_dist ate collided body
    Sortie standard :
    <mouvements P1>  (ex: "0 LEFT;1 UP;2 RIGHT")
    <tour_courant>
    <total_tours>
    Sortie standard :
    <mouvements P2>

Usage :
    python simulator/analyse.py outputs/game.txt
    python simulator/analyse.py outputs/game.txt --verbose
"""

from __future__ import annotations

import ast
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ─────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────

DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
SCORE_NEG_INF = -(10**9)

Coord = Tuple[int, int]

# Regex pour une ligne de debug bot :
#   id dir_actuelle dir_choisie score fall_dist ate collided [(x,y), ...]
SNAKE_DEBUG_RE = re.compile(
    r"^(\d+)\s+(UP|DOWN|LEFT|RIGHT)\s+(UP|DOWN|LEFT|RIGHT)\s+(-?\d+)\s+(\d+)\s+(True|False)\s+(True|False)\s+\[(.+)\]$"
)
MOVE_ITEM_RE = re.compile(r"(\d+)\s+(UP|DOWN|LEFT|RIGHT)")


# ─────────────────────────────────────────────────────────────
# Structures de données
# ─────────────────────────────────────────────────────────────

@dataclass
class SnakeDebug:
    """Résultat de l'analyse interne du bot pour un snake, un tour."""
    id: int
    current_dir: str      # direction du snake AVANT le mouvement (inférée)
    chosen_dir: str       # direction choisie par le bot (= mouvement joué)
    score: int            # score d'évaluation interne
    fall_distance: int    # distance de chute après le mouvement simulé
    ate: bool             # le snake a-t-il mangé un fruit dans la simulation ?
    collided: bool        # le bot a-t-il détecté une collision inévitable ?
    projected_body: List[Coord]  # corps projeté après le mouvement simulé (tête en premier)

    def head(self) -> Coord:
        return self.projected_body[0]

    def length(self) -> int:
        return len(self.projected_body)

    def is_doomed(self) -> bool:
        return self.score <= SCORE_NEG_INF


@dataclass
class TurnData:
    turn: int
    total_turns: int
    # État réel reçu par le bot (fruits + corps des snakes AVANT les mouvements du tour)
    fruits: Set[Coord]
    snakes_actual: Dict[int, List[Coord]]   # snake_id → corps réel (depuis game state)
    # Analyse interne du bot (P1 uniquement)
    debug: Dict[int, SnakeDebug]            # snake_id → debug
    # Mouvements joués
    p1_moves: Dict[int, str]               # snake_id → direction
    p2_moves: Dict[int, str]


@dataclass
class GameLog:
    p1_name: str
    p2_name: str
    p1_rank: int
    p2_rank: int
    width: int
    height: int
    walls: Set[Coord]
    p1_snake_ids: List[int]
    p2_snake_ids: List[int]
    turns: List[TurnData] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────

def _parse_body_from_list(raw: str) -> List[Coord]:
    """Parse '[(4, 7), (5, 8)]' ou '[(4,7),(5,8)]' en liste de Coord."""
    try:
        parsed = ast.literal_eval("[" + raw + "]")
        return [(int(x), int(y)) for x, y in parsed]
    except Exception:
        return []


def _parse_snake_debug(line: str) -> Optional[SnakeDebug]:
    m = SNAKE_DEBUG_RE.match(line.strip())
    if not m:
        return None
    return SnakeDebug(
        id=int(m.group(1)),
        current_dir=m.group(2),
        chosen_dir=m.group(3),
        score=int(m.group(4)),
        fall_distance=int(m.group(5)),
        ate=m.group(6) == "True",
        collided=m.group(7) == "True",
        projected_body=_parse_body_from_list(m.group(8)),
    )


def _parse_moves(line: str) -> Dict[int, str]:
    return {int(m.group(1)): m.group(2) for m in MOVE_ITEM_RE.finditer(line)}


def _parse_turn_state(state_list: List[str]) -> Tuple[Set[Coord], Dict[int, List[Coord]]]:
    """
    Parse une liste d'état de tour (fruits + corps des snakes).
    Format : [nb_fruits, x1 y1, x2 y2, ..., nb_snakes, id x,y:x,y, ...]
    Retourne (fruits, {snake_id: corps}).
    """
    fruits: Set[Coord] = set()
    snakes: Dict[int, List[Coord]] = {}
    if not state_list:
        return fruits, snakes
    try:
        i = 0
        n_fruits = int(state_list[i]); i += 1
        for _ in range(n_fruits):
            x, y = map(int, state_list[i].split()); i += 1
            fruits.add((x, y))
        if i >= len(state_list):
            return fruits, snakes
        n_snakes = int(state_list[i]); i += 1
        for _ in range(n_snakes):
            parts = state_list[i].split(" ", 1); i += 1
            sid = int(parts[0])
            body = []
            for raw_cell in parts[1].split(":"):
                xs, ys = raw_cell.split(",")
                body.append((int(xs), int(ys)))
            snakes[sid] = body
    except (ValueError, IndexError):
        pass
    return fruits, snakes


def _is_initial_state(state_list: List[str]) -> bool:
    """Heuristique : la liste d'etat initiale contient les lignes de la carte (points et '#')."""
    if len(state_list) < 5:
        return False
    for item in state_list[3:]:
        if all(c in ".#" for c in item) and len(item) > 4:
            return True
    return False


def _parse_initial_state(
    state_list: List[str],
) -> Tuple[int, int, int, Set[Coord], int, List[int], List[int]]:
    """Retourne (player_id, width, height, walls, snakes_per_player, my_ids, opp_ids)."""
    i = 0
    player_id = int(state_list[i]); i += 1
    width = int(state_list[i]); i += 1
    height = int(state_list[i]); i += 1
    walls: Set[Coord] = set()
    for y in range(height):
        row = state_list[i]; i += 1
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
    snakes_pp = int(state_list[i]); i += 1
    my_ids = [int(state_list[i + j]) for j in range(snakes_pp)]
    i += snakes_pp
    opp_ids = [int(state_list[i + j]) for j in range(snakes_pp)]
    return player_id, width, height, walls, snakes_pp, my_ids, opp_ids


# ─────────────────────────────────────────────────────────────
# Parsing par segments (gere les deux formats de replay)
# ─────────────────────────────────────────────────────────────

STDERR_MARKER = "Sortie d'erreur :"
STDOUT_MARKER = "Sortie standard :"


@dataclass
class _Segment:
    kind: str           # "stdout" ou "stderr"
    content: List[str]  # lignes (sans le marqueur)


def _collect_segments(lines: List[str]) -> List[_Segment]:
    """Decoupe le fichier en segments bornes par les marqueurs Sortie."""
    segments: List[_Segment] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped in (STDERR_MARKER, STDOUT_MARKER):
            kind = "stderr" if stripped == STDERR_MARKER else "stdout"
            content: List[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() not in (STDERR_MARKER, STDOUT_MARKER):
                content.append(lines[i])
                i += 1
            segments.append(_Segment(kind, content))
        else:
            i += 1
    return segments


def _parse_stdout_segment(
    seg: _Segment,
) -> Tuple[Dict[int, str], int, int]:
    """
    Retourne (moves, turn_num, total_turns).
    turn_num == 0 si ce segment n'a pas de compteur de tour (stdout secondaire).
    """
    non_empty = [l.strip() for l in seg.content if l.strip()]
    if not non_empty:
        return {}, 0, 0
    moves = _parse_moves(non_empty[0])
    turn_num = 0
    total_turns = 0
    # Le compteur de tour est deux entiers consecutifs en fin de segment
    if len(non_empty) >= 3 and non_empty[1].isdigit() and non_empty[2].isdigit():
        turn_num = int(non_empty[1])
        total_turns = int(non_empty[2])
    return moves, turn_num, total_turns


def _parse_stderr_segment(
    seg: _Segment,
) -> Tuple[List[List[str]], Dict[int, SnakeDebug]]:
    """
    Retourne (state_lists, debug_snakes).
    state_lists : liste de listes Python parsees (peut en avoir 0, 1 ou 2).
    """
    state_lists: List[List[str]] = []
    debug_snakes: Dict[int, SnakeDebug] = {}
    for line in seg.content:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                raw = ast.literal_eval(stripped)
                state_lists.append([str(x) for x in raw])
            except Exception:
                pass
        else:
            sd = _parse_snake_debug(stripped)
            if sd:
                debug_snakes[sd.id] = sd
    return state_lists, debug_snakes


def _extract_player_header(
    lines: List[str],
) -> Tuple[str, str, int, int]:
    """
    Cherche le bloc rang+suffixe+nom avant le premier marqueur Sortie.
    Retourne (p1_name, p2_name, p1_rank, p2_rank).
    Les defauts sont ("Bot-1", "Bot-2", 1, 2).
    """
    first_sortie = next(
        (j for j, l in enumerate(lines) if l.strip() in (STDERR_MARKER, STDOUT_MARKER)),
        len(lines),
    )
    header = [l.strip() for l in lines[:first_sortie] if l.strip()]

    # Pattern attendu : rank (chiffre), suffix (er/eme/…), nom, rank, suffix, nom
    try:
        # Trouver l'index du premier rang dans l'en-tete
        start = next(j for j, h in enumerate(header) if h.isdigit())
        p1_rank = int(header[start])
        p1_name = header[start + 2]   # skip suffix
        p2_rank = int(header[start + 3])
        p2_name = header[start + 5]   # skip suffix
        return p1_name, p2_name, p1_rank, p2_rank
    except (StopIteration, IndexError, ValueError):
        return "Bot-1", "Bot-2", 1, 2


def parse_game_log(path: Path) -> GameLog:
    lines = path.read_text(encoding="utf-8").splitlines()

    p1_name, p2_name, p1_rank, p2_rank = _extract_player_header(lines)

    game = GameLog(
        p1_name=p1_name,
        p2_name=p2_name,
        p1_rank=p1_rank,
        p2_rank=p2_rank,
        width=0,
        height=0,
        walls=set(),
        p1_snake_ids=[],
        p2_snake_ids=[],
    )

    segments = _collect_segments(lines)

    # -- Etat persistent entre segments
    current_fruits: Set[Coord] = set()
    current_snakes: Dict[int, List[Coord]] = {}
    turns: List[TurnData] = []

    # -- Accumulateur par tour
    # Cle: turn_num ; valeur: dict partiel
    pending: Dict[str, object] = {}

    def flush_turn() -> None:
        """Finalise et enregistre le tour en attente si complet."""
        turn_num = pending.get("turn_num", 0)
        if not turn_num:
            return
        # Assigner les mouvements au bon joueur en fonction des IDs de snakes
        moves_a: Dict[int, str] = pending.get("moves_a", {})  # type: ignore
        moves_b: Dict[int, str] = pending.get("moves_b", {})  # type: ignore
        my_ids_set = set(game.p1_snake_ids)

        # Le bloc qui contient des IDs de p1_snake_ids est attribue a p1
        if any(k in my_ids_set for k in moves_a):
            p1_moves, p2_moves = moves_a, moves_b
        elif any(k in my_ids_set for k in moves_b):
            p1_moves, p2_moves = moves_b, moves_a
        else:
            p1_moves, p2_moves = moves_a, moves_b

        turns.append(TurnData(
            turn=int(turn_num),
            total_turns=int(pending.get("total_turns", 0)),  # type: ignore
            fruits=set(current_fruits),
            snakes_actual=dict(current_snakes),
            debug=dict(pending.get("debug", {})),  # type: ignore
            p1_moves=p1_moves,
            p2_moves=p2_moves,
        ))

    for seg in segments:
        if seg.kind == "stdout":
            moves, turn_num, total_turns = _parse_stdout_segment(seg)
            if turn_num:
                # Ce bloc est le "stdout primaire" : il demarre ou termine un tour
                flush_turn()
                pending.clear()
                pending["turn_num"] = turn_num
                pending["total_turns"] = total_turns
                pending["moves_a"] = moves
            else:
                # Stdout secondaire : mouvements de l'autre joueur
                pending["moves_b"] = moves

        else:  # stderr
            state_lists, debug_snakes = _parse_stderr_segment(seg)

            # Traiter les listes d'etat dans l'ordre
            for sl in state_lists:
                if _is_initial_state(sl):
                    _, w, h, walls, _, my_ids, opp_ids = _parse_initial_state(sl)
                    game.width = w
                    game.height = h
                    game.walls = walls
                    game.p1_snake_ids = my_ids
                    game.p2_snake_ids = opp_ids
                else:
                    current_fruits, current_snakes = _parse_turn_state(sl)

            if debug_snakes:
                pending["debug"] = debug_snakes

    # Flush le dernier tour
    flush_turn()

    game.turns = turns
    return game


# ─────────────────────────────────────────────────────────────
# Métriques & événements
# ─────────────────────────────────────────────────────────────

@dataclass
class Event:
    turn: int
    kind: str          # "eat", "death", "collision", "fall", "doomed"
    player: int        # 1 ou 2
    snake_id: int
    detail: str


def build_events(game: GameLog) -> List[Event]:
    events: List[Event] = []
    prev_fruit_count = None
    prev_snake_ids: Set[int] = set()

    for td in game.turns:
        fruit_count = len(td.fruits)

        # ── Morts de snakes (disparition entre deux tours) ──
        alive_ids = set(td.snakes_actual.keys())
        if prev_snake_ids:
            died = prev_snake_ids - alive_ids
            for sid in died:
                player = 1 if sid in game.p1_snake_ids else 2
                events.append(Event(td.turn, "death", player, sid,
                                    f"Snake {sid} (J{player}) disparu du plateau"))
        prev_snake_ids = alive_ids

        # ── Fruits mangés (via debug P1) ──────────────────────
        for sd in td.debug.values():
            if sd.ate:
                events.append(Event(td.turn, "eat", 1, sd.id,
                                    f"Snake {sd.id} mange un fruit (score eval={sd.score:+d})"))

        # ── Fruits mangés P2 (inférence via variation du nombre de fruits) ──
        if prev_fruit_count is not None:
            delta = prev_fruit_count - fruit_count
            p1_ate = sum(1 for sd in td.debug.values() if sd.ate)
            p2_ate_estimated = max(0, delta - p1_ate)
            if p2_ate_estimated > 0:
                events.append(Event(td.turn, "eat", 2, -1,
                                    f"J2 mange ~{p2_ate_estimated} fruit(s) "
                                    f"(fruits: {prev_fruit_count}->{fruit_count})"))
        prev_fruit_count = fruit_count

        # ── Collisions détectées par le bot ──────────────────
        for sd in td.debug.values():
            if sd.collided:
                events.append(Event(td.turn, "collision", 1, sd.id,
                                    f"Snake {sd.id} collision detectee "
                                    f"(move={sd.chosen_dir}, score={sd.score:+d})"))
            if sd.is_doomed():
                events.append(Event(td.turn, "doomed", 1, sd.id,
                                    f"Snake {sd.id} accule (score=-inf, move={sd.chosen_dir})"))

        # ── Chutes significatives ─────────────────────────────
        for sd in td.debug.values():
            if sd.fall_distance >= 3:
                events.append(Event(td.turn, "fall", 1, sd.id,
                                    f"Snake {sd.id} chute de {sd.fall_distance} case(s) "
                                    f"(move={sd.chosen_dir})"))

    return events


# ─────────────────────────────────────────────────────────────
# Analyse de stratégie
# ─────────────────────────────────────────────────────────────

def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearest_fruit_dist(head: Coord, fruits: Set[Coord]) -> Optional[int]:
    if not fruits:
        return None
    return min(manhattan(head, f) for f in fruits)


def _add_coord(c: Coord, d: Tuple[int, int]) -> Coord:
    return (c[0] + d[0], c[1] + d[1])


def _classify_strategy_p1(game: GameLog) -> Dict[str, object]:
    """Analyse détaillée du bot P1 grâce aux données de debug."""
    eat_turns = []
    collision_turns = []
    doomed_turns = []
    fall_events = []
    score_series: List[float] = []
    direction_counts: Counter = Counter()
    fruit_chase_rate = []   # proportion de tours où le bot va vers le fruit le plus proche

    for td in game.turns:
        if not td.debug:
            continue
        # Score moyen par tour (snakes P1 uniquement)
        scores = [sd.score for sd in td.debug.values() if not sd.is_doomed()]
        if scores:
            score_series.append(sum(scores) / len(scores))

        for sd in td.debug.values():
            direction_counts[sd.chosen_dir] += 1
            if sd.ate:
                eat_turns.append(td.turn)
            if sd.collided or sd.is_doomed():
                if sd.is_doomed():
                    doomed_turns.append(td.turn)
                if sd.collided:
                    collision_turns.append(td.turn)
            if sd.fall_distance > 0:
                fall_events.append((td.turn, sd.id, sd.fall_distance))

            # Est-ce que le mouvement choisi rapproche du fruit le plus proche ?
            if td.fruits and sd.projected_body:
                head = sd.projected_body[0]
                dist_after = nearest_fruit_dist(head, td.fruits)
                # Corps actuel (avant move) depuis snakes_actual
                actual_body = td.snakes_actual.get(sd.id)
                if actual_body and dist_after is not None:
                    dist_before = nearest_fruit_dist(actual_body[0], td.fruits)
                    if dist_before is not None:
                        fruit_chase_rate.append(1 if dist_after < dist_before else 0)

    avg_fruit_chase = sum(fruit_chase_rate) / len(fruit_chase_rate) if fruit_chase_rate else 0
    avg_score = sum(score_series) / len(score_series) if score_series else 0
    score_volatility = (
        (max(score_series) - min(score_series)) if score_series else 0
    )

    return {
        "eat_turns": eat_turns,
        "collision_turns": collision_turns,
        "doomed_turns": doomed_turns,
        "fall_events": fall_events,
        "direction_counts": direction_counts,
        "avg_score": avg_score,
        "score_volatility": score_volatility,
        "avg_fruit_chase_rate": avg_fruit_chase,
        "score_series": score_series,
    }


def _classify_strategy_p2(game: GameLog) -> Dict[str, object]:
    """
    Inférence de la stratégie de P2 à partir de ses mouvements et de l'état du plateau.
    Pas de données de debug pour P2.
    """
    direction_counts: Counter = Counter()
    fruit_chase_rate = []
    aggression_rate = []   # proportion de tours où P2 se rapproche d'un snake P1
    direction_changes = 0
    total_moves = 0
    prev_dirs: Dict[int, str] = {}

    for td in game.turns:
        p2_alive = {sid: td.snakes_actual[sid]
                    for sid in game.p2_snake_ids
                    if sid in td.snakes_actual}

        for sid, move in td.p2_moves.items():
            direction_counts[move] += 1
            total_moves += 1

            # Changement de direction par rapport au tour précédent
            if sid in prev_dirs and prev_dirs[sid] != move:
                direction_changes += 1
            prev_dirs[sid] = move

            # Est-ce que le mouvement rapproche du fruit le plus proche ?
            if sid in p2_alive and td.fruits:
                head = p2_alive[sid][0]
                dx, dy = DIRECTIONS[move]
                new_head = (head[0] + dx, head[1] + dy)
                dist_before = nearest_fruit_dist(head, td.fruits)
                dist_after = nearest_fruit_dist(new_head, td.fruits)
                if dist_before is not None and dist_after is not None:
                    fruit_chase_rate.append(1 if dist_after < dist_before else 0)

            # Est-ce que le mouvement rapproche d'un snake P1 (agressivité) ?
            p1_heads = [td.snakes_actual[sid2][0]
                        for sid2 in game.p1_snake_ids
                        if sid2 in td.snakes_actual]
            if sid in p2_alive and p1_heads:
                head = p2_alive[sid][0]
                dx, dy = DIRECTIONS[move]
                new_head = (head[0] + dx, head[1] + dy)
                dist_before = min(manhattan(head, h) for h in p1_heads)
                dist_after = min(manhattan(new_head, h) for h in p1_heads)
                aggression_rate.append(1 if dist_after < dist_before else 0)

    avg_fruit_chase = sum(fruit_chase_rate) / len(fruit_chase_rate) if fruit_chase_rate else 0
    avg_aggression = sum(aggression_rate) / len(aggression_rate) if aggression_rate else 0
    dir_change_rate = direction_changes / total_moves if total_moves else 0

    return {
        "direction_counts": direction_counts,
        "avg_fruit_chase_rate": avg_fruit_chase,
        "avg_aggression_rate": avg_aggression,
        "direction_change_rate": dir_change_rate,
        "total_moves": total_moves,
    }


def _infer_strategy_label(stats: Dict[str, object], is_p1: bool) -> str:
    """Retourne une étiquette de stratégie à partir des statistiques."""
    fruit_chase = float(stats.get("avg_fruit_chase_rate", 0))  # type: ignore
    labels = []

    if is_p1:
        avg_score = float(stats.get("avg_score", 0))  # type: ignore
        doomed = len(stats.get("doomed_turns", []))   # type: ignore
        fall_events = stats.get("fall_events", [])    # type: ignore
        falls = len([f for f in fall_events if f[2] >= 2])  # type: ignore

        if fruit_chase > 0.65:
            labels.append("chasseur de fruits (greedy)")
        elif fruit_chase > 0.45:
            labels.append("semi-greedy (fruits + tactique)")
        else:
            labels.append("tactique / controle de territoire")

        if doomed > 0:
            labels.append(f"accule {doomed}x (score=-inf)")
        if falls > 1:
            labels.append(f"chutes frequentes ({falls}x)")
        if avg_score > 300:
            labels.append("evaluations confiantes (score moyen eleve)")
        elif avg_score < 50:
            labels.append("evaluations prudentes (score moyen faible)")
    else:
        aggression = float(stats.get("avg_aggression_rate", 0))  # type: ignore
        dir_change = float(stats.get("direction_change_rate", 0))  # type: ignore

        if fruit_chase > 0.65:
            labels.append("chasseur de fruits (greedy)")
        elif fruit_chase > 0.45:
            labels.append("semi-greedy")
        else:
            labels.append("non-greedy (controle / attente ?)")

        if aggression > 0.5:
            labels.append("agressif (rapprochement vers P1)")
        elif aggression < 0.3:
            labels.append("defensif / evite P1")

        if dir_change > 0.5:
            labels.append("reactif (beaucoup de changements de direction)")
        elif dir_change < 0.2:
            labels.append("persistant (peu de changements de direction)")

    return " | ".join(labels) if labels else "indetermine"


# ─────────────────────────────────────────────────────────────
# Pistes d'optimisation
# ─────────────────────────────────────────────────────────────

def build_optimization_hints(
    game: GameLog,
    events: List[Event],
    p1_stats: Dict[str, object],
) -> List[str]:
    hints = []

    doomed_turns = p1_stats.get("doomed_turns", [])   # type: ignore
    collision_turns = p1_stats.get("collision_turns", [])  # type: ignore
    fall_events = p1_stats.get("fall_events", [])  # type: ignore
    eat_turns = p1_stats.get("eat_turns", [])  # type: ignore
    avg_score = float(p1_stats.get("avg_score", 0))  # type: ignore
    fruit_chase = float(p1_stats.get("avg_fruit_chase_rate", 0))  # type: ignore
    score_series = p1_stats.get("score_series", [])  # type: ignore

    total_turns = game.turns[-1].total_turns if game.turns else 0

    # 1. Situations de score -inf (bot accule)
    if doomed_turns:
        hints.append(
            f"[PIEGE] Le bot a ete accule (score=-inf) aux tours {doomed_turns}. "
            "Ameliorer l'anticipation des impasses : lookahead plus profond ou "
            "penalite plus forte pour les espaces fermes (reachable_cells)."
        )

    # 2. Collisions detectees mais inevitables
    if collision_turns and not doomed_turns:
        hints.append(
            f"[COLLISION] Des collisions ont eu lieu aux tours {collision_turns} "
            "sans que le bot soit accule. Verifier la gestion des collisions simultanees "
            "(head-on avec l'adversaire)."
        )
    elif collision_turns and doomed_turns:
        overlap = set(collision_turns) & set(doomed_turns)
        if overlap:
            hints.append(
                f"[COLLISION+PIEGE] Aux tours {sorted(overlap)}, le bot etait accule ET "
                "a subi une collision. Priorite : eviter les dead-ends plus tot."
            )

    # 3. Taux de chasse aux fruits
    if fruit_chase < 0.45:
        hints.append(
            f"[FRUITS] Taux de rapprochement vers les fruits : {fruit_chase:.0%}. "
            "Le bot s'eloigne souvent des fruits. Verifier si score_eat_bonus ou "
            "score_power_distance_penalty sont correctement calibres."
        )
    elif fruit_chase > 0.80:
        hints.append(
            f"[FRUITS] Taux de chasse tres eleve ({fruit_chase:.0%}). "
            "Risque de foncer tete baissee vers les fruits sans anticiper les pieges. "
            "Equilibrer avec la penalite score_head_danger et score_no_exit."
        )

    # 4. Chutes frequentes
    big_falls = [(t, sid, fd) for t, sid, fd in fall_events if fd >= 3]  # type: ignore
    if big_falls:
        hints.append(
            f"[GRAVITE] {len(big_falls)} chute(s) de 3+ cases detectee(s) "
            f"(tours: {[t for t, _, _ in big_falls]}). "
            "Augmenter score_fall_distance_penalty ou ajouter une penalite "
            "pour les positions en hauteur sans support."
        )

    # 5. Volatilite des scores
    if isinstance(score_series, list) and len(score_series) > 3:
        neg_ratio = sum(1 for s in score_series if s < 0) / len(score_series)
        if neg_ratio > 0.4:
            hints.append(
                f"[SCORE] {neg_ratio:.0%} des tours ont un score moyen negatif. "
                "Le bot manque souvent de bonnes options. Envisager une meilleure gestion "
                "de l'espace libre (voronoi / flood-fill) comme score principal."
            )

    # 6. Fruits manges vs total
    total_initial_fruits = max(
        (len(t.fruits) for t in game.turns), default=0
    )
    p1_fruits_eaten = len(eat_turns)
    p2_events = [e for e in events if e.kind == "eat" and e.player == 2]
    p2_fruits_estimated = sum(
        int(re.search(r"~(\d+)", e.detail).group(1))
        for e in p2_events
        if re.search(r"~(\d+)", e.detail)
    )
    if total_initial_fruits > 0 and p2_fruits_estimated > p1_fruits_eaten:
        hints.append(
            f"[COMPETITION] J2 a mange ~{p2_fruits_estimated} fruits contre "
            f"{p1_fruits_eaten} pour J1. Augmenter contest_closer_bonus ou "
            "contest_unreachable_opp_bonus pour mieux intercepter les fruits convoites."
        )

    # 7. Morts precoces
    death_events = [e for e in events if e.kind == "death" and e.player == 1]
    if death_events:
        first_death = death_events[0]
        if first_death.turn < total_turns * 0.5:
            hints.append(
                f"[MORT PRECOCE] Premier snake P1 mort au tour {first_death.turn}/{total_turns} "
                f"(< 50% de la partie). Priorite : survivre plus longtemps avec une meilleure "
                "evaluation des risques (score_head_danger, reachable_cells_few_penalty)."
            )

    # 8. Suggestion lookahead
    if len(doomed_turns) == 0 and avg_score > 0:
        hints.append(
            "[LOOKAHEAD] Aucune situation de score=-inf detectee. "
            "Profil de risque faible : tester un lookahead_future_score_divisor plus bas "
            "pour exploiter davantage le futur (actuellement dans META_PARAMS)."
        )

    if not hints:
        hints.append("Aucune anomalie majeure detectee dans ce log.")

    return hints


# ─────────────────────────────────────────────────────────────
# Rapport
# ─────────────────────────────────────────────────────────────

def _bar(value: float, max_val: float, width: int = 20) -> str:
    if max_val == 0:
        return "-" * width
    filled = int(round(value / max_val * width))
    return "#" * filled + "." * (width - filled)


def print_report(game: GameLog, verbose: bool = False) -> None:
    sep = "-" * 72

    print()
    print("=" * 72)
    print("  ANALYSE DE PARTIE - Snakebird CG 2026 Winter")
    print("=" * 72)

    # ── Infos générales ──────────────────────────────────────
    winner = game.p1_name if game.p1_rank < game.p2_rank else game.p2_name
    loser  = game.p2_name if game.p1_rank < game.p2_rank else game.p1_name
    total_turns = game.turns[-1].total_turns if game.turns else "?"

    print(f"\n  Plateau      : {game.width}x{game.height}, {len(game.walls)} murs")
    print(f"  Duree        : {len(game.turns)} tours joues / {total_turns} max")
    print(f"  Joueur 1     : {game.p1_name}  (snakes {game.p1_snake_ids})  rang {game.p1_rank}")
    print(f"  Joueur 2     : {game.p2_name}  (snakes {game.p2_snake_ids})  rang {game.p2_rank}")
    print(f"  Resultat     : {winner} GAGNE / {loser} perd")

    # -- Evenements
    events = build_events(game)
    print(f"\n{sep}")
    print("  CHRONOLOGIE DES EVENEMENTS")
    print(sep)
    for e in events:
        icon = {"eat": "[EAT]", "death": "[MORT]", "collision": "[COLL]", "fall": "[CHUTE]",
                "doomed": "[DOOM]"}.get(e.kind, "[--]")
        print(f"  Tour {e.turn:3d}  {icon}  {e.detail}")

    # ── Statistiques P1 ──────────────────────────────────────
    p1_stats = _classify_strategy_p1(game)
    p2_stats = _classify_strategy_p2(game)

    print(f"\n{sep}")
    print(f"  ANALYSE DU BOT P1 - {game.p1_name} (debug complet disponible)")
    print(sep)

    eat_count = len(p1_stats["eat_turns"])  # type: ignore
    doomed_count = len(p1_stats["doomed_turns"])  # type: ignore
    fall_count = len(p1_stats["fall_events"])  # type: ignore
    avg_score = p1_stats["avg_score"]
    score_series = p1_stats["score_series"]  # type: ignore
    dir_counts = p1_stats["direction_counts"]  # type: ignore

    print(f"  Fruits manges       : {eat_count}")
    print(f"  Tours accule (-inf) : {doomed_count}  (tours: {p1_stats['doomed_turns']})")
    print(f"  Tours en collision  : {len(p1_stats['collision_turns'])}  (tours: {p1_stats['collision_turns']})")
    print(f"  Chutes (fall>0)     : {fall_count}")
    print(f"  Score moyen eval    : {avg_score:+.1f}")
    print(f"  Chasse aux fruits   : {p1_stats['avg_fruit_chase_rate']:.0%}")

    print("\n  Distribution des directions P1:")
    sum(dir_counts.values())
    for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
        cnt = dir_counts.get(d, 0)
        print(f"    {d:6s}  {cnt:4d}  {_bar(cnt, max(dir_counts.values(), default=1))}")

    print(f"\n  Strategie inferee P1 : {_infer_strategy_label(p1_stats, True)}")

    if verbose and isinstance(score_series, list) and score_series:
        print("\n  Evolution du score moyen d'evaluation par tour :")
        max_s = max(abs(s) for s in score_series) or 1
        for idx, s in enumerate(score_series, 1):
            bar = _bar(abs(s), max_s, 30)
            sign = "+" if s >= 0 else "-"
            print(f"    T{idx:3d}  {sign}{abs(s):7.0f}  {bar}")

    # -- Statistiques P2
    print(f"\n{sep}")
    print(f"  ANALYSE DU BOT P2 - {game.p2_name} (inference mouvements uniquement)")
    print(sep)

    dir_counts_p2 = p2_stats["direction_counts"]  # type: ignore
    total_moves_p2 = p2_stats["total_moves"]

    p2_events_eat = [e for e in events if e.kind == "eat" and e.player == 2]
    p2_fruits = sum(
        int(m.group(1))
        for e in p2_events_eat
        if (m := re.search(r"~(\d+)", e.detail))
    )

    print(f"  Fruits manges (est.): {p2_fruits}")
    print(f"  Total mouvements    : {total_moves_p2}")
    print(f"  Chasse aux fruits   : {p2_stats['avg_fruit_chase_rate']:.0%}")
    print(f"  Agressivite vs P1   : {p2_stats['avg_aggression_rate']:.0%}")
    print(f"  Taux chgt direction : {p2_stats['direction_change_rate']:.0%}")

    print("\n  Distribution des directions P2:")
    if dir_counts_p2:
        for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
            cnt = dir_counts_p2.get(d, 0)
            print(f"    {d:6s}  {cnt:4d}  {_bar(cnt, max(dir_counts_p2.values(), default=1))}")

    print(f"\n  Strategie inferee P2 : {_infer_strategy_label(p2_stats, False)}")

    # -- Pistes d'optimisation
    hints = build_optimization_hints(game, events, p1_stats)
    print(f"\n{sep}")
    print("  PISTES D'OPTIMISATION")
    print(sep)
    for idx, hint in enumerate(hints, 1):
        # Wrap à 68 chars
        words = hint.split()
        _lines_out, cur = [], ""
        for w in words:
            if len(cur) + len(w) + 1 > 68:
                print(f"  {idx if not cur else ' '}. {cur}")
                cur = w
                idx = " "  # type: ignore
            else:
                cur = (cur + " " + w).strip()
        if cur:
            print(f"  {idx}. {cur}")
        print()

    print("=" * 72)
    print()


# ─────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    # Forcer UTF-8 sur stdout/stderr Windows avant tout affichage
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

    parser = argparse.ArgumentParser(
        description="Analyse un log de partie Snakebird CG 2026."
    )
    parser.add_argument("log", type=Path, help="Chemin vers le fichier de log (game.txt)")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Affiche l'evolution du score d'evaluation tour par tour",
    )
    args = parser.parse_args()

    if not args.log.exists():
        print(f"Erreur : fichier introuvable : {args.log}", file=sys.stderr)
        sys.exit(1)

    game = parse_game_log(args.log)

    if not game.turns:
        print("Erreur : aucun tour analyse. Verifiez le format du fichier.", file=sys.stderr)
        sys.exit(1)

    print_report(game, verbose=args.verbose)


if __name__ == "__main__":
    main()
