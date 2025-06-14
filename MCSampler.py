import math
import random
import networkx as nx
import chess.pgn
from typing import Any, List, Tuple

def gaussian_density(depth: float, mu: float, sigma: float) -> float:
    return (1.0 / (2 * math.pi * sigma * sigma)) * math.exp(-((depth - mu) ** 2) / (2 * sigma * sigma))

def gaussian_tail(depth: float, mu: float, sigma: float) -> float:
    return 0.5 * math.erfc((depth - mu) / (math.sqrt(2) * sigma))

def compute_node_weights(G: nx.DiGraph,
                         mu: float,
                         sigma: float,
                         epsilon: float) -> List[Tuple[Any, float]]:
    weights = []
    for n, data in G.nodes(data=True):
        d = data.get('depth', 0)
        c = data.get('count', 0)
        is_ext = data.get('is_exterior', False)
        if is_ext:
            w = epsilon + c * gaussian_tail(d, mu, sigma)
        else:
            w = epsilon + c * gaussian_density(d, mu, sigma)
        weights.append((n, w))
    return weights

def sample_pruned_node(G: nx.DiGraph,
                       mu: float,
                       sigma: float,
                       epsilon: float) -> Any:
    node_weights = compute_node_weights(G, mu, sigma, epsilon)
    total = sum(w for _, w in node_weights)
    r = random.uniform(0, total)
    upto = 0.0
    for n, w in node_weights:
        upto += w
        if r <= upto:
            return n
    return node_weights[-1][0]

def sample_game_from_leaf(G: nx.DiGraph, leaf: Any) -> Any:
    games = G.nodes[leaf].get('games', [])
    if not games:
        raise ValueError(f"Leaf node {leaf} has no games to sample.")
    return random.choice(games)

def sample_continuation(games: List[Any], mu: float, sigma: float, epsilon: float, start_depth: int) -> List[str]:
    variation = []
    current_games = games
    depth = start_depth
    while len(current_games) > 1:
        move_groups = {}
        for g in current_games:
            moves = list(g.mainline_moves())
            if len(moves) <= depth:
                continue
            next_move = moves[depth]
            move_groups.setdefault(next_move, []).append(g)
        if not move_groups:
            break
        items = []
        for move, grp in move_groups.items():
            c = len(grp)
            w = epsilon + c * gaussian_density(depth + 1, mu, sigma)
            items.append((move, grp, w))
        total_w = sum(w for _, _, w in items)
        r = random.uniform(0, total_w)
        upto = 0.0
        for move, grp, w in items:
            upto += w
            if r <= upto:
                board = current_games[0].board()
                for m in current_games[0].mainline_moves()[:depth]:
                    board.push(m)
                variation.append(board.san(move))
                current_games = grp
                depth += 1
                break
    return variation

def sample_variation(G: nx.DiGraph,
                     mu: float,
                     sigma: float,
                     epsilon: float) -> Tuple[List[str], Any]:
    chosen_node = sample_pruned_node(G, mu, sigma, epsilon)
    data = G.nodes[chosen_node]
    raw_moves = chosen_node.split(" ") if isinstance(chosen_node, str) else []
    variation = []
    if raw_moves:
        game = chess.pgn.Game()
        board = game.board()
        for mv in raw_moves:
            if mv == "root": continue
            
            move = board.parse_san(mv)
            board.push(move)
            variation.append(mv)
    if not data.get('is_exterior', False):
        return " ".join(variation)
    game = sample_game_from_leaf(G, chosen_node)
    cont = sample_continuation([game], mu, sigma, epsilon, data.get('depth', 0))
    full_variation = variation + cont
    return " ".join(full_variation)
