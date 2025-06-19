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
		w = data.get('weight', 0)
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

def sample_continuation(game:chess.pgn.ChildNode, mu: float, sigma: float, epsilon: float, start_depth: int, number_of_games:int) -> List[str]:
	while game.parent:
		game = game.parent
	board = chess.Board()
	moves = list(game.mainline_moves())
	san_moves = []
	# push prefix moves
	for i, m in enumerate(moves):
		if i < start_depth:
			board.push(m)
		else:
			# collect SANs of remaining moves
			san_moves.append(board.san(m))
			board.push(m)

	items = []  # (move, weight)
	depth = start_depth
	for i, m in enumerate(san_moves):
		w = epsilon/number_of_games + gaussian_density(i + depth, mu, sigma)
		items.append((m, w))
	total_w = sum(w for _, w in items)
	r = random.uniform(0, total_w)
	upto = 0.0
	for i, it in enumerate(items):
		m, w = it
		upto += w
		if r <= upto:
			break

	return san_moves[:i]

def sample_variation(G: nx.DiGraph,
					mu: float,
					sigma: float,
					epsilon: float,
					number_of_games: int) -> Tuple[List[str], Any]:
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
		return "root " + " ".join(variation)
	game = sample_game_from_leaf(G, chosen_node)
	cont = sample_continuation(game, mu, sigma, epsilon, data.get('depth', 0), number_of_games)
	full_variation = variation + cont
	return "root " + " ".join(full_variation)
