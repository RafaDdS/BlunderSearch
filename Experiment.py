import math
from chess import Board
from getGames import fromAPI, fromLocalFile
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from scipy.special import kl_div
import pickle
import numpy as np
import MCSampler
from randomWalk import draw_chess_trees_grid

mu = 10
sigma = 4

def build_tree_from_games(games, min_count, return_full = False):
	"""
	Construct a deterministic game‐tree from a list of PGN Game objects.

	Parameters
	----------
	games : list of chess.pgn.Game
		Each Game represents a complete, linear (no‐variation) game.
	min_count : int
		Only include a child node if at least `min_count` games follow that move.

	Returns
	-------
	G : networkx.DiGraph
		Directed graph where each node key is a full‐path string of SAN moves
		(e.g. "", "e4", "e4 e5", ...). The root is "".
	labels : dict
		Mapping from each node key → string to display (the last move in SAN, or "root").
	"""
	G = nx.DiGraph()
	G.add_node("root", weight=0.0, games=list(games))      # root (empty‐path)
	labels = {"root": "root"}           # root’s displayed label

	queue = deque()
	# At depth 0, the “frontier” is just the list of full‐PGN Game objects
	initial_frontier = list(games)
	root_board = Board()
	queue.append(("root", root_board, initial_frontier))

	while queue:
		path, board, frontier = queue.popleft()

		# Group frontier‐games by their next UCI move
		move_groups = {}  # { chess.Move : [list of PGN nodes at that next ply] }
		for game_node in frontier:
			# Each game_node is a chess.pgn.Game or a MoveNode
			next_node = game_node.next()    # next_node is None if game is over
			if not next_node:
				continue
			mv = next_node.move            # a chess.Move in UCI
			move_groups.setdefault(mv, []).append(next_node)

		# For each distinct next move, only branch if at least `min_count` games use it
		for mv, next_nodes in move_groups.items():
			# Convert UCI to SAN given the current board
			alg = board.san(mv)

			# Build the full‐path string
			new_path = alg if path == "" else f"{path} {alg}"

			# Add node and edge to G
			is_exterior = (len(next_nodes) <= min_count)
			depth = frontier[0].ply()+1
			count = len(next_nodes)
			G.add_node(new_path, weight=0.0, games=next_nodes, count=count, depth=depth, degree=len(next_nodes)+1, is_exterior=is_exterior)      # weight is a placeholder here
			G.add_edge(path, new_path)

			# Record the displayed label (just the last move)
			labels[new_path] = alg

			# Prepare the board copy for this branch
			child_board = board.copy()
			child_board.push(mv)

			if not return_full and is_exterior:
				continue

			# The next frontier is exactly next_nodes (they already point to the next ply)
			queue.append((new_path, child_board, next_nodes))

	return G, labels

def calculate_theoretical_stationary_distribution(G):
	weights = nx.get_node_attributes(G, "weight")
	depths = nx.get_node_attributes(G, "depth")
	counts = nx.get_node_attributes(G, "count")
	is_exteriors = nx.get_node_attributes(G, "is_exterior")
	
	tmp = []
	Z = 0
	for key in weights:
		depth = depths.get(key, False)
		count = counts.get(key, False)
		is_exterior =is_exteriors.get(key, False)

		if not depth:
			continue
		
		if is_exterior:
			fn = 0.5 * math.erfc((depth - mu) / (math.sqrt(2) * sigma))
		else:
			fn = (count/(2*math.pi*sigma*sigma)) * math.exp(-pow(depth-mu, 2)/(2*sigma*sigma))
		
		Z += fn
		tmp.append( (fn, key) )

	out = {}
	for fn, key in tmp:
		out[key] = fn/Z

	nx.set_node_attributes(G, out, "weight")

	return out

def saveGraph(graph, file="complete_graph"):
	with open(f"{file}.pkl", "wb") as f:
		pickle.dump(graph, f)

def loadGraph(file="complete_graph"):
	with open(f"{file}.pkl", "rb") as f:
		graph = pickle.load(f)
	return graph

def compare_distribution(true_pi, pi, kl=False):
	nodes = sorted(set(true_pi.keys()) | set(pi.keys()), key=lambda k: len(k.split()))

	# 2. Build aligned probability vectors
	P = np.array([true_pi.get(n, 0.0) for n in nodes], dtype=float)
	Q = np.array([pi.get(n, 0.0) for n in nodes], dtype=float)

	if kl:
		return kl_div(P, Q)
	else:
		return np.abs(P - Q) * 0.5

def plot_kl_div(kl_list: list, title: str = None, kl =False) -> None:
	"""
	Plot the element-wise max, min, and average across multiple KL-divergence arrays.

	Parameters
	----------
	kl_list : list of np.ndarray
		A list of 1D arrays (all the same length) containing element-wise KL divergences.
	title : str, optional
		An optional title for the plot.
	"""
	# Stack arrays (shape: num_runs x num_nodes)
	stacked = np.vstack(kl_list)

	# Compute statistics across runs
	max_vals = np.max(stacked, axis=0)
	min_vals = np.min(stacked, axis=0)
	avg_vals = np.mean(stacked, axis=0)

	# X-axis: node indices (no labels for individual nodes)
	x = np.arange(stacked.shape[1])

	# Plot
	plt.figure()
	plt.plot(x, max_vals, label='max')
	plt.plot(x, avg_vals, label='average')
	plt.plot(x, min_vals, label='min')
	plt.xlabel('Node index')
	plt.ylabel('KL divergence' if kl else 'Variation')
	if title:
		plt.title(title)
	plt.legend()
	plt.tight_layout()
	plt.show()

def plot_distribution(true_pi: list, title: str = None) -> None:
	nodes = sorted(set(true_pi.keys()), key=lambda k: len(k.split()))

	# 2. Build aligned probability vectors
	P = np.array([true_pi.get(n, 0.0) for n in nodes], dtype=float)
	# Plot
	plt.figure()
	plt.plot(P)
	plt.xlabel('Node index')
	plt.ylabel('Probability')
	if title:
		plt.title(title)
	
	plt.tight_layout()
	plt.show()


#number_of_games = 10
#Z = 100000
H = 1
TVs = []
for number_of_games in [10, 100, 1000]:
	for Z in [10, 100, 1000, 10000, 100000]:
		print()
		true_graph = loadGraph(file=f"test{number_of_games}graph")
		true_pi = calculate_theoretical_stationary_distribution(true_graph)
		#plot_distribution(true_pi, "IPD 10 Games")


		games = fromLocalFile(max=number_of_games)
		#print(true_graph)
		print("Games loaded")

		graphs = []
		labels = []
		pis = []

		graph, label = build_tree_from_games(games, 0)
		calculate_theoretical_stationary_distribution(graph)
		#saveGraph(graph, file=f"test{number_of_games}graph")

		graphs.append(graph)
		labels.append(label)

		for i in range(10):
			samples = dict()
			for s in range(Z):
				va = MCSampler.sample_variation(graph, mu, sigma, 0, number_of_games)
				samples[va] = samples.get(va, 0) + 1
			
			pi_n = {key: val/Z for key ,val in samples.items()}

			pis.append(pi_n)
			print(i)

		print(graph)


		pis2 = [compare_distribution(true_pi, pi) for pi in pis]

		TVs.append([sum(pi) for pi in pis2])

print(TVs)
#plot_kl_div(pis2, f"{number_of_games} Games Horizon-{H} {Z} Samples")

#draw_chess_trees_grid(graphs, labels, draw_labels=False, ncols=1)