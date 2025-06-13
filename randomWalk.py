import random
import math
from chess import Board
from stockfish import Stockfish
from getGames import fromAPI, fromLocalFile
import networkx as nx
from networkx import DiGraph
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from collections import deque
import scipy as sp
from scipy.special import kl_div
import scipy.sparse

def uci_to_algebraic(uci_move, board):
	move = board.parse_uci(str(uci_move))
	return board.san(move)

def randomPathForward(games):
	board = Board()
	current = games
	path = ""
	moves = []
	while current:
		l = len(current)
		move = random.choice(current).next()
		move = move.move if move else move
		moves.append(move)

		if not move:
			break

		path += uci_to_algebraic(move, board)
		print(l)
		print()
		print(path)
		
		current = [g.next() for g in current if g.next() and g.next().move == move]
		board.push(move)

	print() 
	print(board.fen())

def getGamesWithMoves(games, moves):
	current = games
	for move in moves:
		current = [g.next() for g in current if g.next() and g.next().move == move]
	
	return current

def randomWalk(games, n=pow(10, 5), horizonSize=1, shouldMoveStart=False):
	# Setup variables
	board = Board()
	current = games
	moves = []
	history = []
	path = ""
	pi = {}
	for _ in range(n):
		nGames = len(current)

		move = random.choice(current).next()
		move = move.move if move else move
		moves.append(move)

		moveStart = (not move or nGames==1) and shouldMoveStart
		moveUp = (moves and random.uniform(0, 1) < 1/(nGames+1)) or (nGames<=horizonSize) or (not move and not shouldMoveStart)

		if moveStart:
			board = Board()
			current = games
			moves = []
			history = []
			path = ""
			continue

		if moveUp:
			if history:
				moves.pop()
				board.pop()
				current = history.pop()
				tmp = path.split()
				tmp.pop()
				path = " ".join(tmp)
			continue

		if not move:
			raise "No move down, should have been handled before"

		path += f" {uci_to_algebraic(move, board)}"
		path = path.strip()

		pi[path] = pi.get(path, 0) + 1
		
		history.append(current)
		current = [g.next() for g in current if g.next() and g.next().move == move]
		board.push(move)

	return pi

def randomWalk2(games, n=pow(10, 5), horizonSize=1, shouldMoveStart=False):
	# Setup variables
	board = Board()
	current = games
	moves = []
	history = []
	path = ""
	pi = {}
	for _ in range(n):
		nGames = len(current)
		choices = {m.next().uci():m.next() for m in current if m.next()}
		move = random.choice(list(choices.values())).move
		moves.append(move)

		moveStart = (not move or nGames==1) and shouldMoveStart
		moveUp = (moves and random.uniform(0, 1) < 1/2) or ((nGames<=horizonSize or not move) and not shouldMoveStart)

		if moveStart:
			board = Board()
			current = games
			moves = []
			history = []
			path = ""
			continue

		if moveUp:
			if history:
				moves.pop()
				board.pop()
				current = history.pop()
				tmp = path.split()
				tmp.pop()
				path = " ".join(tmp)
			continue

		if not move:
			raise "No move down, should have been handled before"

		path += f" {uci_to_algebraic(move, board)}"
		path = path.strip()

		pi[path] = pi.get(path, 0) + 1
		
		history.append(current)
		current = [g.next() for g in current if g.next() and g.next().move == move]
		board.push(move)

	return pi

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


def buildGraphFromCount(pi, limitDepth=-1):
    # Filter paths by depth limit if specified
    if limitDepth != -1:
        remove = [path for path in pi if len(path.split()) > limitDepth]
        for path in remove:
            pi.pop(path, None)
    
    total_visits = float(sum(pi.values()))
    
    # Build a mapping of parent nodes to their children
    children_map = {}
    children_map["root"] = []  # Initialize root's children
    
    for path, count in pi.items():
        tokens = path.split()
        parent = "root" if len(tokens) == 1 else " ".join(tokens[:-1])
        
        if parent not in children_map:
            children_map[parent] = []
        children_map[parent].append((path, count))
    
    # Sort children of each parent by count (descending)
    for parent in children_map:
        children_map[parent].sort(key=lambda x: x[1], reverse=True)
    
    # Build graph with BFS, adding children in sorted order
    G = DiGraph()
    G.add_node("root", weight=0.0)
    
    queue = deque(["root"])
    while queue:
        node = queue.popleft()
        if node in children_map:
            for child_path, count in children_map[node]:
                # Calculate weight for child node
                prob = count / total_visits if child_path in pi else 0.0
                G.add_node(child_path, weight=prob)
                G.add_edge(node, child_path)
                queue.append(child_path)
    
    # Create labels for leaf nodes (last move only)
    labels = {path: path.split()[-1] for path in pi}
    
    return (G, labels)

def draw_chess_trees_grid(
	G_list,
	labels_list,
	draw_labels=False,
	scalar=5000,
	cmap=plt.cm.Reds,
	figsize=(12, 8),
	ncols=2,
):
	if len(G_list) != len(labels_list):
		raise ValueError("G_list and labels_list must have the same length.")

	n_plots = len(G_list)
	ncols = max(1, ncols)
	nrows = math.ceil(n_plots / ncols)

	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
	# Flatten axes array for easy indexing, even if nrows or ncols is 1
	if isinstance(axes, plt.Axes):
		axes = [axes]
	else:
		axes = axes.flatten()

	# Compute global vmin, vmax across all graphs so colors are comparable
	all_weights = []
	for G in G_list:
		w = nx.get_node_attributes(G, "weight")
		if not w:
			raise ValueError("Each graph must have a 'weight' attribute on its nodes.")
		all_weights.extend(w.values())
	global_vmin = min(all_weights)
	global_vmax = max(all_weights)

	for idx, (G, labels) in enumerate(zip(G_list, labels_list)):
		ax = axes[idx]
		weights = nx.get_node_attributes(G, "weight")

		# Node sizes and colors (in the same order as G.nodes())
		node_sizes  = [weights.get(node, 0.0) * scalar for node in G.nodes()]
		node_colors = [weights.get(node, 0.0)       for node in G.nodes()]

		# Compute positions
		pos = graphviz_layout(G, prog="dot")

		# Draw nodes
		nx.draw_networkx_nodes(
			G,
			pos,
			node_size=node_sizes,
			node_color=node_colors,
			cmap=cmap,
			vmin=global_vmin,
			vmax=global_vmax,
			alpha=0.8,
			linewidths=0.4,
			edgecolors="black",
			ax=ax
		)

		# Draw edges
		nx.draw_networkx_edges(
			G,
			pos,
			arrows=True,
			arrowstyle="-|>",
			alpha=0.3,
			width=0.7,
			ax=ax
		)

		# Draw labels (only last‐move or "root")
		if draw_labels:
			nx.draw_networkx_labels(
				G,
				pos,
				labels=labels,
				font_size=8,
				font_color="black",
				verticalalignment="center",
				horizontalalignment="center",
				ax=ax
			)

		ax.set_title(f"Experiment {idx+1}")
		ax.set_axis_off()

		# Add a colorbar to each subplot (optional; comment out if cluttered)
		sm = plt.cm.ScalarMappable(
			cmap=cmap,
			norm=plt.Normalize(vmin=global_vmin, vmax=global_vmax)
		)
		sm.set_array([])
		cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
		cbar.set_label("Stationary Probability", rotation=270, labelpad=15)

	# Hide any unused subplots
	for j in range(n_plots, len(axes)):
		axes[j].set_visible(False)

	plt.tight_layout()
	plt.show()

def calculate_theoretical_stationary_distribution(G):
	alpha = 1.0
	beta = .0
	lambd = 0.5
	weights = nx.get_node_attributes(G, "weight")
	depths = nx.get_node_attributes(G, "depth")
	counts = nx.get_node_attributes(G, "count")
	is_exteriors = nx.get_node_attributes(G, "is_exterior")
	
	tmp = []
	Zint = 0
	Zext = 0
	for key in weights:
		depth = depths.get(key, False)
		count = counts.get(key, False)
		is_exterior =is_exteriors.get(key, False)
		if not depth:
			continue

		ad = alpha*depth
		bc = beta*count
		if is_exterior:
			fn = 1 / (ad)
			Zext += fn
			tmp.append( (fn, key,  is_exterior) )
		else:
			fn = (ad + bc)
			Zint += fn
			tmp.append( (fn, key, is_exterior) )

	out = {}
	for fn, key, iss in tmp:
		out[key] = ( lambd*fn*Zint if iss else (1-lambd)*fn*Zext)/(Zint*Zext)

	nx.set_node_attributes(G, out, "weight")

	out = [out[i] for i in out]
	return out

def calculate_MH_transitions(G:DiGraph, Pi):
	A = nx.adjacency_matrix(G)
	n = len(G.nodes())
	sp.sparse.spdiags(1, 0, n, n)
	return

#white, black = fromAPI(max=20)
games = fromLocalFile(max=1000)
print("Games loaded")

#process = [games, white, black]

graphs = []
labels = []


graph, label = build_tree_from_games(games, 5)
pi_th = calculate_theoretical_stationary_distribution(graph)
graphs.append(graph)
labels.append(label)
print(graph)
##print(pi_th)


"""
for i in range(3):
	pi = randomWalk(games, 1000, horizonSize=1+2*i, shouldMoveStart=True)
	graph, label = buildGraphFromCount(pi)
	graphs.append(graph)
	labels.append(label)
	print(graph)
"""

draw_chess_trees_grid(graphs, labels, draw_labels=False, ncols=1)