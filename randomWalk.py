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

def randomWalk(games, n=pow(10, 5)):
	board = Board()
	current = games
	moves = []
	history = []
	path = ""
	pi = {}
	for _ in range(n):
		l = len(current)
		p = 1/(l+1)

		move = random.choice(current).next()
		move = move.move if move else move
		moves.append(move)

		if (not move) or (moves and random.uniform(0, 1) < p):
			if history:
				moves.pop()
				board.pop()
				current = history.pop()
				tmp = path.split()
				tmp.pop()
				path = " ".join(tmp)
			continue
		
		path += f" {uci_to_algebraic(move, board)}"
		path = path.strip()

		pi[path] = pi.get(path, 0) + 1
		
		print(path)
		
		history.append(current)
		current = [g.next() for g in current if g.next() and g.next().move == move]
		board.push(move)

	return pi

def randomWalk2(games, n=pow(10, 5)):
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

		moveStart = not move or nGames==1
		moveUp = moves and random.uniform(0, 1) < 1/(nGames+1)

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
		
		path += f" {uci_to_algebraic(move, board)}"
		path = path.strip()

		pi[path] = pi.get(path, 0) + 1
		
		history.append(current)
		current = [g.next() for g in current if g.next() and g.next().move == move]
		board.push(move)

	return pi

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

#white, black = fromAPI()
games = fromLocalFile(max=1000)

graphs = []
labels = []
for i in range(9):
	pi = randomWalk2(games, 1000)
	graph, label = buildGraphFromCount(pi, 3)
	graphs.append(graph)
	labels.append(label)
	print(graph)

draw_chess_trees_grid(graphs, labels, ncols=3)