import random
from chess import Board
from stockfish import Stockfish
from getGames import fromAPI, fromLocalFile
import networkx as nx
from networkx import DiGraph
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

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
		
		print(l)
		#print()
		#print(path)
		
		history.append(current)
		current = [g.next() for g in current if g.next() and g.next().move == move]
		board.push(move)

	return pi


def buildGraphFromCount(pi):
	G = DiGraph()

	total_visits = float(sum(pi.values()))
	G.add_node("root", weight=0.0)

	labels = {}
	for path, count in pi.items():
		labels[path] = path.split()[-1]
		print(path, " - ",count)
		prob = count / total_visits

		if path not in G:
			G.add_node(path, weight=prob)

		tokens = path.split()
		if len(tokens) == 1:
			parent = "root"
		else:
			parent = " ".join(tokens[:-1])

		if parent not in G:
			G.add_node(parent, weight=0.0)

		G.add_edge(parent, path)

	return (G, labels)

def draw_chess_tree(G, labels, scalar=10000, cmap=plt.cm.Reds, figsize=(12, 8)):
	"""
	Draws a directed chess‐tree where:
	- G is a NetworkX DiGraph. Each node in G must have a "weight" attribute (a float in [0,1]).
	- labels is a dict: { node_key_in_G : string_to_display }.
		Typically labels[node] = last‐move token, e.g. "e4".
	- scalar is a multiplier for node_size (so weights * scalar → actual circle sizes).
	- cmap is the Matplotlib colormap used to shade nodes by weight.
	- figsize controls the overall figure size.

	Usage:
		# Suppose G was returned by randomWalkGraph(...).
		# Build a labels dict that extracts only the last move:
		labels = {}
		for node in G.nodes():
			if node == "":
				labels[node] = "root"
			else:
				labels[node] = node.split()[-1]

		draw_chess_tree(G, labels)
	"""
	weights = nx.get_node_attributes(G, "weight")
	if not weights:
		raise ValueError("Each node in G must have a 'weight' attribute.")

	# 2) Compute node sizes and colors in the same order as G.nodes()
	node_sizes  = [weights.get(node, 0.0) * scalar for node in G.nodes()]
	node_colors = [weights.get(node, 0.0)       for node in G.nodes()]

	# 3) Compute a layout. Prefer Graphviz "dot" for a tidy top‐down tree.
	pos = graphviz_layout(G, prog="dot")
	

	# 4) Begin drawing
	plt.figure(figsize=figsize)

	# 4a) Draw nodes (size & color by weight)
	nx.draw_networkx_nodes(
		G,
		pos,
		node_size=node_sizes,
		node_color=node_colors,
		cmap=cmap,
		alpha=0.8,
		linewidths=0.4,
		edgecolors="black"
	)

	# 4b) Draw edges (thin/faint arrows)
	nx.draw_networkx_edges(
		G,
		pos,
		arrows=True,
		arrowstyle="-|>",
		alpha=0.3,
		width=0.7
	)

	# 4c) Draw labels using the provided labels dict
	nx.draw_networkx_labels(
		G,
		pos,
		labels=labels,
		font_size=8,
		font_color="black",
		verticalalignment="center",
		horizontalalignment="center"
	)

	# 5) Add a colorbar to interpret node_color → weight
	vmin = min(weights.values())
	vmax = max(weights.values())
	sm = plt.cm.ScalarMappable(
		cmap=cmap,
		norm=plt.Normalize(vmin=vmin, vmax=vmax)
	)
	sm.set_array([])
    

#white, black = fromAPI()
games = fromLocalFile(max=50)
pi = randomWalk(games, 10000)
graph, labels = buildGraphFromCount(pi)
print(graph)
draw_chess_tree(graph, labels)
plt.title("Chess‐Tree Stationary Distribution")
plt.axis("off")
plt.tight_layout()
plt.show()