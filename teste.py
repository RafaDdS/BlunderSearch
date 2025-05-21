import io
import random
import berserk
from chess import Board
from chess.pgn import read_game
from stockfish import Stockfish

with open("apikey.txt") as f:
  API_TOKEN = f.readline()

session = berserk.TokenSession(API_TOKEN)
client = berserk.Client(session=session)

#stockfish = Stockfish(path="./stockfish/stockfish-ubuntu-x86-64-avx2")

userId = "DiasNoites"

gamesStream = client.games.export_by_player(userId, True, max=1000)

games = [read_game(io.StringIO(pgn)) for pgn in gamesStream]
whites = [g for g in games if userId in g.headers.get("White", "?")]
blacks = [g for g in games if userId in g.headers.get("Black", "?")]

board = Board()
current = whites
while current:
	move = random.choice(current).next()
	move = move.move if move else move

	print(len(current))
	print()
	print(move)

	if not move:
		break
	
	current = [g.next() for g in current if g.next().move == move]
	board.push(move)

print(board.fen()) 