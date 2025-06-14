import io
import berserk
from chess.pgn import read_game
from typing import AnyStr
import random
import pickle

with open("apikey.txt") as f:
  API_TOKEN = f.readline()

session = berserk.TokenSession(API_TOKEN)
client = berserk.Client(session=session)

defaultUserId = "DiasNoites"
file = "games/lichess_db_standard_rated_2016-02.pgn"
offsets = []

try:
	with open("game_index.pkl", "rb") as f:
		offsets = pickle.load(f)
except:
	print("No file index")

def build_game_index():
	"""
	Scan file and return a list of byte-offsets for the start of every
	game.
	"""
	with open(file) as f:
		while True:
			pos = f.tell()
			line = f.readline()
			if not line:
				break
			if line.startswith("[Event"):
				offsets.append(pos)

	with open("game_index.pkl", "wb") as f:
		pickle.dump(offsets, f)

def fromAPI(userId:AnyStr = defaultUserId, max:int = 1000) -> list:
	gamesStream = client.games.export_by_player(userId, True, max=max)
	games = [read_game(io.StringIO(pgn)) for pgn in gamesStream]
	white = [g for g in games if userId in g.headers.get("White", "?")]
	black = [g for g in games if userId in g.headers.get("Black", "?")]
	
	return (white, black)

def fromLocalFile(max:int=1000, rand=False):
	if not offsets:
		build_game_index()
	with open(file) as f:
		if rand:
			chosen_offset = random.choices(offsets, k=max)
			games = []
			for i in chosen_offset:
				f.seek(i)
				games.append(read_game(f))
		else:
			games = [read_game(f) for _ in range(max)]

	return games