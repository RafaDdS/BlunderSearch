import io
import berserk
from chess.pgn import read_game
from typing import AnyStr

with open("apikey.txt") as f:
  API_TOKEN = f.readline()

session = berserk.TokenSession(API_TOKEN)
client = berserk.Client(session=session)

defaultUserId = "DiasNoites"

def fromAPI(userId:AnyStr = defaultUserId, max:int = 1000) -> list:
	gamesStream = client.games.export_by_player(userId, True, max=max)
	games = [read_game(io.StringIO(pgn)) for pgn in gamesStream]
	white = [g for g in games if userId in g.headers.get("White", "?")]
	black = [g for g in games if userId in g.headers.get("Black", "?")]
	
	return (white, black)
  
def fromLocalFile(max:int=1000):
	with open("games/lichess_db_standard_rated_2016-02.pgn") as f:
		games = [read_game(f) for _ in range(max)]

	return games