from enum import Enum
from threading import Lock
from sqlite3 import connect as connect_database
from trueskill import Rating, rate
from typing import Dict
from collections import ChainMap


class LoginResult(Enum):
    LoginSuccess = 1
    LoginFail = 2
    LoginDuplicate = 3
    NoUser = 4


class RankingDatabase:
    LoginResult = LoginResult

    def __init__(self, database_file: str):
        self.filepath: str = database_file
        self.__db = connect_database(database_file, check_same_thread=False)
        self.__db_lock = Lock()
        self.__logged_in = set()

        cursor = self.__db.cursor()
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='userz'")
        if len(tables.fetchall()) == 0:
            cursor.execute("CREATE TABLE userz (username text, password text, mu real, sigma real)")
            self.__db.commit()

    def __del__(self):
        self.__db.close()

    def get(self, username: str) -> (str, str, float, float):
        cursor = self.__db.cursor()
        with self.__db_lock:
            result = cursor.execute("SELECT * FROM userz WHERE username=?", (username.lower(), ))
            return result.fetchone()

    def get_multi(self, *args) -> [(str, str, float, float)]:
        cursor = self.__db.cursor()
        with self.__db_lock:
            query = "SELECT * FROM userz WHERE username IN ({})".format(', '.join('?' for _ in args))
            result = cursor.execute(query, [name.lower() for name in args])
            return result.fetchall()

    def set(self, username: str, password: str) -> None:
        cursor = self.__db.cursor()
        with self.__db_lock:
            # Default values for ranking and ranking confidence from trueskill
            cursor.execute("INSERT INTO userz VALUES (?, ?, ?, ?)", (username.lower(), password, 25, 25 / 3))
            self.__db.commit()

    def login(self, username: str, password: str) -> LoginResult:
        entry = self.get(username)

        if entry is None:
            return LoginResult.NoUser

        if entry[1] != password:
            return LoginResult.LoginFail

        if username in self.__logged_in:
            return LoginResult.LoginDuplicate

        with self.__db_lock:
            self.__logged_in.add(username)

        return LoginResult.LoginSuccess

    def logoff(self, username: str) -> None:
        with self.__db_lock:
            self.__logged_in.remove(username)

    def update_ranking(self, match_ranking: Dict[str, int]):
        # Transform the usernames to be lowercase for compatibility with database
        match_ranking = {key.lower(): value for key, value in match_ranking.items()}

        # Get the current trueskill ratings for each player
        entries = self.get_multi(*match_ranking.keys())
        old_trueskill_rankings = [{str(name): Rating(mu=mu, sigma=sigma)} for name, _, mu, sigma in entries]
        ordered_match_rankings = [match_ranking[name] for name, _, _, _ in entries]

        # Update rankings using trueskill algorithm
        new_trueskill_rankings = rate(old_trueskill_rankings, ordered_match_rankings)

        # Update database with new rankings
        cursor = self.__db.cursor()
        with self.__db_lock:
            for name, rating in ChainMap(*new_trueskill_rankings).items():
                cursor.execute("UPDATE userz SET mu=?, sigma=? WHERE username=?", (rating.mu, rating.sigma, name))
            self.__db.commit()
