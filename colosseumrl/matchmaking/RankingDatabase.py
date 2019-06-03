from enum import Enum
from threading import Lock
from sqlite3 import connect as connect_database
from trueskill import Rating, rate
from typing import Dict
from collections import ChainMap


class LoginResult(Enum):
    """ Numerated results after attempting a login. """
    LoginSuccess = 1
    LoginFail = 2
    LoginDuplicate = 3
    NoUser = 4


class RankingDatabase:
    """ Primary class for managing the SQL database storing all of the players
    usernames, password, and rankings.
    """
    LoginResult = LoginResult

    def __init__(self, database_file: str):
        """ SQL database manager for storing player logins and rankings.

        Parameters
        ----------
        database_file : str
            The sqlite file that will be used as the database backend.

        Notes
        -----
        This database stores information in a users table with the following columns
        (username, password_hash, ranking_mean, ranking_confidence)
        """
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
        """ Get all data associated with a particular user.

        Parameters
        ----------
        username : str
            Username of the player you're getting info for

        Returns
        -------
        username : str
            The stored username again
        password : str
            The password hash for the user
        ranking_mean : float
            The point estimate of the players ranking
        ranking_confidence : float
            The variance of the players ranking.
        """
        cursor = self.__db.cursor()
        with self.__db_lock:
            result = cursor.execute("SELECT * FROM userz WHERE username=?", (username.lower(), ))
            return result.fetchone()

    def get_multi(self, *args) -> [(str, str, float, float)]:
        """ Generalization of get to allow for multiple user lookups at once.

        Parameters
        ----------
        args : str
            Usernames as star args.

        Returns
        -------
        List of stored information for each player

        username : str
            The stored username again
        password : str
            The password hash for the user
        ranking_mean : float
            The point estimate of the players ranking
        ranking_confidence : float
            The variance of the players ranking.

        """
        cursor = self.__db.cursor()
        with self.__db_lock:
            query = "SELECT * FROM userz WHERE username IN ({})".format(', '.join('?' for _ in args))
            result = cursor.execute(query, [name.lower() for name in args])
            return result.fetchall()

    def set(self, username: str, password: str) -> None:
        """ Create a new entry in the database for a user.

        Parameters
        ----------
        username : str
            The username of the player.
        password : str
            The password hash of the player.
        """
        cursor = self.__db.cursor()
        with self.__db_lock:
            # Default values for ranking and ranking confidence from trueskill
            cursor.execute("INSERT INTO userz VALUES (?, ?, ?, ?)", (username.lower(), password, 25, 25 / 3))
            self.__db.commit()

    def login(self, username: str, password: str) -> LoginResult:
        """ Perform a login check for a player. This will verify the password, ensure only one session for each player,
        and prevent two people with the same name from connecting.

        Parameters
        ----------
        username : str
            The username of the player.
        password : str
            The password hash of the player.

        Returns
        -------
        LoginResult
            The outcome of the login attempt

        See Also
        --------
        colosseumrl.matchmaking.RankingDatabase.LoginResults
        """
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
        """ Remove a player from the login queue

        Parameters
        ----------
        username : str
            The player that is logging off.
        """
        with self.__db_lock:
            self.__logged_in.remove(username)

    def update_ranking(self, match_ranking: Dict[str, int]) -> None:
        """ Update the rankings of all players in a match.

        Parameters
        ----------
        match_ranking: Dict[str, int]
            A dictionary mapping each username for each player to.
            A lower ranking implies that the player did better, with a ranking of
            0 usually implying that that player won the game.

        """
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
