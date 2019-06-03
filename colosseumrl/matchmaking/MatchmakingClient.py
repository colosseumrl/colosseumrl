from typing import NamedTuple
from hashlib import sha256

import grpc
from .grpc_gen.server_pb2 import QuickMatchRequest
from .grpc_gen.server_pb2_grpc import MatchmakerStub


class GameResponse(NamedTuple):
    """ The container object for the response from the server. """
    host: str
    port: int
    username: str
    token: str
    ranking: float


def hash_password(username: str, password: str):
    """ Has a password and salt it with the username. """
    m = sha256()
    m.update(password.encode())
    m.update(username.encode())
    return m.digest()


def request_game(hostname: str, port: int, username: str, password: str = "") -> GameResponse:
    """ Contact a matchmaking server and ask for a new game.

    This function will block until enough players connect to create a server.

    Parameters
    ----------
    hostname: str
    port: int
        Hostname and port of the remote matchmaking server
    username: str
        Username that will identify you in the game.
    password: str
        Password to confirm your identity for ranking and other metadata.

    Returns
    -------
    GameResponse
        host: str
            Hostname of the assigned gamer server.
        port: int
            Port of the game server that was created for you
        username: str
            Your username again to verify.
        token: str
            Authentication string you will need to provide to connect to the match server

    See Also
    --------
    colosseumrl.matchmaking.MatchmakingClient.GameResponse
        The named tuple that will be returned by this function.
    """
    username = username.lower()

    try:
        with grpc.insecure_channel('{}:{}'.format(hostname, port)) as channel:
            hashed_password = hash_password(username, password)
            response = MatchmakerStub(channel).GetMatch(QuickMatchRequest(username=username, password=hashed_password))
    except grpc.RpcError as e:
        raise ConnectionError("Failed to connect to specified host. {}:{}".format(e.code(), e.details()))

    if response.server == "FAIL":
        raise ConnectionError("Could not connect to matchmaking server. Error message: {}".format(response.response))

    _, port = response.server.split(":")
    return GameResponse(hostname, int(port), username, response.auth_key, response.ranking)
