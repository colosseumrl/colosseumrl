import struct
import logging

from time import sleep, time
from typing import Callable, Type, Optional
from spacetime import Dataframe, Node

from .data_model import ServerState, Player, Observation
from .BaseEnvironment import BaseEnvironment
from .ClientEnvironment import ClientEnvironment

logger = logging.getLogger(__name__)


def client_app(dataframe: Dataframe,
               app: "RLApp",
               client_function: Callable,
               observation_class: Type[Observation],
               dimension_names: [str],
               host: str,
               auth_key: str,
               *args, **kwargs):

    client_env = app.client_environment(dataframe=dataframe,
                                        dimensions=dimension_names,
                                        observation_class=observation_class,
                                        server_environment=app.server_environment,
                                        host=host,
                                        auth_key=auth_key)

    client_function(client_env, *args, **kwargs)


class RLApp:
    def __init__(self,
                 host: str,
                 port: int,
                 auth_key: str = '',
                 client_environment: Type[ClientEnvironment] = ClientEnvironment,
                 server_environment: Optional[Type[BaseEnvironment]] = None,
                 time_out: int = 0):
        """ The decorator for online reinforcement learning agents.

        Any function that is using the client environment should be wrapped in this in order to properly set up
        and connect to the server.

        Notes
        -----
        That the wrapping function must take the client environment as the first parameter.

        Parameters
        ----------
        host : str
            Hostname of game server.
        port : int
            Port of the game server.
        auth_key : str
            Authorization key if the server is whitelisted.
        client_environment : Type[ClientEnvironment], optional.
            The client environment to create when connected. By default, this will create a generic client environment.
            Most games will have a provided client environment with specialized functions for interacting
            with the environment.
        server_environment : Type[BaseEnvironment], optional.
            The server environment if you know what game and environment the server is running.
            While this parameter is technically optional, unless you're running some strange dynamic game with
            different possible environments at the same time, you should provide this so that
            all of the functionality of the client environment is enabled.
        time_out : int, optional.
            The timeout for connecting to the server.
        """
        self.client_environment = client_environment
        self.server_environment = server_environment
        self.host = host
        self.port = port
        self.auth_key = auth_key
        self.time_out = time_out

    def __call__(self, main_func: Callable):
        # Get the dimensions required for the player dataframe
        start_time = time()

        while self.time_out == 0 or (time() - start_time) < self.time_out:
            try:
                while True:
                    try:
                        df = Dataframe("dimension_getter", [ServerState], details=(self.host, self.port))
                    except ConnectionRefusedError as e:
                        if (time() - start_time) > self.time_out:
                            raise e
                    else:
                        break

                df.pull()
                df.checkout()

                if df.read_all(ServerState)[0].server_no_longer_joinable:
                    # This server is from an old game and just hasn't exited yet, wait for a new server.
                    sleep(0.1)
                    continue
                else:
                    break

            except (ConnectionResetError, struct.error):
                sleep(0.1)
                continue

        # If we know what game were supposed to be playing, then check to make sure
        # we match the server environment
        server_state = df.read_all(ServerState)[0]
        environment_name = server_state.env_class_name
        dimension_names: [str] = server_state.env_dimensions
        if self.server_environment is not None and self.server_environment.__name__ != environment_name:
            raise ValueError("Client and Server environment mismatch. We are using: {}. Server is using: {}".format(
                self.server_environment.__name__,
                environment_name
            ))

        # Create the correct observation type for our connecting dataframe
        observation_class = Observation(dimension_names)
        del df

        def app(*args, **kwargs):
            client = Node(client_app,
                          dataframe=(self.host, self.port),
                          Types=[Player, observation_class, ServerState],
                          threading=True)
            client.start(self, main_func, observation_class, dimension_names, self.host, self.auth_key, *args, **kwargs)

        return app


def create_rl_agent(agent_fn: Callable[[ClientEnvironment], None],
                    host: str,
                    port: int,
                    auth_key: str = '',
                    client_environment: Type[ClientEnvironment] = ClientEnvironment,
                    server_environment: Optional[Type[BaseEnvironment]] = None,
                    time_out: int = 0):
    """ Create an online reinforcement learning agent from an agent function.

    Parameters
    ----------
    agent_fn :
        Agent function that has the client environment as the first parameter.
    host : str
        Hostname of game server.
    port : int
        Port of the game server.
    auth_key : str
        Authorization key if the server is whitelisted.
    client_environment : Type[ClientEnvironment], optional.
        The client environment to create when connected. By default, this will create a generic client environment.
        Most games will have a provided client environment with specialized functions for interacting
        with the environment.
    server_environment : Type[BaseEnvironment], optional.
        The server environment if you know what game and environment the server is running.
        While this parameter is technically optional, unless you're running some strange dynamic game with
        different possible environments at the same time, you should provide this so that
        all of the functionality of the client environment is enabled.
    time_out : int, optional.
        The timeout for connecting to the server.

    Returns
    -------
    RLApp
        An application ready to run with the agent function.

    """
    return RLApp(host, port, auth_key, client_environment, server_environment, time_out)(agent_fn)


def launch_rl_agent(agent_fn: Callable[[ClientEnvironment], None],
                    host: str,
                    port: int,
                    auth_key: str = '',
                    client_environment: Type[ClientEnvironment] = ClientEnvironment,
                    server_environment: Optional[Type[BaseEnvironment]] = None,
                    time_out: int = 0,
                    **kwargs):
    """ Create and launch an online reinforcement learning agent from an agent function.

    Parameters
    ----------
    agent_fn :
        Agent function that has the client environment as the first parameter.
    host : str
        Hostname of game server.
    port : int
        Port of the game server.
    auth_key : str
        Authorization key if the server is whitelisted.
    client_environment : Type[ClientEnvironment], optional.
        The client environment to create when connected. By default, this will create a generic client environment.
        Most games will have a provided client environment with specialized functions for interacting
        with the environment.
    server_environment : Type[BaseEnvironment], optional.
        The server environment if you know what game and environment the server is running.
        While this parameter is technically optional, unless you're running some strange dynamic game with
        different possible environments at the same time, you should provide this so that
        all of the functionality of the client environment is enabled.
    time_out : int, optional.
        The timeout for connecting to the server.
    **kwargs
        Any additional options that will be passed into your client function

    Returns
    -------
    object
        The return values of your agent function.
    """
    return create_rl_agent(agent_fn, host, port, auth_key, client_environment, server_environment, time_out)(**kwargs)




