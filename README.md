## Codebase for the UCI multi agent reinforcement learning framework

A framework for developing large multiplayer reinforcement learning agents that can play free-for-all games. Currently available games are:

- Tron: A simple snake-like game where the goal is not to crash into the walls.
- Blokus: A board game of controlling territory with various shaped blocks.
- Tic Tac Toe: Generalization of tic tac toe to 2, 3, and 4 players.


## Requirements
This library requires at least Python 3.5 in order to run correctly.
Python 2.7 is not currently supported.


Basic requirements are listed in the `requirements.txt` and can be installed from PyPi with
`pip install -r requirements.txt`.

## Install
Simply clone the repo and run `pip install -e .` in the root directory
to install a development copy of the library. Full pip install is 
not supported yet.

## Important scripts
`python3 -m colosseumrl.matchmaking.MatchmakingServer` launches the main matchmaking
server for allowing any number of agents to play against each other in a dynamic
way. Run `python -m rlcompetition.matchmaking.MatchmakingServer -h` for
more information. 

`./colosseumrl/examples` contains a list of example scripts that will connect
to a matchmaking server and launch an example agent.
