from .matchmaking.MatchmakingServer import main
from .rl_logging import init_logging

if __name__ == '__main__':
    logger = init_logging()
    main()