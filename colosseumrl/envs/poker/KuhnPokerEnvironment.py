import numpy as np
from typing import Dict, Tuple, List, Union
from colosseumrl.BaseEnvironment import BaseEnvironment, SimpleConfigParser


class KuhnPokerEnvironment(BaseEnvironment):
    parser = SimpleConfigParser((int, 2), (int, None))

    def __init__(self, config: str = None):
        super().__init__(config)
        self.num_players, seed = self.parser.parse(config)
        self.random_state = np.random.RandomState(seed)

    @classmethod
    def create(cls, num_players: int = 2, seed: int = None):
        return cls(cls.parser.store(num_players, seed))

    @property
    def min_players(self) -> int:
        return self.num_players

    @property
    def max_players(self) -> int:
        return self.num_players

    @staticmethod
    def observation_names() -> List[str]:
        return ["hands", "bets"]

    @property
    def observation_shape(self) -> Dict[str, tuple]:
        return {"hands": (self.num_players, ),
                "bets": (self.num_players, )}

    def new_state(self, num_players: int = None) -> Tuple[Tuple[np.ndarray, np.ndarray, int, bool], List[int]]:
        hands = self.random_state.choice(self.num_players + 1, self.num_players, replace=False)
        bets = np.ones(self.num_players, np.int64)
        return (hands, bets, -1, False), [0]

    def showdown(self, hands, bets):
        showdown_players = np.where(bets == np.max(bets))[0]
        showdown_hands = hands[showdown_players]
        showdown_winner = np.argmax(showdown_hands)

        rewards = -np.copy(bets)
        rewards[showdown_winner] += np.sum(bets)
        return [showdown_winner], rewards

    def next_state(self, state: Tuple[np.ndarray, np.ndarray, int, bool], players: [int], actions: [str]):
        hands, bets, last_bet, terminal = state
        hands = hands.copy()
        bets = bets.copy()
        player = players[0]
        action = actions[0]

        # Default return values
        rewards = [0]
        winners = None
        new_players = [(player + 1) % self.num_players]

        # Slightly confusing action mapping
        # -----------------------------------------------
        # At any point there are only two actions
        # A passive action for checking / folding
        # and an aggressive action for betting / calling
        fold = check = (action == "check")
        call = bet = (action == "bet")

        # Process game logic
        # -----------------------------------------------
        # If no outstanding bet
        if last_bet < 0:
            last_player = self.num_players - 1
            # If all players have checked, then showdown
            if check and (player == last_player):
                terminal = True
                new_players = np.arange(self.num_players)
                winners, rewards = self.showdown(hands, bets)

            # First player to bet
            elif bet:
                last_bet = player
                bets[player] += 1

        # If there is already an outstanding bet
        else:
            last_player = (last_bet - 1) % self.num_players
            if call:
                bets[player] += 1

            if player == last_player:
                terminal = True
                new_players = np.arange(self.num_players)
                winners, rewards = self.showdown(hands, bets)

        new_state = (hands, bets, last_bet, terminal)
        return new_state, new_players, rewards, terminal, winners

    def valid_actions(self, state: object, player: int) -> [str]:
        return ["check", "bet"]

    def is_valid_action(self, state: object, player: int, action: str) -> bool:
        return player in ("check", "bet")

    def state_to_observation(self, state: object, player: int) -> Dict[str, np.ndarray]:
        hands, bets, last_bet, terminal = map(np.copy, state)

        # Create masked hand view.
        # If the game isn't over, then we just see ourselves
        # If the game is over, we see all showdown players
        output_hands = np.zeros(self.num_players, np.int64) - 1
        output_hands[player] = hands[player]

        if terminal:
            showdown_players = np.where(bets == np.max(bets))[0]
            for sp in showdown_players:
                output_hands[sp] = hands[sp]

        output_hands = np.roll(output_hands, -player)
        output_bets = np.roll(bets, -player)

        return {"hands": output_hands,
                "bets": output_bets}

