import random

import gymnasium as gym
from gymnasium import spaces

import numpy as np

PLAYER_STARTING_BANK = 500_000
PLAYER_COUNT = 4
NUM_OBSERVATION_BINS = 12


class Deck:
    def __init__(self, seed):
        self.seed = seed
        random.seed(self.seed)
        self.cards = []

    def draw_card(self):
        if self.is_empty():
            self.reset()

        return self.cards.pop()

    def is_empty(self):
        return len(self.cards) == 0

    def size(self):
        return len(self.cards)

    def reset(self):
        self.cards = [i for i in range(1, 14) for _ in range(4)]
        random.shuffle(self.cards)


class InBetween:
    def __init__(self, seed, player_bank, player_count):
        self.seed = seed
        self.deck = Deck(seed)
        self.player_count = player_count
        self.current_player = 0
        self.player_bank = [player_bank for _ in range(player_count)]
        self.pair = [0, 0]
        self.pot = 0

    def step(self, action):
        self.resolve_action(action)
        self.current_player = (self.current_player + 1) % 4
        return self.draw_pair()

    def resolve_action(self, action):
        bet_amount = int(
            (action / 20) * min(self.pot, self.player_bank[self.current_player])
        )

        if bet_amount == 0:
            return

        card3 = self.deck.draw_card()
        low_card, high_card = self.pair

        if card3 == low_card or card3 == high_card:
            delta = -2 * bet_amount
        elif low_card < card3 < high_card:
            delta = bet_amount
        else:
            delta = -bet_amount

        self.player_bank[self.current_player] += delta

    def new_pot(self):
        self.pot += self.player_count
        for i in range(self.player_count):
            self.player_bank[i] -= 1

    def draw_pair(self):
        if self.pot == 0:
            self.new_pot()

        if self.deck.size() < 3:
            self.deck.reset()

        card1 = self.deck.draw_card()
        card2 = self.deck.draw_card()

        if card1 == card2:
            return self.draw_pair()

        low_card, high_card = (card1, card2) if card1 < card2 else (card2, card1)
        self.pair = [low_card, high_card]

        window_size = high_card - low_card - 1
        return [
            window_size,
            min(
                min(self.pot, self.player_bank[self.current_player]) // 5,
                NUM_OBSERVATION_BINS - 1,
            ),
        ]


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        """
        Action = int(action / 20 * min(pot, bank))
        Observation = [window_size, pot // 5]
        """
        self.action_space = spaces.Discrete(21)
        self.observation_space = spaces.MultiDiscrete([12, NUM_OBSERVATION_BINS])
        self.game = None

    def step(self, action):
        # Play for other 3 players with the same policy
        self.game.step(action)
        for _ in range(PLAYER_COUNT - 1):
            observation = self.game.step(self.action_space.sample())

        self.total_reward = self.game.player_bank[0] - PLAYER_STARTING_BANK
        self.reward = self.total_reward - self.prev_reward
        self.prev_reward = self.total_reward

        if self.game.player_bank[0] <= 0:
            self.done = True

        if self.steps == 100:
            self.truncated = True

        self.steps += 1

        return (
            np.array(observation),
            self.reward,
            self.done,
            self.truncated,
            {},
        )

    def reset(self, seed=None, options=None):
        self.game = InBetween(seed, PLAYER_STARTING_BANK, PLAYER_COUNT)
        self.score = 0
        self.prev_reward = 0
        self.steps = 0
        self.done = False
        self.truncated = False

        return np.array(self.game.draw_pair()), {}

    def render(self):
        print(f"Banks: {', '.join(map(str, self.game.player_bank))}")

    def close(self): ...
