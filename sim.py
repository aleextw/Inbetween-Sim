import numpy as np
import matplotlib.pyplot as plt
import random, math
from collections import defaultdict
import gymnasium as gym
from gymnasium import spaces

class Player:
    def __init__(self, name, policy, starting_money=0):
        self.name     = name
        self.policy   = policy
        self.balance  = starting_money
        self.history  = []

    def update(self, reward):
        self.balance += reward

    
    def decide_bet(self,low,high,pot):
        # return min(self.policy(low, high, pot, self.balance), pot, self.balance)
        return min(self.policy(low, high, pot, self.balance), pot)   # ignore bankroll

        
class Deck:
    def __init__(self):
        self.cards = []

    def shuffle(self):
        self.cards = list(range(1,14))*4
        random.shuffle(self.cards)
    def draw(self):
        if len(self.cards) == 0:
            self.shuffle()
        return self.cards.pop()
    
def cautious(low: int, high: int, pot: int, bal: int) -> int:
    """Bet some beta shit"""
    base = 1                       # change to any amount you want
    if pot >= base:
        bet = base
    else:
        bet = pot                     
    return bet if (high - low - 1) >= 9 else 0


def greedy(low: int, high: int, pot: int, bal: int) -> int:
    """ALL IN BABY"""
    return pot if (high - low - 1) >= 2 else 0


def kelly_approx(low: int, high: int, pot: int, bal: int) -> int:
    """
    Very rough Kelly-style fraction:
    f ≈ (p - q/2), where p = win prob, q = lose prob.
    Caps to player balance and pot.
    """
    gap = high - low - 1
    p = gap / 13                 # ignores 'post' risk (quick & dirty)
    q = 1 - p
    if p <= q:                   # negative EV → pass
        return 0
    f = p - q / 2
    return max(1, int(f * pot))

def turn(player,deck,pot):
    card1,card2 = deck.draw(),deck.draw()
    low = min(card1,card2)
    high = max(card1,card2)

    bet = player.decide_bet(low,high,pot)
    
    if bet == 0:
        player.update(0)
        return pot
    
    target = deck.draw()
    reward = 0
    
    if low < target < high:
        pot -= bet
        reward += bet
    elif target == low or target == high:
        pot += 2*bet
        reward -= 2*bet
    else:
        pot += bet
        reward -= bet
    player.update(reward)
    return pot

def ante_up(players,pot,ante: int = 1):
    if pot > 0:
        return pot

    for p in players:
        pot += ante
        p.update(-ante)
        """contrib = min(ante, p.balance)
        pot += contrib
        p.update(-contrib)
        if p.balance == 0:                     
            players.remove(p)
            print(f"{p.name} is out")"""
    return pot

class InBetweenEnv(gym.Env):
    metadata = {"render_modes":[]}
    def __init__(self,ante: int = 1):
        super().__init__()
        self.ante = ante
        self.deck = Deck()
        self.pot = 0
        
        self.action_space = spaces.Discrete(21)                 # we let 0=pass 1 be 5% and so on to 20 being FULL POT
        self.observation_space = spaces.MultiDiscrete([12,13])    # gap_bucket and pot_bucket
    def _gap_bucket(self,low,high):
        return (high - low - 1)
    
    def _pot_bucket(self):
        return min(self.pot//5, 12)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.pot == 0:
            self.pot += self.ante

        c1, c2 = self.deck.draw(), self.deck.draw()
        self.low, self.high = sorted((c1,c2))
        
        obs = (self._gap_bucket(self.low, self.high), self._pot_bucket)
        info = {"low": self.low, "high": self.high}
        return obs, info
    
    def step(self, action):
        bet_frac = action/20.0
        bet = int(bet_frac * self.pot)
        
        reward = 0
        if bet > 0:
            target = self.deck.draw()
            if self.low < target < self.high:        # win
                self.pot  -= bet
                reward = +bet
            elif target in (self.low, self.high):    # post
                self.pot  += 2 * bet
                reward = -2 * bet
            else:                                    # outside
                self.pot  += bet
                reward = -bet

        terminated = True    # hand finished
        obs = (0, 0)         # dummy; not used after done
        info = {}
        return obs, reward, terminated, False, info
    
def train_q_agent(episodes: int = 400_000, alpha = 0.1, eps_start = 0.2, eps_end = 0.01, eps_decay = 2e5):
    env = InBetweenEnv()
    Q = defaultdict(lambda: np.zeros(env.action_space.n, dtype = float))
    
    for ep in range(episodes):
        state, _ =env.reset()
        eps = eps_end + (eps_start-eps_end) * math.exp(-ep / eps_decay)
        if random.random() < eps:
            action = env.action_space.sample()          # explore
        else:
            action = int(np.argmax(Q[state]))
        _, reward, done, _, _ = env.step(action)
        Q[state][action] += alpha * (reward - Q[state][action])
    return Q
def make_q_policy(Q):
    """Return a function f(low, high, pot, balance) → bet"""
    def q_policy(low, high, pot, bal):
        gap_bucket = min(high - low - 1, 11)
        pot_bucket = min(pot // 5, 12)
        state      = (gap_bucket, pot_bucket)
        action     = int(np.argmax(Q[state]))
        bet        = int((action / 20.0) * pot)
        return bet
    return q_policy

    
'''if __name__ == "__main__":
    LOG_EVERY = 1_000
    n_hands   = 2_000_000
    n_runs    = 10

    for run in range(n_runs):
        print(f"\n=== RUN {run+1} ===")
        random.seed(run)
        deck = Deck()
        players = [
            Player("Alice-Cautious", greedy, starting_money=0),
            Player("Bob-Greedy",     greedy, starting_money=0)
        ]

        pot, ante = 0, 1
        n_samples = (n_hands // LOG_EVERY) + 1
        for p in players:
            p.history = [0] * n_samples    # pre-allocate

        sample_idx = 0

        for hand in range(n_hands):
            pot = ante_up(players, pot, ante)
            for p in players:
                pot = turn(p, deck, pot)
            if len(deck.cards) < 15:
                deck.shuffle()

            if hand % LOG_EVERY == 0:
                for p in players:
                    p.history[sample_idx] = p.balance
                sample_idx += 1
        for pl in players:
            pl.history[sample_idx] = pl.balance
        
        x = range(0, n_hands+1, LOG_EVERY)
        plt.figure(figsize=(8, 3))
        for p in players:
            plt.plot(x, p.history, label=p.name)
            print(f"{p.name}: final bankroll {p.balance}")
        print(f"Pot: {pot}")
        plt.xlabel("Hands played")
        plt.ylabel("Bankroll")
        plt.title(f"Run {run+1} (sampled every {LOG_EVERY} hands)")
        plt.legend()
        plt.tight_layout()
        plt.show()'''
        
if __name__ == "__main__":
    """train"""
    print("Training Q-learner …")
    Q = train_q_agent(episodes=400_000)
    q_policy = make_q_policy(Q)

    """play"""
    print("play")
    LOG_EVERY = 1_000
    n_hands   = 2_000_000
    deck      = Deck()

    players = [
        Player("RL-Bot",   q_policy, starting_money=0),
        Player("Greedy",   greedy,   starting_money=0),
    ]

    pot, ante = 0, 1
    n_samples = (n_hands // LOG_EVERY) + 1
    for p in players:
        p.history = [0] * n_samples
    sample_idx = 0

    for hand in range(n_hands):
        pot = ante_up(players, pot, ante)
        for pl in players:
            pot = turn(pl, deck, pot)
        if len(deck.cards) < 15:
            deck.shuffle()
        if hand % LOG_EVERY == 0:
            for pl in players:
                pl.history[sample_idx] = pl.balance
            sample_idx += 1
    for pl in players:
        pl.history[sample_idx] = pl.balance

    # ---------- plot ----------
    import matplotlib.pyplot as plt
    x = range(0, n_hands + 1, LOG_EVERY)
    plt.figure(figsize=(8,3))
    for p in players:
        plt.plot(x, p.history, label=p.name)
        print(f"{p.name}: final bankroll {p.balance}")
    plt.xlabel("Hands played");  plt.ylabel("Bankroll")
    plt.title("RL-Bot vs. Greedy")
    plt.legend();  plt.tight_layout(); plt.show()