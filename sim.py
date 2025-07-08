import numpy as np
import matplotlib.pyplot as plt
import random

class Player:
    def __init__(self,name,policy,starting_money=0):
        self.name =  name
        self.policy = policy
        self.balance = starting_money
        self.history =[starting_money]
        
    def update(self,reward):
        self.balance += reward
        self.history.append(self.balance)
    
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
    return pot if (high - low - 1) >= 8 else 0


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

if __name__ == "__main__":
    random.seed(42)                   

    deck = Deck()
    players = [
        Player("Alice-Cautious", cautious, starting_money=0),
        Player("Bob-Greedy",     greedy,   starting_money=0),
        Player("Carol-Kelly",    kelly_approx, starting_money=0),
    ]

    pot  = 0
    ante = 1
    n_hands = 500

    for _ in range(n_hands):
        pot = ante_up(players, pot, ante)
        for p in players:
            pot = turn(p, deck, pot)


    for p in players:
        print(f"{p.name}: final bankroll {p.balance}")


    for p in players:
        plt.plot(p.history, label=p.name)
    plt.xlabel("Hands played")
    plt.ylabel("Bankroll (infinite mode ignores balance cap)")
    plt.title("In-Between test run")
    plt.legend()
    plt.show()
