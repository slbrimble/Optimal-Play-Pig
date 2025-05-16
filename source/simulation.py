# source/simulation.py

import random
import numpy as np
from pig import Pig
import random
from statistics import mean

def simulate_one(pig: Pig, start_i=0, start_j=0):
    """
    Simulate one game under pig.policy.
    Returns (turns_taken, margin) if the starting player wins,
    or (None, None) if the opponent wins.
    """
    T = pig.T
    scores = [start_i, start_j]
    turn = 0
    turns_taken = 0

    while True:
        i, j = scores[turn], scores[1-turn]
        k = 0

        # If already terminal (shouldn't happen), break
        if i >= T or j >= T:
            break

        # Play one turn
        while True:
            # Terminal check: if our next hold would win, do it
            if i + k >= T:
                scores[turn] += k
                if turn == 0:
                    turns_taken += 1
                break

            # Look up policy now safe: k < T−i
            action = pig.policy.get((i, j, k), "hold")
            if action == "hold":
                scores[turn] += k
                if turn == 0:
                    turns_taken += 1
                break

            # Otherwise roll
            r = random.randint(1, 6)
            if r == 1:
                # Pig out: turn ends with zero points
                if turn == 0:
                    turns_taken += 1
                break
            else:
                k += r

        # Check for win
        if scores[turn] >= T:
            if turn == 0:
                return turns_taken, scores[0] - scores[1]
            else:
                return None, None

        # Next player's turn
        turn = 1 - turn

def simulate_many(pig: Pig, n=5000):
    """
    Simulate n games for each possible starting score i=0..T-1 (opp=0).
    Returns two arrays:
      avg_turns[i]  = average turns to win starting at i,
      avg_margin[i] = average margin of victory starting at i.
    """
    T = pig.T
    avg_turns = np.zeros(T)
    avg_margin = np.zeros(T)
    counts = np.zeros(T)

    for start_i in range(T):
        for _ in range(n):
            result = simulate_one(pig, start_i=start_i, start_j=0)
            if result[0] is not None:
                t, m = result
                avg_turns[start_i] += t
                avg_margin[start_i] += m
                counts[start_i] += 1

        if counts[start_i] > 0:
            avg_turns[start_i] /= counts[start_i]
            avg_margin[start_i] /= counts[start_i]

    return avg_turns, avg_margin



def hold_at_twenty(i, j, k):
    """
    A simple heuristic policy: continue rolling until the turn score (k) reaches 20.

    Parameters:
    - i: current player's score
    - j: opponent's score
    - k: current turn total

    Returns:
    - 'roll' if k < 20, otherwise 'hold'
    """
    return 'roll' if k < 20 else 'hold'


def optimal_policy(i, j, k, result_pig):
    """
    Retrieve the action from a precomputed optimal policy.

    Parameters:
    - i: current player's score
    - j: opponent's score
    - k: current turn total
    - result_pig: a Pig object with a .policy attribute populated by value iteration

    Returns:
    - The optimal action ('roll' or 'hold') for the state (i, j, k)
    """
    return result_pig.policy[i, j, k]


def game(strats):
    """
    Simulates a single game of Pig with two strategies.

    Parameters:
    - strats: a list of two policy functions. Each function takes (i, j, k) as input.

    Returns:
    - The index (0 or 1) of the winning player
    """
    scores = [0, 0]
    while max(scores) < 100:
        for i in range(2):
            round_score = 0
            roll = 0
            policy = strats[i]
            # Player's turn
            while policy(scores[i], scores[1 - i], round_score) == 'roll' and roll != 1:
                roll = random.choice([1, 2, 3, 4, 5, 6])
                round_score = (roll != 1) * (round_score + roll)
                if scores[i] + round_score >= 100 and roll != 1:
                    return i
            scores[i] += round_score
    return scores.index(max(scores))


def tournament(n, result_pig):
    """
    Simulates multiple games to compare the optimal policy with itself and with
    a simple 'hold-at-twenty' policy.

    Parameters:
    - n: number of games to simulate for each scenario
    - result_pig: Pig object with an optimal policy computed

    Returns:
    - A list of estimated winning probabilities:
        [optimal vs optimal (goes second),
         optimal vs hold-at-20 (goes second),
         hold-at-20 vs optimal (goes second)]
    """
    def op(i, j, k):
        return optimal_policy(i, j, k, result_pig)

    opt_v_opt = [1 - game([op, op]) for _ in range(n)]
    opt_v_hold = [1 - game([op, hold_at_twenty]) for _ in range(n)]
    hold_v_opt = [1 - game([hold_at_twenty, op]) for _ in range(n)]

    return [mean(opt_v_opt), mean(opt_v_hold), mean(hold_v_opt)]


def compute_confidence_intervals(sim_fn, policy, n_repeats=100, games_per_repeat=1000, z=1.96):
    """
    Compute 95% confidence intervals for win probabilities across repeated simulations.

    Parameters:
    ----------
    sim_fn : callable
        A function that runs the tournament simulation and returns a list of win probabilities.
    n_repeats : int
        Number of times to repeat the simulation.
    games_per_repeat : int
        Number of games to simulate per tournament.
    z : float
        Z-score for desired confidence level (default: 1.96 for 95%).

    Returns:
    -------
    CIs : list of lists
        Each inner list contains the lower and upper bounds of the 95% confidence interval 
        for a win probability.
    means : list of floats
        Mean estimated win probabilities.
    stds : list of floats
        Standard deviations of win probabilities.
    all_results : list of lists
        Raw win probabilities from each simulation run (shape: n_repeats × scenarios).
    """
    from statistics import mean, stdev

    all_results = [sim_fn(games_per_repeat,policy) for _ in range(n_repeats)]
    means = [mean([all_results[i][j] for i in range(n_repeats)]) for j in range(3)]
    stds = [stdev([all_results[i][j] for i in range(n_repeats)]) for j in range(3)]
    CIs = [[means[i] + u * z * stds[i] / (n_repeats ** 0.5) for u in [-1, 1]] for i in range(3)]

    return CIs, means, stds, all_results
