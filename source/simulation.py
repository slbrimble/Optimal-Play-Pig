# source/simulation.py

import random
import numpy as np
from pig import Pig

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
