import numpy as np

def optimistic_greedy_bandit(
    true_rewards: list,
    initial_q: float,
    n_steps: int,
    step_size: float
) -> tuple:
    """
    Simulate a greedy bandit agent with optimistic initialization.
    
    Args:
        true_rewards: List of true deterministic rewards for each arm
        initial_q: Optimistic initial Q-value for all arms
        n_steps: Number of steps to simulate
        step_size: Constant step-size (alpha) for Q-value updates
    
    Returns:
        Tuple of (Q_values, action_counts) where Q_values is a list of
        floats rounded to 4 decimal places, and action_counts is a list of ints.
    """
    action_counts=[0]*len(true_rewards)
    Q_values=[initial_q]*len(true_rewards)
    for i in range(n_steps):
        max_ind=-1
        maxi=-1
        for j in range(len(Q_values)):
            if maxi<Q_values[j]:
                maxi=Q_values[j]
                max_ind=j
        action_counts[max_ind]+=1
        Q_values[max_ind]=Q_values[max_ind]+step_size*(true_rewards[max_ind]-Q_values[max_ind])
    return (Q_values,action_counts)



