import numpy as np
from collections.abc import Callable

def value_iteration(states: list, actions: list, probs: Callable[[], np.array], rewards: Callable[[], float],
                    horizon: int = 1000, discount: float = 1) -> [dict, dict]:
    """
    Calculates the value function and optimal policy for a given problem

    Arguments
    ----
    states: a list of all possible states

    actions: a list of all possible action that can be taken
    
    probs: a function for transition probabilities between states given an action is taken. The inputs to the function should be the state that you are
    currently in and the action taken, and the function should return a list or np.array with a probability of moving to each state.
    
    rewards: a function for the (immediate) rewards for each action in each state. The inputs to the function should be the state that you are
    currently in and the action taken, and the function should return a numeric value giving the immediate reward for taking the action in that state.

    horizon: the time horizon the problem is over i.e. how many actions can be taken before the end. For infinite horizon problems, this should be large
    enough that the value function is changing less than the precision required each iteration.

    discount: for infinite horizon problems, how much future rewards are discounted versus immediate rewards

    Output
    ---
    policy_dict: the optimal policy. A dictionary with the keys being the list of states and the values being the optimal action to take given
    that you are in that state.
    
    val_dict: A dictionary with the keys being the list of state and the dictionary values being the value function (found by value iteration) of
    each state

    """

    #Checking inputs for validity and returning appropriate error messages
    if discount > 1 or discount < 0:
        return("Error: the discount must be between 0 and 1")
    if horizon < 0 or int(horizon) != horizon:
        return("Error: the horizon must be a positive integer")
    
    
    val_dict = dict(zip(states,[0]*len(states)))
    val_action = {}
    k = 0
    #Find value function for all states
    for k in range(horizon):
        k = k + 1
        for s in states:
            for a in actions:
                #print(probs(state = s, action = a))
                #print(type(probs(state =s, action = a)))
                #print(val_dict.values())
                val_action[a] = rewards(state = s, action = a) + discount*sum(probs(state = s, action = a)*list(val_dict.values()))
            val_dict[s] = float(max((val_action.values())))
    #Find optimal policy by finding what action leads to highest value function for all states
    policy_dict = {}
    for s in states:
        for a in actions:
            val_action[a] = rewards(state = s, action = a) + discount*sum(probs(state = s, action = a)*list(val_dict.values()))
        policy_dict[s] = max(val_action, key = val_action.get)
    return(policy_dict, val_dict)
