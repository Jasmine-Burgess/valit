# Valit Package: Value Iteration

## Overview

This package implements value iteration for Markov Decision Processes (MDPs), returning an optimal policy of the best action to take in each state to maximise total rewards, and the value function giving the value of each state.

## Installation

#### Installing from PyPi

```
pip install valit
```
### Installing from github with pip

```
!python -m pip install git+https://github.com/Jasmine-Burgess/valit
```
The package can then be imported:
```
import valit
```


## Examples

### Example 1 - Two State MDP

This example is taken from [Artificial Intelligence: Foundations and Computational Agents 2nd edition](https://artint.info/2e/html2e/ArtInt2e.Ch9.S5.html#Ch9.Thmciexamplered27) and is a simple two stage Markov Decision Process. Imagine Sam wishes to decide whether to party to relax. He enjoys partying more but is worried about getting sick. This can be modelled as a MDP with the states and actions as follows:
```
states_example1 = ["healthy", "sick"]
actions_example1 = ["party", "relax"]
```
Sam estimates the dynamics P(s' | s, a): the probabilities of being healthy the next day for each state and action, as follows:

![image](https://github.com/user-attachments/assets/d17a7883-b6db-43e4-8e0f-a0c64235e958)

These transition probabilities can be represented by the following function, which takes a state and an action as an input and returns a numpy array of probabilities.

```
import numpy as np

def probs_example1 (state, action):
    if (state == "healthy"):
        if (action == "relax"):
            return(np.array([0.95, 0.05]))
        if (action == "party"):
            return(np.array([0.7, 0.3]))
    else:
        if (action == "relax"):
            return(np.array([0.5, 0.5]))
        if (action == "party"):
            return(np.array([0.1, 0.9]))
```
Sam also determines the immediate rewards based on his enjoyment as follows: 

![image](https://github.com/user-attachments/assets/881ae246-3798-4643-a99b-31f698dd0df8)

He always prefers partying to relaxing but prefers to be in the healthy state rather than the sick state. Therefore, the immediate rewards can be represented by the following function. It takes state and an action as an input and returns a numeric (in this case int) value.

```
def rewards_example1 (state, action):
    if (state == "healthy"):
        if (action == "relax"):
            return(7)
        if (action == "party"):
            return(10)
    else:
        if (action == "relax"):
            return(0)
        if (action == "party"):
            return(2)
```
This is an infinite horizon problem, as there is no fixed endpoint where Sam will stop partying or relaxing, so the default horizon of 1000 is used. A discount of 0.8 assigns more importance to immediate rewards than future rewards. 

The value_iteration function can be used as follows to determine the best policy to maximise rewards over a long time period.
```
valit.value_iteration(states = states_example1, actions = actions_example1, probs = probs_example1, rewards = rewards_example1, discount = 0.8)
```

```
({'healthy': 'party', 'sick': 'relax'},
 {'healthy': 35.71428571428571, 'sick': 23.80952380952381})
```

The code returns a policy of partying when healthy and relaxing when sick. It also returns the value function which is 35.71428571 for the healthy state and 23.8095238 for the sick state; the value function is higher for the healthy state as the rewards are higher there.

## Example 2 - Grid World

Grid worlds are a common problem used in robotics and reinforcement learning. The following grid world problem is taken from [Artificial Intelligence: Foundations and Computational Agents 2nd edition](https://artint.info/2e/html2e/ArtInt2e.Ch9.S5.html#Ch9.Thmciexamplered28).

![image](https://github.com/user-attachments/assets/4784c8f2-4768-4783-a01f-832327b69499)

Consider the following setup. A robot must navigate on the grid shown above. The robot can choose to move one square up, down, left, or right. There is 0.7 chance of going in this direction, and 0.1 chance of moving in any of the other three directions. However, if the location computed is outside of the grid - the robot has bumped against the walls, the robot recieves a reward of -1 and the robot does not actually move. 

The states will be represented as coordinates, going from 0 to 9 in both the x and y directions. Using this notation, the 4 actions: moving up, down, left, or right can be represented by [(1,0), (0,1), (-1,0), (0, -1)].

```
import itertools
grid_states = list(itertools.product(range(10),range(10)))
grid_actions = [(1,0), (0,1), (-1,0), (0, -1)]
```
There are additionally squares which confer rewards as shown on the grid. In each of these states, the robot gets the reward after it carries out an action in that state, not when it enters the state. When the agent reaches one of the states with positive reward, no matter what action it performs, at the next step it is thrown randomly to one of the four corners of the grid.

```
reward_states = [(7,2), (8,7) ] 
penalty_states = [(3,4), (3,7) ]

def grid_probs(state, action):
    probs = dict(zip(grid_states, [0 for i in range(100)]))
    #If you are in one of the positive rewarding state, the robot is flung to one of the 4 corners with equal probability
    if state in reward_states:
        for corner in [(0,0), (0,9), (9,0), (9,9)]:
            probs[corner] = 0.25
    else:
    #Otherwise attempt to move to the state given by the action.
        attempted_state = tuple(map(lambda i, j: i + j, state, action))
        if attempted_state in grid_states:
            probs[attempted_state] = 0.7
        else: #if attempting to move out of the grid, stay in same state with probability 0.7.
            probs[state] = 0.7
        for move in [x for x in grid_actions if x != action]: #for each of the other actions which the robot did not attempt, assign the 0.1 probability
            new_state = tuple(map(lambda i, j: i + j, state, move))
            if new_state in grid_states:
                probs[new_state] = 0.1
            else:
                probs[state] += 0.1
    return(np.array(list(probs.values()))) #convert dictionary to np.array for use in value_iteration function

def grid_reward(state, action):
    if state == (7,2):
        return(3)
    elif state == (8,7):
        return(10)
    elif state == (3,4):
        return(-5)
    elif state == (3,7):
        return(-10)
    else:
        attempted_state = tuple(map(lambda i, j: i + j, state, action))
        if attempted_state in grid_states:
            return(0)
        else:
            return(-1)
```
As before, the probabibility and reward functions require inputs of a state and a action, and the reward function returns a numeric value and the probability function returns a numpy array.

The optimal policy and value function can thus be calculated:
```
valit.value_iteration(states = grid_states, actions = grid_actions, probs = grid_probs, rewards = grid_reward, horizon = 50, discount = 0.8)
```
```
({(0, 0): (1, 0),
  (0, 1): (1, 0),
  (0, 2): (1, 0),
  (0, 3): (0, -1),
  (0, 4): (0, -1),
  (0, 5): (1, 0),
  (0, 6): (0, 1),
  (0, 7): (0, 1),
  (0, 8): (0, 1),
  (0, 9): (1, 0),
  (1, 0): (1, 0),
  (1, 1): (1, 0),
  (1, 2): (1, 0),
  (1, 3): (0, -1),
  (1, 4): (0, -1),
  (1, 5): (1, 0),
  (1, 6): (0, -1),
  (1, 7): (0, 1),
  (1, 8): (0, 1),
  (1, 9): (1, 0),
  (2, 0): (1, 0),
  (2, 1): (1, 0),
  (2, 2): (1, 0),
  (2, 3): (0, -1),
  (2, 4): (0, -1),
  (2, 5): (1, 0),
  (2, 6): (1, 0),
  (2, 7): (0, 1),
  (2, 8): (0, 1),
  (2, 9): (1, 0),
  (3, 0): (1, 0),
  (3, 1): (1, 0),
  (3, 2): (1, 0),
  (3, 3): (1, 0),
  (3, 4): (1, 0),
  (3, 5): (1, 0),
  (3, 6): (1, 0),
  (3, 7): (1, 0),
  (3, 8): (1, 0),
  (3, 9): (1, 0),
  (4, 0): (1, 0),
  (4, 1): (1, 0),
  (4, 2): (1, 0),
  (4, 3): (1, 0),
  (4, 4): (1, 0),
  (4, 5): (1, 0),
  (4, 6): (1, 0),
  (4, 7): (1, 0),
  (4, 8): (1, 0),
  (4, 9): (1, 0),
  (5, 0): (1, 0),
  (5, 1): (1, 0),
  (5, 2): (1, 0),
  (5, 3): (1, 0),
  (5, 4): (1, 0),
  (5, 5): (1, 0),
  (5, 6): (1, 0),
  (5, 7): (1, 0),
  (5, 8): (1, 0),
  (5, 9): (1, 0),
  (6, 0): (0, 1),
  (6, 1): (0, 1),
  (6, 2): (1, 0),
  (6, 3): (1, 0),
  (6, 4): (1, 0),
  (6, 5): (1, 0),
  (6, 6): (1, 0),
  (6, 7): (1, 0),
  (6, 8): (1, 0),
  (6, 9): (1, 0),
  (7, 0): (0, 1),
  (7, 1): (0, 1),
  (7, 2): (1, 0),
  (7, 3): (0, -1),
  (7, 4): (0, 1),
  (7, 5): (0, 1),
  (7, 6): (1, 0),
  (7, 7): (1, 0),
  (7, 8): (1, 0),
  (7, 9): (0, -1),
  (8, 0): (0, 1),
  (8, 1): (0, 1),
  (8, 2): (-1, 0),
  (8, 3): (0, 1),
  (8, 4): (0, 1),
  (8, 5): (0, 1),
  (8, 6): (0, 1),
  (8, 7): (1, 0),
  (8, 8): (0, -1),
  (8, 9): (0, -1),
  (9, 0): (0, 1),
  (9, 1): (-1, 0),
  (9, 2): (-1, 0),
  (9, 3): (0, 1),
  (9, 4): (0, 1),
  (9, 5): (0, 1),
  (9, 6): (0, 1),
  (9, 7): (-1, 0),
  (9, 8): (0, -1),
  (9, 9): (0, -1)},
 {(0, 0): 0.24827891604612456,
  (0, 1): 0.2844401120493796,
  (0, 2): 0.2994695470265028,
  (0, 3): 0.2225623422430906,
  (0, 4): 0.16472212713533485,
  (0, 5): 0.14888583778217865,
  (0, 6): 0.14637310999156036,
  (0, 7): 0.19706106790952738,
  (0, 8): 0.2748721158865317,
  (0, 9): 0.37150761993607606,
  (1, 0): 0.3317843659979658,
  (1, 1): 0.3890449076874794,
  (1, 2): 0.41955695361014955,
  (1, 3): 0.29845834448031067,
  (1, 4): 0.18748261206019748,
  (1, 5): 0.20015618542199307,
  (1, 6): 0.15497876325568116,
  (1, 7): 0.19572557926172116,
  (1, 8): 0.36341604493158153,
  (1, 9): 0.5179941308803114,
  (2, 0): 0.454028293097821,
  (2, 1): 0.5467544518128044,
  (2, 2): 0.608212891420127,
  (2, 3): 0.38378589674052505,
  (2, 4): -0.11055381241858174,
  (2, 5): 0.28722947245719965,
  (2, 6): 0.1940428269383242,
  (2, 7): -0.44938145223887505,
  (2, 8): 0.44614483601411187,
  (2, 9): 0.7460013812396428,
  (3, 0): 0.6203982981749986,
  (3, 1): 0.7690208183382373,
  (3, 2): 0.8932235601447032,
  (3, 3): 0.35192917833907855,
  (3, 4): -4.543135717248232,
  (3, 5): 0.472389200828577,
  (3, 6): 0.3475299537746773,
  (3, 7): -9.130049510095848,
  (3, 8): 0.44076701560041365,
  (3, 9): 1.0878396698286925,
  (4, 0): 0.8445046763592107,
  (4, 1): 1.07891214625062,
  (4, 2): 1.3480189513121075,
  (4, 3): 1.095034575593458,
  (4, 4): 0.7138628742513197,
  (4, 5): 1.4018917787309455,
  (4, 6): 1.8296775703112873,
  (4, 7): 1.5050666556385655,
  (4, 8): 1.8722362172160012,
  (4, 9): 1.6176269319026142,
  (5, 0): 1.1446420919698974,
  (5, 1): 1.5035512031423415,
  (5, 2): 1.9690096564231512,
  (5, 3): 1.6105888798651327,
  (5, 4): 1.5670709132554503,
  (5, 5): 2.0725312407855627,
  (5, 6): 2.8023546117703098,
  (5, 7): 3.4630671234896724,
  (5, 8): 2.8342132697744753,
  (5, 9): 2.2346620707865124,
  (6, 0): 1.5450470554958065,
  (6, 1): 2.085975303470599,
  (6, 2): 2.8786375139701814,
  (6, 3): 2.214463820494434,
  (6, 4): 2.170200639752333,
  (6, 5): 2.8764748933987243,
  (6, 6): 3.952022529944559,
  (6, 7): 5.163815059887708,
  (6, 8): 3.979671599176664,
  (6, 9): 3.0352534612311626,
  (7, 0): 2.0215730361382223,
  (7, 1): 2.8756313602409165,
  (7, 2): 4.244788716655939,
  (7, 3): 3.003053076063614,
  (7, 4): 2.9242141989102617,
  (7, 5): 3.9658832511450415,
  (7, 6): 5.50823400167582,
  (7, 7): 7.5932754110773635,
  (7, 8): 5.530373412930484,
  (7, 9): 4.098725957414136,
  (8, 0): 1.5736243664991862,
  (8, 1): 2.1243235071872872,
  (8, 2): 2.9317609955213606,
  (8, 3): 2.685964662513842,
  (8, 4): 3.618242010155426,
  (8, 5): 5.215214489200456,
  (8, 6): 7.620249247823317,
  (8, 7): 11.244788716655938,
  (8, 8): 7.636856477745916,
  (8, 9): 5.387481294019039,
  (9, 0): 1.2048434894472038,
  (9, 1): 1.5824618556740966,
  (9, 2): 2.123203455518395,
  (9, 3): 2.312051101615272,
  (9, 4): 3.1113457730799974,
  (9, 5): 4.26431201277731,
  (9, 6): 5.816146948145204,
  (9, 7): 7.8573041906788035,
  (9, 8): 5.829330381676332,
  (9, 9): 4.399313700440841})
```

The optimal policy tells us which direction it is best to move in given the square the robot is currently on. For example, the policy dictionary tells us

  (6, 1): (0, 1),
  
  (6, 2): (1, 0),

which means that if the robot is at the (6,1) square, it should move down, and at the (6,2) square, it should move right. This makes sense as there is a positive reward at (7,2), so moving right when at (6,2), the robot aims to get this reward. At the (6,1) square, the robot needs to move down and right (in either order) to reach (7,2), however it is better to attempt to move down first, because this moves the robot further from the edge, so it is less likely to randomly be moved into the edge.

The horizon was set to 50 to balance precision and runtime, as this MDP is more computationally intensive. By calculating
```
val1 = valit.value_iteration(states = grid_states, actions = grid_actions, probs = grid_probs, rewards = grid_reward, horizon = 50, discount = 0.8)[1]
val2 = valit.value_iteration(states = grid_states, actions = grid_actions, probs = grid_probs, rewards = grid_reward, horizon = 51, discount = 0.8)[1]
max(np.array(list(val2.values())) - np.array(list(val1.values())))
```
there is less than 1.65e-07 difference between iterations after 50 iterations, so 50 iterations is enough to have high accuracy.
