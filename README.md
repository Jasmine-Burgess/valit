# Valit Package: Value Iteration

## Overview

This package implements value iteration for Markov Decision Processes (MDPs).

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
The value_iteration function can be used as follows to determine the best policy to maximise rewards over a long time period.
```
valit.value_iteration(states = states_example1, actions = actions_example1, probs = probs_example1, rewards = rewards_example1, discount = 0.8)
```
This is an infinite horizon problem, as there is no fixed endpoint where Sam will stop partying or relaxing, so the default horizon of 1000 is used. A discount of 0.8 assigns more importance to immediate rewards than future rewards. The code returns a policy of partying when healthy and relaxing when sick. It also returns the value function which is 35.71428571 for the healthy state and 23.8095238 for the sick state; the value function is higher for the healthy state as the rewards are higher there.

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
The optimal policy tells us which direction it is best to move in given the square the robot is currently on. For example, the policy dictionary tells us

  (6, 1): (0, 1),
  
  (6, 2): (1, 0),

which means that if the robot is at the (6,1) square, it should move down, and at the (6,2) square, it should move right. This makes sense as there is a positive reward at (7,2), so moving right when at (6,2), the robot aims to get this reward. At the (6,1) square, the robot needs to move down and right (in either order) to reach (7,2), however it is better to attempt to move down first, because this moves the robot further from the edge, so it is less likely to randomly be moved into the edge.

The horizon was set to 50 to balance precision and runtime, as the MDP is more computationally intensive. By calculating
```
val1 = valit.value_iteration(states = grid_states, actions = grid_actions, probs = grid_probs, rewards = grid_reward, horizon = 50, discount = 0.8)[1]
val2 = valit.value_iteration(states = grid_states, actions = grid_actions, probs = grid_probs, rewards = grid_reward, horizon = 51, discount = 0.8)[1]
max(np.array(list(val2.values())) - np.array(list(val1.values())))
```
there is less than 1.65e-07 difference between iterations after 50 iterations, so 50 iterations is enough to have high accuracy.
