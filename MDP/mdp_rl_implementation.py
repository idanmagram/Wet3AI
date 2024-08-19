from mdp import Action, MDP
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np
from copy import deepcopy


def value_iteration(mdp: MDP, U_init: np.ndarray, epsilon: float=10 ** (-3)) -> np.ndarray:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the utility for each of the MDP's state obtained at the end of the algorithms' run.
    #
    
    U_final = deepcopy(U_init)
    U_tag = deepcopy(U_init)
    delta = float('inf')
    # ====== YOUR CODE: ======
    while delta >= epsilon * ((1 - mdp.gamma) / mdp.gamma):
        U_final = deepcopy(U_tag)
        delta = 0
        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                if mdp.board[r][c] == "WALL":
                    continue
                if (r, c) in mdp.terminal_states:
                    delta = max(delta, abs(float(mdp.board[r][c]) - U_final[r][c]))
                    U_tag[r][c] = float(mdp.board[r][c])
                    continue
                utilities_actions = []
                for action in mdp.actions:
                    sum_uti = 0
                    for i, optional_act in enumerate(mdp.actions):
                        next_state = mdp.step((r, c), optional_act)
                        sum_uti += U_final[next_state[0]][next_state[1]] * mdp.transition_function[action][i]
                    utilities_actions.append(sum_uti)
                U_tag[r][c] = float(mdp.get_reward((r, c))) + mdp.gamma * max(utilities_actions)
                delta = max(abs(U_tag[r][c] - U_final[r][c]), delta)
    return U_final
    # ========================



def get_policy(mdp: MDP, U: np.ndarray) -> np.ndarray:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    policy = deepcopy(U)
    max_neighbor_utility = -float('inf')
    max_action = 0
    # ====== YOUR CODE: ======
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            if mdp.board[r][c] == "WALL" or (r, c) in mdp.terminal_states:
                policy[r][c] = None
                continue

            utilities = [U[mdp.step((r, c), action)[0]][mdp.step((r, c), action)[1]] for action in mdp.actions.keys()]
            max_neighbor_utility = -float('inf')
            max_action = -1
            for action in mdp.actions.keys():
                probabilities = mdp.transition_function[action]
                if sum(np.multiply(utilities, probabilities)) > max_neighbor_utility:
                    max_neighbor_utility = sum(np.multiply(utilities, probabilities))
                    max_action = action
            policy[r][c] = max_action
    # ========================
    return policy

def state_to_cord(mdp: MDP, state):
    rows = mdp.num_row
    cols = mdp.num_col
    i = int(state / cols)
    j = int(state % cols)
    return i, j

def get_neighbors(mdp: MDP, x, y):
    neighbors = []
    for optional_act in mdp.actions.keys():
        next_state = mdp.step((x, y), optional_act)
        neighbors.append((optional_act, next_state))
    return neighbors


def policy_evaluation(mdp: MDP, policy: np.ndarray) -> np.ndarray:

    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    # ====== YOUR CODE: ======
    U_res = deepcopy(policy)
    flattened_board_array = [(item) for sublist in mdp.board for item in sublist]
    probs = np.zeros((len(flattened_board_array), len(flattened_board_array)), dtype=float)
    b = np.zeros((len(flattened_board_array), 1))

    for i in range(len(flattened_board_array)):
        if flattened_board_array[i] == 'WALL':
            b[i] = float(0)
        else:
            b[i] = float(flattened_board_array[i])

    for state in range(len(flattened_board_array)):
            x, y = state_to_cord(mdp, state)
            if mdp.board[x][y] == 'WALL' or (x, y) in mdp.terminal_states:
                continue
            neighbors = get_neighbors(mdp, x, y)

            for index, neighbor in enumerate(neighbors):
                #print("index is ", index)
                #print("states are ", neighbor[1][0]*mdp.num_col + neighbor[1][1])
                probs[state][neighbor[1][0]*mdp.num_col + neighbor[1][1]] += float(mdp.transition_function[Action(policy[x][y])][index])

    I = np.eye(mdp.num_col * mdp.num_row)
    a = np.array(I - float(mdp.gamma) * probs)
    U = np.linalg.solve(a, b)

    #U = np.linalg.inv(np.array(I - float(mdp.gamma) * probs)) @ np.array(flattened_board_array)
    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            if mdp.board[row][col] == 'WALL':
                U_res[row][col] = "None"
            else:
                U_res[row][col] = float(U[row*mdp.num_col + col])
    return U_res
    # ========================


def policy_iteration(mdp: MDP, policy_init: np.ndarray) -> np.ndarray:

    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    optimal_policy = deepcopy(policy_init)

    # ====== YOUR CODE: ======
    unchanged = False
    while not unchanged:
        unchanged = True
        U = policy_evaluation(mdp, optimal_policy)
        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                if mdp.board[r][c] == "WALL" or (r, c) in mdp.terminal_states:
                    optimal_policy[r][c] = "None"
                    continue
                utilities_actions = []
                for action in mdp.actions.keys():
                    sum_uti = 0
                    for i, optional_act in enumerate(mdp.actions):
                        if optional_act == Action(optimal_policy[r][c]):
                            optimal_policy_index = i
                        next_state = mdp.step((r, c), optional_act)
                        sum_uti += U[next_state[0]][next_state[1]] * mdp.transition_function[action][i]
                    utilities_actions.append((sum_uti, action))
                max_sum_uti, best_action = max(utilities_actions, key=lambda x: x[0])
                if max_sum_uti > utilities_actions[optimal_policy_index][0]:
                    optimal_policy[r][c] = best_action
                    unchanged = False

    # ========================
    return optimal_policy



def adp_algorithm(
    sim: Simulator, 
    num_episodes: int,
    num_rows: int = 3, 
    num_cols: int = 4, 
    actions: List[Action] = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT] 
) -> Tuple[np.ndarray, Dict[Action, Dict[Action, float]]]:
    """
    Runs the ADP algorithm given the simulator, the number of rows and columns in the grid, 
    the list of actions, and the number of episodes.

    :param sim: The simulator instance.
    :param num_rows: Number of rows in the grid (default is 3).
    :param num_cols: Number of columns in the grid (default is 4).
    :param actions: List of possible actions (default is [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]).
    :param num_episodes: Number of episodes to run the simulation (default is 10).
    :return: A tuple containing the reward matrix and the transition probabilities.
    
    NOTE: the transition probabilities should be represented as a dictionary of dictionaries, so that given a desired action (the first key),
    its nested dicionary will contain the condional probabilites of all the actions. 
    """

    reward_matrix = np.full((num_rows, num_cols), None)

    # Initialize transition probabilities and number of actions dictionaries
    transition_probs = {action: {a: 0.0 for a in actions} for action in actions}
    number_of_actions = {action: {a: 0 for a in actions} for action in actions}
    # ====== YOUR CODE: ======
    for episode_index, episode_gen in enumerate(sim.replay(num_episodes)):
        print(f"@@@@    episode {episode_index}   @@@@@")
        for step_index, step in enumerate(episode_gen):
            state, reward, action, actual_action = step
            print(f"Step {step_index}: state={state}, reward={reward}, action={action}, actual_action={actual_action}")

            if action == None and actual_action == None:
                reward_matrix[state[0]][state[1]] = reward
                continue
            reward_matrix[state[0]][state[1]] = reward
            number_of_actions[action][actual_action] += 1
            transition_probs[action][actual_action] = number_of_actions[action][actual_action] / sum((number_of_actions[action]).values())

    # ========================
    return reward_matrix, transition_probs 
