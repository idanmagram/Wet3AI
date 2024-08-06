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
    
    U_final = None
    U_tag = deepcopy(U_init)
    delta = 0

    # ====== YOUR CODE: ======
    while delta < epsilon * (1 - mdp.gamma) / mdp.gamma:
        U_final = deepcopy(U_tag)
        delta = 0
        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                if mdp.board[r][c] == "WALL":
                    continue
                if (r, c) in mdp.terminal_states:
                    U_final[r][c] = mdp.get_reward((r, c))
                    continue
                utilities_actions = []
                for action in mdp.actions:
                    sum_uti = 0
                    for optional_act in mdp.actions:
                        next_state = mdp.step((r, c), optional_act)
                        sum_uti += U_final[next_state[0]][next_state[1]] * mdp.transition_function[action][optional_act]
                    utilities_actions.append(sum_uti)
                U_tag[r][c] = mdp.get_reward((r, c)) + mdp.gamma * max(utilities_actions)
                if abs(U_tag[r][c] - U_final[r][c]) > delta:
                    delta = abs(U_tag[r][c] - U_final[r][c])
    return U_final
    # ========================


def get_policy(mdp: MDP, U: np.ndarray) -> np.ndarray:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #
    
    policy = None
    max_neighbor_utility = -float('-inf')
    max_action = 0
    # ====== YOUR CODE: ======
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            max_neighbor_utility = -float('-inf')
            for optional_act in mdp.actions:
                next_state = mdp.step((r, c), optional_act)
                if next_state != (r, c) and U[next_state[0]][next_state[1]] > max_neighbor_utility:
                    max_neighbor_utility = U[next_state[0]][next_state[1]]
                    max_action = optional_act

        policy[r][c] = max_action
    # ========================
    return policy


def policy_evaluation(mdp: MDP, policy: np.ndarray) -> np.ndarray:

    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    # ====== YOUR CODE: ======
    I = np.eye(mdp.num_col * mdp.num_row)
    U = np.linalg.inv(I - mdp.gamma @ policy) @ mdp.board
    return U
    # ========================


def policy_iteration(mdp: MDP, policy_init: np.ndarray) -> np.ndarray:

    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    optimal_policy = None
    # TODO:
    # ====== YOUR CODE: ======
    raise NotImplementedError
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
    

    transition_probs = None
    reward_matrix = None
    # TODO
    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
    return reward_matrix, transition_probs 
