# multi_agents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattan_distance
from game import Directions, Actions
from pacman import GhostRules
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        # Evaluates how good a particular action is for Pacman.
        # This function estimates the value of the resulting state 
        # after Pacman takes the given action.
        # It encourages Pacman to move closer to food while avoiding
        # ghosts, returning a higher score for safer and more rewarding
        # positions.
        successor_game_state = current_game_state.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position()
        new_food = successor_game_state.get_food()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [ghostState.scared_timer for ghostState in new_ghost_states]

        score = successor_game_state.get_score()

        food_list = new_food.as_list()
        if food_list:
            dists = [manhattan_distance(new_pos, f) for f in food_list]
            min_food_dist = min(dists)
            score += 10.0 / min_food_dist

        for ghost_state in new_ghost_states:
            ghost_pos = ghost_state.get_position()

            if ghost_pos:
                dist_to_ghost = manhattan_distance(new_pos, ghost_pos)
                if dist_to_ghost <= 1:
                    return -1000
                score -= 10.0 / dist_to_ghost

        return score

def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.get_score()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, eval_fn='score_evaluation_function', depth='2'):
        super().__init__()
        self.index = 0 # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth) 

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def get_action(self, game_state):
        # Main function that decides Pacman's best move.
        # It iterates through all legal actions for Pacman (agent 0),
        # generates the successor states, and evaluates them using minimax (minAction).
        # Returns the action that maximizes the expected evaluation score. ✡️
        maxScore = -float('inf')
        bestAction = None
        for moves in game_state.get_legal_actions(0):
            successor = game_state.generate_successor(0,moves)
            value = self.minAction(successor,1,1)
            if(value > maxScore or bestAction == None):
                maxScore,bestAction = value,moves
        return bestAction
            

    def maxAction(self, game_state, depth):
        # Represents Pacman's (MAX) turn.
        # If a terminal state is reached (win/lose) or the maximum search depth,
        # it returns the evaluation of the current game state.
        # Otherwise, it generates all successors and returns the highest value
        # among those returned by the ghosts' (minAction) moves.
        if game_state.is_win() or game_state.is_lose() or depth == self.depth * 2:
            return self.evaluation_function(game_state)

        maxScore = -float('inf')
        for action in game_state.get_legal_actions(0):
            successor = game_state.generate_successor(0, action)
            score = self.minAction(successor, 1, depth + 1)
            if score > maxScore:
                maxScore = score
        return maxScore

    def minAction(self, game_state, agent_index, depth):
        # Represents the ghosts’ (MIN) turn.
        # If a terminal state is reached or the maximum depth is hit, it evaluates the state.
        # For each legal ghost action, it generates a successor state.
        # If the next agent is Pacman, it calls maxAction and increases the depth.
        # If there are more ghosts, it continues recursively with minAction without increasing depth.
        # Returns the minimum score among all possible ghost actions,
        # since ghosts try to minimize Pacman’s advantage. 
        if game_state.is_win() or game_state.is_lose() or depth == self.depth * 2:
            return self.evaluation_function(game_state)

        minScore = float('inf')
        for action in game_state.get_legal_actions(agent_index):
            successor = game_state.generate_successor(agent_index, action)
            num_agents = game_state.get_num_agents()
            next_agent = (agent_index + 1) % num_agents

            if next_agent == 0:
                score = self.maxAction(successor, depth + 1)
            else:
                score = self.minAction(successor, next_agent, depth)

            if score < minScore:
                minScore = score
        return minScore


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        # Entry point: choose Pacman's action using minimax with alpha-beta pruning.
        # Initializes alpha and beta bounds and searches over all legal actions for Pacman (agent 0).
        # For each action, it generates the successor state and evaluates it from the ghosts' (MIN) side.
        # Tracks the best action by the highest returned value and updates alpha to enable pruning in deeper calls.
        # Returns the action that maximizes the evaluation at the root.
        alpha = -float('inf')
        beta = float('inf')
        maxScore = -float('inf')
        bestAction = None
        for action in game_state.get_legal_actions(0):
            successor = game_state.generate_successor(0, action)
            value = self.betaPruning(successor, 1, 1, alpha, beta)
            if bestAction is None or value > maxScore:
                maxScore, bestAction = value, action
            if maxScore > alpha:
                alpha = maxScore
        return bestAction
            

    def alfaPruning(self, game_state, depth, alfa, beta):
        # MAX node: Pacman's turn.
        # Terminal check: if win/lose or depth limit reached, return heuristic evaluation.
        # Otherwise, iterate over Pacman's legal actions, recurse into MIN, and keep the maximum score.
        # Update alpha with the best score found so far; if alpha exceeds beta, prune remaining branches.
        if game_state.is_win() or game_state.is_lose() or depth == self.depth * 2:
            return self.evaluation_function(game_state)

        maxScore = -float('inf')
        for action in game_state.get_legal_actions(0):
            successor = game_state.generate_successor(0, action)
            score = self.betaPruning(successor, 1, depth + 1, alfa, beta)
            if score > maxScore:
                maxScore = score
            if maxScore > alfa:
                alfa = maxScore
            if alfa > beta:
                return maxScore
        return maxScore

    def betaPruning(self, game_state, agent_index, depth, alfa, beta):
        # MIN node(s): ghosts' turns.
        # Terminal check: if win/lose or depth limit reached, return heuristic evaluation.
        # For each ghost action, generate the successor and move to the next agent.
        # Depth increases only when control returns to Pacman (next_agent == 0); between ghosts, depth stays the same.
        # Keep the minimum score across actions, update beta, and prune when alpha >= beta.
        if game_state.is_win() or game_state.is_lose() or depth == self.depth * 2:
            return self.evaluation_function(game_state)

        minScore = float('inf')
        num_agents = game_state.get_num_agents()
        for action in game_state.get_legal_actions(agent_index):
            successor = game_state.generate_successor(agent_index, action)
            next_agent = (agent_index + 1) % num_agents

            if next_agent == 0:
                score = self.alfaPruning(successor, depth + 1, alfa, beta)
            else:
                score = self.betaPruning(successor, next_agent, depth, alfa, beta)
    
            if score < minScore:
                minScore = score
            if minScore < beta:
                beta = minScore
            if alfa > beta:
                return minScore

        return minScore


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluation_function

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()

def better_evaluation_function(current_game_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()
    


# Abbreviation
better = better_evaluation_function
