# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem): #! QUESTAO 1 - DFS
    """
    Search the deepest nodes in the search tree first (DFS).
    """
    from util import Stack
    stack = Stack()
    visited = set()

    start_state = problem.getStartState()
    stack.push((start_state, []))

    while not stack.isEmpty():
        state, path = stack.pop()

        if problem.isGoalState(state):
            return path

        if state not in visited:
            visited.add(state)
            for successor, action, _ in problem.expand(state):
                if successor not in visited:
                    stack.push((successor, path + [action]))

    return []


def breadthFirstSearch(problem): #! QUESTAO 2 - BFS
    """Search the shallowest nodes in the search tree first (BFS)."""
    from util import Queue

    queue = Queue()
    visited = set()

    start_state = problem.getStartState()
    queue.push((start_state, []))

    while not queue.isEmpty():
        state, path = queue.pop()

        if problem.isGoalState(state):
            return path

        if state not in visited:
            visited.add(state)
            for successor, action, _ in problem.expand(state):
                if successor not in visited:
                    queue.push((successor, path + [action]))

    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic): #! QUESTAO 3 - A*
    """Search the node that has the lowest combined cost and heuristic first (A*)."""
    from util import PriorityQueue

    frontier = PriorityQueue()
    visited = set()

    start_state = problem.getStartState()
    start_cost = 0
    start_heuristic = heuristic(start_state, problem)
    frontier.push((start_state, [], 0), start_cost + start_heuristic)

    while not frontier.isEmpty():
        state, path, cost = frontier.pop()

        if problem.isGoalState(state):
            return path

        if state not in visited:
            visited.add(state)
            for successor, action, stepCost in problem.expand(state):
                if successor not in visited:
                    new_cost = cost + stepCost
                    priority = new_cost + heuristic(successor, problem)
                    frontier.push((successor, path + [action], new_cost), priority)

    return []

def iterativeDeepeningSearch(problem): #! QUESTAO 9 - IDS
    from util import Stack
    depth = 0
    while True:
        stack = Stack()
        visited = set()
        start_state = problem.getStartState()
        stack.push((start_state, [], 0))  # (state, path, current_depth)

        found = None
        
        while not stack.isEmpty():
            state, path, current_depth = stack.pop()

            if problem.isGoalState(state):
                if found is None or len(path) < len(found):
                    found = path

            if current_depth < depth:
                if state not in visited:
                    visited.add(state)
                    for successor, action, _ in problem.expand(state):
                        if successor not in visited:
                            stack.push((successor, path + [action], current_depth + 1))

        if found is not None:
            return found
            
        depth += 1

        if depth > 1000:  # limite grande máx
            return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ids = iterativeDeepeningSearch
