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

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
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


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    visited = []
    predecessor = {}

    start_state = problem.getStartState()

    fringe = util.Stack()
    fringe.push(start_state)

    visited.append(start_state[0])

    while not fringe.isEmpty():
        curr = fringe.pop()
        visited.append(curr)

        if problem.isGoalState(curr):
            actions = []
            while curr != start_state:
                actions.append(predecessor[curr][1])
                curr = predecessor[curr][0]
            actions.reverse()
            return actions
        for successor in problem.getSuccessors(curr):
            if successor[0] in visited:
                print("Already visited:", successor[0])
                continue
            fringe.push(successor[0])
            predecessor[successor[0]] = (curr, successor[1])

    # check if the cost decreases before adding to visited
    return []


def breadthFirstSearch(problem: SearchProblem):
    
    """Search the shallowest nodes in the search tree first."""
    
    visited = []
    predecessor = {}

    start_state = problem.getStartState()

    fringe = util.Queue()
    fringe.push(start_state)

    visited.append(start_state[0])

    while not fringe.isEmpty():
        curr = fringe.pop()

        if problem.isGoalState(curr):
            actions = []
            while curr != start_state:
                actions.append(predecessor[curr][1])
                curr = predecessor[curr][0]
            actions.reverse()
            return actions
        for successor in problem.getSuccessors(curr):
            if successor[0] in visited:
                # print("Already visited:", successor[0])
                continue
            fringe.push(successor[0])
            visited.append(successor[0])
            predecessor[successor[0]] = (curr, successor[1])

    # check if the cost decreases before adding to visited
    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    
    visited = {}  # Dictionary to track lowest cost to reach each state
    predecessor = {}
    
    start_state = problem.getStartState()
    
    fringe = util.PriorityQueue()
    fringe.push(start_state, 0)
    
    visited[start_state] = 0
    
    while not fringe.isEmpty():
        curr = fringe.pop()
        
        if problem.isGoalState(curr):

            actions = []
            while curr != start_state:
                actions.append(predecessor[curr][1])
                curr = predecessor[curr][0]
            actions.reverse()
            return actions
        
        for successor in problem.getSuccessors(curr):
            state, action, cost = successor
            new_cost = visited[curr] + cost  # Total cost to reach this state
            
            if state not in visited or new_cost < visited[state]:
                visited[state] = new_cost
                predecessor[state] = (curr, action)
                fringe.update(state, new_cost)
    
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    predecessor = {}
    cost = {}
    pred_cost = {}
    
    start_state = problem.getStartState()
    
    fringe = util.PriorityQueue()
    fringe.push(start_state, heuristic(start_state, problem))
    
    cost[start_state] = 0
    pred_cost[start_state] = heuristic(start_state, problem)
    
    while not fringe.isEmpty():
        curr = fringe.pop()
        
        if problem.isGoalState(curr):

            actions = []
            while curr != start_state:
                actions.append(predecessor[curr][1])
                curr = predecessor[curr][0]
            actions.reverse()
            return actions
        
        for successor in problem.getSuccessors(curr):
            state, action, cost_ = successor
            new_pred_cost = cost[curr] + cost_ + heuristic(state, problem) # Total cost to reach this state
            
            if state not in cost or new_pred_cost < pred_cost[state] :
                
                cost[state] = cost[curr] + cost_
                pred_cost[state] = new_pred_cost
                predecessor[state] = (curr, action)
                fringe.update(state, new_pred_cost)
    
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
