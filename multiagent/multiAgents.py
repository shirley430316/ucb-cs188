# multiAgents.py
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def __init__(self, index=0):
        super().__init__(index)

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # bestMoves = [legalMoves[index] for index in bestIndices]
        # if Directions.WEST in bestMoves:
        #     return Directions.WEST
        # elif Directions.SOUTH in bestMoves:
        #     return Directions.SOUTH
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        prevFood = currentGameState.getFood()
        if newPos in prevFood.asList():
            food_bonus = 50
        else:
            food_bonus = 0

        revisited_penalty = 0
        # for pos in self.visited:
        #     if pos == newPos:
        #         revisited_penalty -=20
        
        stop_penalty = 0
        if action == Directions.STOP:
            stop_penalty -= 50

        ghost_poses = [ghost.configuration.pos for ghost in newGhostStates]
        ghost_score = 0
        for ghost_pos in ghost_poses:
            if manhattanDistance(ghost_pos, newPos) == 1 or manhattanDistance(ghost_pos, newPos) == 0:
                ghost_score -= 1000
            elif manhattanDistance(ghost_pos, newPos) > 3:
                pass
            else:
                ghost_score -= 100 / manhattanDistance(ghost_pos, newPos)
        
        for food in newFood.asList():
            if isLocal(newPos, food, 2):
                food_bonus += 1
            if isLocal(newPos, food, 1):
                food_bonus += 7

        if food_bonus != 0 and food_bonus != 30:
            return food_bonus + ghost_score + revisited_penalty + stop_penalty
        
        for food in newFood.asList():
            food_bonus -= manhattanDistance(food, newPos) / 2
        return ghost_score + food_bonus + revisited_penalty + stop_penalty
    
def isLocal(pos1, pos2, region):
    if abs(pos1[0]-pos2[0]) < region and abs(pos1[1]-pos2[1]) < region:
        return True
    else:
        return False
def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(index: int, gameState: GameState, depth: int):
            num_agents = gameState.getNumAgents()


            if gameState.isWin() or gameState.isLose() or (depth == self.depth and index==0):
                #print("reach the depth")
                #print(f"return score {self.evaluationFunction(gameState)}")
                return self.evaluationFunction(gameState)
            
            if index == 0:
                legalActions = gameState.getLegalActions(index)
                newStates = [gameState.generateSuccessor(index, action) for action in legalActions]
                #print("try to compute min for 1 at depth", depth)
                scores = [minimax(1, newState, depth) for newState in newStates]
                #print("scores at max node:", scores)
                return max(scores)
            
            else:
                legalActions = gameState.getLegalActions(index)
                newStates = [gameState.generateSuccessor(index, action) for action in legalActions]

                nextAgent = index + 1
                if nextAgent == num_agents:
                    nextAgent = 0
                    depth += 1
                #print(f"try to compute min/max for {nextAgent} at depth", depth)
                scores = [minimax(nextAgent, newState, depth) for newState in newStates]
                #print(f"scores at min/max node for {nextAgent}:", scores)                
                
                return min(scores)
            
        bestAction = None
        bestScore = float('-inf')

        for action in gameState.getLegalActions(0):  # Pacman's legal actions
            successor = gameState.generateSuccessor(0, action)
            score = minimax(1, successor, 0)  # Start with Ghost 1, depth 0
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def minimax(agentIndex: int, gameState: GameState, depth: int, alpha: int, beta: int):
            num_agents = gameState.getNumAgents()
        
            # Terminal state or max depth reached
            if gameState.isWin() or gameState.isLose() or depth == self.depth * num_agents:
                return self.evaluationFunction(gameState)
        
            # Pacman's turn (maximizing player)
            if agentIndex == 0:
                bestValue = -float('inf')
                legalActions = gameState.getLegalActions(agentIndex)
            
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    nextAgent = (agentIndex + 1) % num_agents
                    # Depth only increases when we complete a full cycle (all agents)
                    newDepth = depth + 1
                
                    value = minimax(nextAgent, successor, newDepth, alpha, beta)
                    bestValue = max(bestValue, value)
                    alpha = max(alpha, bestValue)  # Update alpha FIRST
                
                    if bestValue > beta:  # Then check for pruning
                        return bestValue
                    
                return bestValue
        
            # Ghosts' turn (minimizing players)
            else:
                bestValue = float('inf')
                legalActions = gameState.getLegalActions(agentIndex)
                nextAgent = (agentIndex + 1) % num_agents
                newDepth = depth + 1
            
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = minimax(nextAgent, successor, newDepth, alpha, beta)
                    bestValue = min(bestValue, value)
                    beta = min(beta, bestValue)  # Update beta FIRST
                
                    if bestValue < alpha:  # Then check for pruning
                        return bestValue
                    
                return bestValue
    
        # Root call with proper alpha-beta pruning
        bestAction = None
        bestScore = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
    
        legalActions = gameState.getLegalActions(0)
    
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            # Start with agent 1 (first ghost), depth 1
            score = minimax(1, successor, 1, alpha, beta)

            if score > bestScore:
                bestScore = score
                bestAction = action

            # Update alpha at root level for pruning
            alpha = max(alpha, bestScore)
    
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(agentIndex: int, gameState: GameState, depth: int):
            num_agents = gameState.getNumAgents()
            
            # Terminal state check
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            # Depth check - when we reach the maximum depth
            if depth == self.depth:
                return self.evaluationFunction(gameState)
            
            # Pacman's turn (maximizing player)
            if agentIndex == 0:
                bestValue = -float('inf')
                legalActions = gameState.getLegalActions(agentIndex)
                
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    # Next agent is the first ghost
                    value = expectimax(1, successor, depth)
                    bestValue = max(bestValue, value)
                
                return bestValue
            
            # Ghosts' turn (expectation node)
            else:
                legalActions = gameState.getLegalActions(agentIndex)
                nextAgent = agentIndex + 1
                
                # If this is the last ghost, next is Pacman and we increment depth
                if nextAgent == num_agents:
                    nextAgent = 0
                    newDepth = depth + 1
                else:
                    newDepth = depth
                
                totalValue = 0
                numActions = len(legalActions)
                
                # If there are no legal actions, return evaluation
                if numActions == 0:
                    return self.evaluationFunction(gameState)
                
                # Calculate expected value over all ghost actions
                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    value = expectimax(nextAgent, successor, newDepth)
                    totalValue += value
                
                # Return average value (expectation)
                return totalValue / numActions
        
        # Root level - Pacman chooses the best action
        bestAction = None
        bestScore = -float('inf')
        
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = expectimax(1, successor, 0)  # Start with first ghost at depth 0
            
            if score > bestScore:
                bestScore = score
                bestAction = action
        
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: eval = score + ghost_penalty + food_distance_bonus + food_eating_bonus + 
                        capsule_distance_bonus + capsule_eating_bonus
    """
    "*** YOUR CODE HERE ***"

    score = currentGameState.getScore()
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    ghostScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    evaluation = score

    # Factor 1: Distance to the closest food
    if food:
        minFoodDistance = min(manhattanDistance(pacmanPos, foodPos) for foodPos in food)
        evaluation += 1.0 / (minFoodDistance + 1)  

    # Factor 2: Ghost Distance Penalty
    for ghostState, scaredTime in zip(ghostStates, ghostScaredTimes):
        ghostPos = ghostState.getPosition()
        ghostDistance = manhattanDistance(pacmanPos, ghostPos)

        if scaredTime > 0:
            # If ghosts are scared, get closer to them
            evaluation += 200 / (ghostDistance + 1)  
        else:
            # Rule No.1: avoid dying
            if ghostDistance == 0 or ghostDistance == 1:
                evaluation = -float('inf')
            if ghostDistance <= 2:
                evaluation -= 100

    # Factor 3: Food Eating Bonus
    evaluation -= 10 * len(food)  

    # Factor 4: Capsules Distance Bonus
    capsules = currentGameState.getCapsules()
    if capsules:
        minCapsuleDistance = min(manhattanDistance(pacmanPos, capsulePos) for capsulePos in capsules)
        evaluation += 10 / (minCapsuleDistance + 1)

    # Factor 5: Capsule Eating Bonus
    evaluation -= 50 * len(capsules)

    return evaluation

# Abbreviation
better = betterEvaluationFunction
