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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        allFoodDistance = [manhattanDistance(newPos, food) for food in newFood.asList()]
        allGhostDistance = [manhattanDistance(newPos, ghost.configuration.pos) for ghost in newGhostStates]

        # Calculate distance to nearest food pellet
        if len(allFoodDistance) != 0:
            nearestFood = min(allFoodDistance)
        else:
            return 10000

        # Calculate distance to nearest ghost
        if len(allGhostDistance) != 0:
            nearestGhost = min(allGhostDistance)
        else:
            nearestGhost = float("inf")

        # Score based on scared ghosts
        for ghost in newGhostStates:
            if ghost.configuration.pos == nearestGhost and ghost.scaredTimer > 0:
                nearestGhost = 99999
        
        # Combine scores with appropriate weights
        combinedScore = childGameState.getScore() + nearestGhost / (nearestFood * 10)

        if action == 'Stop':
            combinedScore -= 20

        return combinedScore

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        result = self.MinMax(gameState, 0, 0)
        return result[1]

    def MinMax(self, gameState, depth, agent=None):
        # Get legal actions for the current agent
        legalActions = gameState.getLegalActions(0)

        # Check terminal conditions
        if len(legalActions) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth: 
            # Return the evaluation value and no action
            return self.evaluationFunction(gameState), None
        
        if agent == 0:
            # If it's Pacman's turn, call the max function
            return self.max(gameState=gameState, depth=depth)
        
        else:
            # If it's a ghost's turn, call the min function
            return self.min(gameState=gameState, depth=depth, agent=agent)

    # Max Function (Pacman)
    def max(self, gameState, depth):
        maxValue = float("-inf")
        optimalAction = None
        legalActions = gameState.getLegalActions(0)
        
        # Iterate over legal actions
        for action in legalActions:
            # Get the next state after taking the action
            nextState = gameState.getNextState(0, action)

            # Call MinMax recursively for the next state with depth increased
            value = self.MinMax(gameState=nextState, depth=depth, agent=1)[0]

            # Update maxValue and optimalAction if a better action is found
            if value > maxValue:
                maxValue, optimalAction = value, action

        return maxValue, optimalAction
    
    # Min Function (Ghosts)
    def min(self, gameState, depth, agent):
        minValue = float("inf")
        optimalAction = None
        legalActions = gameState.getLegalActions(agent)

        # Iterate over legal actions
        for action in legalActions:
            # Get the next state after taking the action
            if agent == gameState.getNumAgents() - 1:

                # If it's the last ghost, call MinMax with depth increased and agent set to 0
                nextState = gameState.getNextState(agent, action)
                value = self.MinMax(nextState, depth + 1, 0)[0]

            else:
                # If it's not the last ghost, call MinMax with depth unchanged and agent increased
                nextState = gameState.getNextState(agent, action)
                value = self.MinMax(nextState, depth, agent + 1)[0]
            
            # Update minValue and optimalAction if a better action is found
            if value < minValue:
                minValue, optimalAction = value, action

        return minValue, optimalAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        # Alpha, Beta
        result = self.MinMax(gameState=gameState, depth=0, agent=0, alpha=float("-inf"), beta=float("inf"))
        return result[1]

    def MinMax(self, gameState, depth, alpha, beta, agent=None):
        # Get legal actions for the current agent
        legalActions = gameState.getLegalActions(0)

        # Check terminal conditions
        if len(legalActions) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth: 
            # Return the evaluation value and no action
            return self.evaluationFunction(gameState), None
        
        if agent == 0:
            # If it's Pacman's turn, call the max function
            return self.max(gameState=gameState, depth=depth, alpha=alpha, beta=beta)
        
        else:
            # If it's a ghost's turn, call the min function
            return self.min(gameState=gameState, depth=depth, agent=agent, alpha=alpha, beta=beta)

    # Max Function (Pacman)
    def max(self, gameState, depth, alpha, beta):
        maxValue = float("-inf")
        optimalAction = None
        legalActions = gameState.getLegalActions(0)
        
        # Iterate over legal actions
        for action in legalActions:
            # Get the next state after taking the action
            nextState = gameState.getNextState(0, action)

            # Call MinMax recursively for the next state with depth increased
            value = self.MinMax(gameState=nextState, depth=depth, agent=1, alpha=alpha, beta=beta)[0]

            # Update maxValue and optimalAction if a better action is found
            if value > maxValue:
                maxValue, optimalAction = value, action
            
            alpha = max(alpha, maxValue)

            if beta <= alpha:
                break

        return maxValue, optimalAction
    
    # Min Function (Ghosts)
    def min(self, gameState, depth, agent, alpha, beta):
        minValue = float("inf")
        optimalAction = None
        legalActions = gameState.getLegalActions(agent)

        # Iterate over legal actions
        for action in legalActions:
            # Get the next state after taking the action
            if agent == gameState.getNumAgents() - 1:

                # If it's the last ghost, call MinMax with depth increased and agent set to 0
                nextState = gameState.getNextState(agent, action)
                value = self.MinMax(gameState=nextState, depth=depth + 1, agent=0, alpha=alpha, beta=beta)[0]

            else:
                # If it's not the last ghost, call MinMax with depth unchanged and agent increased
                nextState = gameState.getNextState(agent, action)
                value = self.MinMax(gameState=nextState, depth=depth, agent=agent+1, alpha=alpha, beta=beta)[0]

            # Update minValue and optimalAction if a better action is found
            if value < minValue:
                minValue, optimalAction = value, action
            
            beta = min(beta, minValue)

            if beta <= alpha:
                break

        return minValue, optimalAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, depth, agentIndex ):
            if agentIndex >= state.getNumAgents():
                agentIndex = 0
                depth += 1
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return max(expectimax(state.getNextState(agentIndex, action), depth, agentIndex + 1) for action in state.getLegalActions(agentIndex))
            else:
                return sum(expectimax(state.getNextState(agentIndex, action), depth , agentIndex + 1) for action in state.getLegalActions(agentIndex)) / len(state.getLegalActions(agentIndex))
            
        return max(gameState.getLegalActions(0), key = lambda x: expectimax(gameState.getNextState(0, x), 0, 1))

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()    

# Abbreviation
better = betterEvaluationFunction
