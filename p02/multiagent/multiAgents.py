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
import math

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
        # Collect legal moves and successor states
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
        closestDist = 100
        for food in newFood.asList():
            if util.manhattanDistance(food, newPos)< closestDist:
                closestDist = util.manhattanDistance(food, newPos)

        ghostDistance = -1
        for i in range(len(newGhostStates)):
            if util.manhattanDistance(newGhostStates[i].getPosition(), newPos) < 4:
                ghostDistance = util.manhattanDistance(newGhostStates[i].getPosition(), newPos)

        if ghostDistance == -1:
            if(len(newFood.asList()) < len(currentGameState.getFood().asList())):
                closestDist = 0.1
            return 2/closestDist
        else:  
            return 2/closestDist + -4 * (1/(ghostDistance+0.1))

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
        v, action = self.maxx(gameState, None, 0, 0)
        return action
        
    def maxx(self, gameState, action, depth, agentNum):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), action
        
        bestV = -math.inf
        bestAction = None
        for nextAction in gameState.getLegalActions(agentNum):
            nextGameState = gameState.generateSuccessor(0, nextAction)
            tempV, tempAction = self.minx(nextGameState, nextAction, depth, 1)
            if tempV > bestV:
                bestV = tempV
                bestAction = tempAction
        return bestV, bestAction
        
    def minx(self, gameState, action, depth, agentNum):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), action
        if agentNum+1 == gameState.getNumAgents():
            worstV = math.inf
            for nextAction in gameState.getLegalActions(agentNum):
                nextGameState = gameState.generateSuccessor(agentNum, nextAction)
                tempV, tempAction = self.maxx(nextGameState, nextAction, depth+1, 0)
                if tempV < worstV:
                    worstV = tempV
            return worstV, action
        else:
            worstV = math.inf
            for nextAction in gameState.getLegalActions(agentNum):
                nextGameState = gameState.generateSuccessor(agentNum, nextAction)
                tempV, tempAction = self.minx(nextGameState, nextAction, depth, agentNum+1)
                if tempV < worstV:
                    worstV = tempV
            return worstV, action

        
    


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        v, action = self.maxx(gameState, None, 0, 0 , float('-inf'), float("inf"))
        return action
        
    def maxx(self, gameState, action, depth, agentNum ,a,b):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), action
        
        bestV = -math.inf
        bestAction = None
        for nextAction in gameState.getLegalActions(agentNum):
            nextGameState = gameState.generateSuccessor(0, nextAction)
            tempV, tempAction = self.minx(nextGameState, nextAction, depth, 1,a,b)
            if tempV > bestV:
                bestV = tempV
                bestAction = tempAction
            a= max(a,bestV)
            if a>b:
                return bestV,bestAction
        return bestV, bestAction
        
    def minx(self, gameState, action, depth, agentNum,a,b):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), action
        if agentNum+1 == gameState.getNumAgents():
            worstV = math.inf
            for nextAction in gameState.getLegalActions(agentNum):
                nextGameState = gameState.generateSuccessor(agentNum, nextAction)
                tempV, tempAction = self.maxx(nextGameState, nextAction, depth+1, 0,a,b)
                if tempV < worstV:
                    worstV = tempV
                b= min(b,worstV)
                if a>b:
                    return worstV,action
                
            return worstV, action
        else:
            worstV = math.inf
            for nextAction in gameState.getLegalActions(agentNum):
                nextGameState = gameState.generateSuccessor(agentNum, nextAction)
                tempV, tempAction = self.minx(nextGameState, nextAction, depth, agentNum+1,a,b)
                if tempV < worstV:
                    worstV = tempV
                b= min(b,worstV)
                if a > b :
                    return worstV,action
            return worstV, action

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
        v,action = self.expectimax(gameState,None,0,0)
        return action
    def expectimax(self,gameState,action,depth,agentNum):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), None
        if agentNum== 0:
            return self.maxx(gameState,action,depth,agentNum)
        else:
            return self.evaluate(gameState,action,depth,agentNum)

    def maxx(self, gameState, action, depth, agentNum):
        bestV = -math.inf
        bestAction = None
        for nextAction in gameState.getLegalActions(agentNum):
            nextGameState = gameState.generateSuccessor(agentNum, nextAction)
            tempV, tempAction = self.expectimax(nextGameState, nextAction, depth,1)
            if tempV > bestV:
                bestV = tempV
                bestAction = nextAction
        return bestV, bestAction
    
    def evaluate(self,gameState,action,depth,agentNum):
        v=0
        for nextAction in gameState.getLegalActions(agentNum):
            succ = gameState.generateSuccessor(agentNum, nextAction)
            prob = 1/(len(gameState.getLegalActions(agentNum)))
            if agentNum == gameState.getNumAgents()-1:
                tempV=prob*(self.expectimax(succ,nextAction,depth+1,0)[0])
                v+=tempV
            else:
                tempV=prob*(self.expectimax(succ,nextAction,depth,agentNum+1)[0])
                v+=tempV
        return v, None

        
        
        

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    1) so my approach was to check the state and see all the foods and the ghost and the current situation
    pacman is in. its reward based, for the closest 4 foods it gets +10 for each food but also gets a minor penalty for
    all the food left.

    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return 1000+currentGameState.getScore()
    if currentGameState.isWin():
        return -1000
    
    pacPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    current_score=currentGameState.getScore()
    ghost_states = currentGameState.getGhostStates()

    food_Score =0
    for food in foods():
        closest_food = min(manhattanDistance(pacPosition,food))
# Abbreviation
better = betterEvaluationFunction
