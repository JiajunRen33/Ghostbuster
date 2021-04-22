# bustersAgents.py
# ----------------
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


import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters

class NullGraphics:
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        noisyDistance = observation
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if noisyDistance != None and \
                    busters.getObservationProbability(noisyDistance, trueDistance) > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent:
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        try:
            inferenceType = util.lookup(inference, globals())
        except Exception:
            inferenceType = util.lookup('inference.' + inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        for index, inf in enumerate(self.inferenceModules):
            if not self.firstMove and self.elapseTimeEnable:
                inf.elapseTime(gameState)
            self.firstMove = False
            if self.observeEnable:
                inf.observe(gameState)
            self.ghostBeliefs[index] = inf.getBeliefDistribution()
        self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

from distanceCalculator import Distancer
from game import Actions
from game import Directions

class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closest to the closest ghost (according to mazeDistance!).
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        """
        Calculate the most likely position of each ghost - the lattice with the highest response index. 
        """
        min_distance_pos = None
        for positionDistribution in livingGhostPositionDistributions:
            max_key = None
            #max_key is the most likely location
            for key in positionDistribution.keys():
                if max_key == None or positionDistribution[key] > positionDistribution[max_key]:
                    max_key = key
            #calculate the distance between max_key and the current location, find the shortest distance
            if max_key != None:
                distance = self.distancer.getDistance(pacmanPosition, max_key)
                #inally min_distance_pos save the most likely ghost location and the shortest distance.
                if min_distance_pos == None or min_distance_pos[1] > distance:
                    min_distance_pos = [max_key, distance]
        max_position = min_distance_pos[0]
        #Calculate the transverse and longitudinal distances
        x = max_position[0] - pacmanPosition[0]
        y = max_position[1] - pacmanPosition[1]
        suitable = []
        #If the transverse distance is higher than the longitudinal distance, go east-west.
        if x > 0:
            if y > 0:
                if y > x:
                    suitable.append('North')
                    suitable.append('East')
                else:
                    suitable.append('East')
                    suitable.append('North')
            else:
                if -y > x:
                    suitable.append('South')
                    suitable.append('East')
                else:
                    suitable.append('East')
                    suitable.append('South')
        #If the longitudinal distance is higher than the transverse distance, go north-south.
        else:
            if y > 0:
                if y > -x:
                    suitable.append('North')
                    suitable.append('West')
                else:
                    suitable.append('West')
                    suitable.append('North')
            else:
                if -y > -x:
                    suitable.append('South')
                    suitable.append('West')
                else:
                    suitable.append('West')
                    suitable.append('South')
        #If there is an optimal direction in legal, go there
        if suitable[0] in legal:
            return suitable[0]
        #If not, go in the second best direction
        if suitable[1] in legal:
            return suitable[1]
        #If don't have a second best direction, go in any direction
        return legal[0]
    
