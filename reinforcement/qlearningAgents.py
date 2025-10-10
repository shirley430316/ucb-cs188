# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.qvalues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qvalues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return 0.0
        else:
            best_value = -float("inf")

            for action in actions:
                if self.qvalues[(state, action)] > best_value:
                    best_value = self.qvalues[(state, action)]
            return best_value
        


    def computeActionFromQValues(self, state):
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return None
    
        best_value = -float("inf")
        best_actions = []  # Track all actions with best value
    
        for action in actions:
            qvalue = self.getQValue(state, action)
            if qvalue > best_value:
                best_value = qvalue
                best_actions = [action]
            elif qvalue == best_value:
                best_actions.append(action)
    
        # Randomly choose among best actions if there's a tie
        return random.choice(best_actions) if best_actions else None

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        actions = self.getLegalActions(state)
        qvalues = [(action, self.qvalues[(state, action)]) for action in actions]

        qvalues_sort = sorted(qvalues, key=lambda x: x[1])


        if len(actions) == 0:
            return None

        if flipCoin(self.epsilon):
            action = random.choice(actions)
        else:
            #action = random.choice([k[0] for k in qvalues_sort[-2:]])
            action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        next_state_value = self.computeValueFromQValues(nextState)
        self.qvalues[(state, action)] = (reward + next_state_value * self.discount) * self.alpha + self.qvalues[(state, action)] * (1-self.alpha)
        

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        qvalue = 0
        features = self.featExtractor.getFeatures(state, action)
        for feature in features.keys():
            if feature in self.weights:
                qvalue += self.weights[feature] * features[feature]
            else:
                self.weights[feature] = 0
        return qvalue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        pred_qvalue = self.getQValue(state, action)

        next_state_q = -float('inf')
        actions = self.getLegalActions(nextState)
        if len(actions) == 0:
            next_state_q = 0
        else:
            for A in actions:
              qvalue = self.getQValue(nextState, A)
              if qvalue > next_state_q:
                  next_state_q = qvalue

        real_qvalue = reward + self.discount * next_state_q

        features = self.featExtractor.getFeatures(state, action)

        for feature in features.keys():
            if feature not in self.weights:
                self.weights[feature] = 0
            self.weights[feature] += self.alpha * (real_qvalue - pred_qvalue) * features[feature]


        return
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            pass
