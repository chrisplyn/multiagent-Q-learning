import numpy as np
from player import Player

class QPlayer(Player):
    def __init__(self, numStates, numActions, decay, expl, gamma,x, y, has_ball,p_id=None):
        super(QPlayer, self).__init__(x, y, has_ball, p_id)
        self.decay = decay
        self.expl = expl
        self.gamma = gamma
        self.alpha = 0.1
        self.V = np.zeros(numStates)
        self.Q = np.zeros((numStates, numActions))
        self.numStates = numStates
        self.numActions = numActions

    def chooseAction(self, state):
        if np.random.rand() < self.expl:
            action = np.random.randint(self.numActions)
        else:
            action = np.argmax(self.Q[state])
        return action

    def update_Q_and_V(self, initialState, finalState, actions, reward):
        actionA, actionB = actions
        self.Q[initialState, actionA] = (1 - self.alpha) * self.Q[initialState, actionA] + \
            self.alpha * (reward + self.gamma * self.V[finalState])
        bestAction = np.argmax(self.Q[initialState])
        self.V[initialState] = self.Q[initialState, bestAction]
