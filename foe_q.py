"""
Created on Thu Nov 30 14:59:59 2017

@author: Yinuo Liu
"""
import numpy as np
from player import Player
from cvxopt import matrix,solvers

class foeQplayer(Player):

    def __init__(self, numStates, numActionsA, numActionsB, decay,\
                 expl, gamma,x, y, has_ball, p_id=None):
        super(foeQplayer, self).__init__(x, y, has_ball, p_id)
        self.decay = decay
        self.expl = expl
        self.gamma = gamma
        self.alpha = 0.1
        self.V = np.ones(numStates)
        self.Q = np.ones((numStates, numActionsA, numActionsB))
        self.numStates = numStates
        self.numActionsA = numActionsA
        self.numActionsB = numActionsB

    def chooseAction(self, state):
        action = np.random.randint(self.numActionsA)
        return action


    def update_Q_and_V(self, initialState, finalState, actions, reward):
        actionA, actionB = actions
        self.Q[initialState, actionA, actionB] = (1 - self.alpha) * self.Q[initialState, actionA, actionB] + \
            self.alpha * (reward + self.gamma * self.V[finalState])
        self.V[initialState] = self.updateV(initialState)[0]

    def updateV(self, state):
        c = np.zeros(self.numActionsA + 1)
        c[0] = -1
        A_ub = np.ones((self.numActionsB, self.numActionsA + 1))
        A_ub[:, 1:] = -self.Q[state].T
        b_ub = np.zeros(self.numActionsB)
        A_eq = np.ones((1, self.numActionsA + 1))
        A_eq[0, 0] = 0
        x_min = np.zeros(self.numActionsA)
        G2 = np.hstack([np.zeros((5, 1)),-np.eye(5)])
        h = np.hstack([b_ub, -x_min])
        
        c = matrix(c)
        G = matrix(np.vstack([A_ub, G2]))
        h = matrix(h)
        A = matrix(A_eq)
        b = matrix(np.ones(1))
        sol = solvers.lp(c, G, h, A, b, solver='glpk')['x']
        return sol
