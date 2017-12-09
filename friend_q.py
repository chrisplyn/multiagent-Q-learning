# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:04:54 2017

@author: Yinuo Liu
"""

import numpy as np
from player import Player


class friendQPlayer(Player):

    def __init__(self, numStates, numActionsA, numActionsB, decay, expl, gamma, x, y, has_ball, p_id=None):
        super(friendQPlayer, self).__init__(x, y, has_ball, p_id)
        self.decay = decay
        self.expl = expl
        self.gamma = gamma
        self.alpha = 0.1
        self.V = np.zeros(numStates)
        self.Q = np.zeros((numStates, numActionsA, numActionsB))
        self.numStates = numStates
        self.numActionsA = numActionsA
        self.numActionsB = numActionsB

    def chooseAction(self, state):
        if np.random.rand() < self.expl:
            action = np.random.randint(self.numActionsA)
        else:
            action,_ = np.unravel_index(np.argmax(self.Q[state]),(self.numActionsA, self.numActionsB))
        return action


    def update_Q_and_V(self, initialState, finalState, actions, reward):
        actionA, actionB = actions
        self.Q[initialState, actionA, actionB] = (1 - self.alpha) * self.Q[initialState, actionA, actionB] + \
            self.alpha * (reward + self.gamma * self.V[finalState])
        bestAction_pair_ind = np.argmax(self.Q[initialState])
        bestActionA,bestActionB = np.unravel_index(bestAction_pair_ind,(self.numActionsA, self.numActionsB))
        self.V[initialState] = self.Q[initialState, bestActionA, bestActionB]
