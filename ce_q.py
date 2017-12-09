# -*- coding: utf-8 -*-
"""
Created on Fri Dec 01 12:23:18 2017

@author: Yinuo Liu
"""


import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
from player import Player


class ceQplayer(Player):
    def __init__(self, numStates, numActionsA, numActionsB, decay,\
                 expl, gamma,x, y, has_ball, p_id=None):
        super(ceQplayer, self).__init__(x, y, has_ball, p_id)
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
        action = np.random.randint(self.numActionsA)
        return action

    def update_Q(self, initialState, finalState, actions, reward):
        actionA, actionB = actions
        self.Q[initialState, actionA, actionB] = (1 - self.alpha) * \
            self.Q[initialState, actionA, actionB] + \
            self.alpha * (reward + self.gamma * self.V[finalState])


def ce(A, solver='conelp'):
    num_vars = len(A)
    # maximize matrix c
    c = [sum(i) for i in A] 
    c = np.array(c, dtype="float")
    c = matrix(c)
    c *= -1
    G = build_ce_constraints(A)
    G = np.vstack([G, np.eye(num_vars) * -1]) # > 0 constraint for all vars
    h_size = len(G)
    G = matrix(G)
    h = np.zeros(h_size)
    h = matrix(h)
    # contraints Ax = b
    A = np.ones((1, num_vars))
    A = matrix(A)
    b = matrix(np.ones(1))
    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver)
    return sol

def build_ce_constraints(A):
    num_vars = int(np.sqrt(len(A)))
    G = []
    # row player
    for i in range(num_vars): # action row i
        for j in range(num_vars): # action row j
            if i != j:
                constraints = [0 for a in A]
                base_idx = i * num_vars
                comp_idx = j * num_vars
                for k in range(num_vars):
                    constraints[base_idx+k] = (- A[base_idx+k][0]
                                               + A[comp_idx+k][0])
                G += [constraints]

    for i in range(num_vars): # action column i
        for j in range(num_vars): # action column j
            if i != j:
                constraints = [0 for a in A]
                for k in range(num_vars):
                    constraints[i + (k * num_vars)] = (
                        - A[i + (k * num_vars)][1] 
                        + A[j + (k * num_vars)][1])
                G += [constraints]
    return np.matrix(G, dtype="float")

