# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:26:42 2017

@author: Yinuo Liu
"""

class Player(object):

    def __init__(self, x, y, has_ball, p_id=None):
        self.p_id = p_id
        self.x = x
        self.y = y
        self.has_ball = has_ball

    def update_state(self, x, y, has_ball):
        self.x = x
        self.y = y
        self.has_ball = has_ball

    def update_x(self, x):
        self.x = x

    def update_y(self, y):
        self.y = y

    def update_ball_pos(self, has_ball):
        self.has_ball = has_ball