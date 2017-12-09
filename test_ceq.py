# -*- coding: utf-8 -*-
"""
Created on Sat Dec 02 00:31:11 2017

@author: Yinuo Liu
"""

from soccer import World
import numpy as np
from ce_q import ceQplayer,ce
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

def create_state_comb(p_a_states, p_b_states):
    states = {}
    ball_pos = ['A', 'B']
    id_q = 0
    for b in ball_pos:
        for p_a in p_a_states:
            for p_b in p_b_states:
                if p_a != p_b:
                    states[b + str(p_a) + str(p_b)] = id_q
                    id_q += 1

    return states


rows = 2
cols = 4
num_states = rows * cols
total_states = create_state_comb(range(num_states), range(num_states))
player_a = ceQplayer(numStates=len(total_states),numActionsA=5,numActionsB=5,\
                   decay=0.9999954, expl=0.5, gamma=0.9,x=2, y=0, has_ball=False, p_id='A')
player_b = ceQplayer(numStates=len(total_states),numActionsA=5,numActionsB=5,\
                   decay=0.9999954, expl=0.5, gamma=0.9,x=1, y=0, has_ball=True, p_id='B')

world = World()
world.set_world_size(x=cols, y=rows)
world.place_player(player_a, player_id='A')
world.place_player(player_b, player_id='B')
world.set_goals(100, 0, 'A')
world.set_goals(100, 3, 'B')

state_s, cur_state = total_states[world.map_player_state()],total_states[world.map_player_state()]
A_Q = []
num_steps = 1000000
counter = 0

while counter < num_steps:
    goal = False
    player_a.update_state(2,0,False)
    player_b.update_state(1,0,True)
    cur_state = state_s
    while not goal:
        actionA = player_a.chooseAction(cur_state)
        actionB = player_b.chooseAction(cur_state)
        actions = {'A': actionA, 'B': actionB}
        new_state, rewards, goal = world.move(actions)
        new_state = total_states[new_state]
        player_a.update_Q(cur_state, new_state, [actionA, actionB], rewards['A'])
        player_b.update_Q(cur_state, new_state, [actionB, actionA], rewards['B'])

        Q1 = np.reshape(player_a.Q[cur_state],(25,1))
        Q2 = np.reshape(player_b.Q[cur_state].T,(25,1))
        Q = np.hstack([Q1,Q2])
        sol = np.array(ce(Q)['x'])
        player_a.V[cur_state] = np.sum(sol*Q1)
        player_b.V[cur_state] = np.sum(sol*Q2)
        cur_state = new_state
        if player_a.alpha > 0.001:
            player_a.alpha *= player_a.decay
            player_b.alpha *= player_b.decay
        A_Q.append(player_a.Q[state_s,1,4])
        counter += 1
        if counter > num_steps:
            break


#############################################
## find policy
############################################
Q1 = np.reshape(player_a.Q[state_s],(25,1))
Q2 = np.reshape(player_b.Q[state_s].T,(25,1))
Q = np.hstack([Q1,Q2])
sol = np.reshape(np.array(ce(Q)['x']),(5,5))
sol[sol < 1e-6] = 0
np.savetxt("ce_q_sol.csv", sol, delimiter=",")


#############################################
## plot
############################################
diff = [abs(t - s) for s, t in zip(A_Q, A_Q[1:])]
diff = np.array(diff)
num_iter = np.arange(0, num_steps,1)
indice = np.where(diff > 0)
plt.figure(figsize=(20,15))
plt.plot(num_iter[indice], diff[indice],linewidth=1,color='k')
plt.xlim(0,num_steps)
plt.ylim(0,0.5)
plt.ylabel('Q-value difference')
plt.xlabel('Number of Iteration')
plt.title('CE-Q')
plt.show()

