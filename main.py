import gym
import numpy as np

import helper

# Constants
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 3000
DISCRETE_UNITS = [20]
SHOW_EVERY = 100

# init gym's builtin car game
env = gym.make("MountainCar-v0")

DISCRETE_OBS_SIZE = DISCRETE_UNITS * len(env.observation_space.high)
DISCRETE_WIN_SIZE = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBS_SIZE

# q-table (states x actions)
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBS_SIZE + [env.action_space.n]))

for episode in range(EPISODES):

    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False
        
    disc = helper.get_discrete(env.reset(), env, DISCRETE_UNITS)
    done = False
    while not done:
        action = np.argmax(q_table[disc])
        
        new_state, reward, done, _ = env.step(action)
        new_disc = helper.get_discrete(new_state, env, DISCRETE_UNITS)
        
        # update the enviroment
        if render:
            env.render()
            
        if not done:
            max_future_q = np.max(q_table[new_disc])
            current_q = q_table[disc + (action, )]
            # Q-learning formula
            new_q = (1 - LEARNING_RATE ) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[disc + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            print("made it on: ", episode)
            q_table[disc + (action,)] = 0
            
        disc = new_disc

env.close()
