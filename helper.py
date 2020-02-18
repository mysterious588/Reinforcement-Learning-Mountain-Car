import numpy as np

def get_discrete(state, env, disc_units):
    DISCRETE_OBS_SIZE = disc_units * len(env.observation_space.high)
    DISCRETE_WIN_SIZE = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBS_SIZE
    return tuple(((state - env.observation_space.low) / DISCRETE_WIN_SIZE).astype(np.int))