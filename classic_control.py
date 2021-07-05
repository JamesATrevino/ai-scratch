import gym
import numpy as np
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from rl.agents import SARSAAgent
from rl.policy import EpsGreedyQPolicy

# Taken from an open-ai tutorial
env = gym.make('CartPole-v1')


def agent(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1, states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


model = agent(env.observation_space.shape[0], env.action_space.n)
policy = EpsGreedyQPolicy()

sarsa = SARSAAgent(model = model, policy = policy, nb_actions = env.action_space.n)
sarsa.compile(optimizer=Adam(), metrics = ['mse'])
#sarsa.fit(env, nb_steps = 100000, visualize = False, verbose = 1)
sarsa.load_weights('sarsa_weights.h5f')
scores = sarsa.test(env, nb_episodes = 100, visualize= False)
print('Average score over 100 test games:{}'.format(np.mean(scores.history['episode_reward'])))
#sarsa.save_weights('sarsa_weights.h5f', overwrite=True)
_ = sarsa.test(env, nb_episodes = 2, visualize= True)