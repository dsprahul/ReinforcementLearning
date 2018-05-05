

# ## Observations
# * Looks like, every time step, `hero` has to move in one of known four directions.
# * If he passes through `fire`, a -1 but passes through a `goal`, a +1, other-wise a zero.
# * There is no `end` condition for this env, so the goal must be to maximise total reward.
#
# * Another thing, this env is dynamically populated. Meaning, the objects are randomly placed for each episode, so roting won't help the env.

# <br>

# # Visual Deep Q Agent

# ## Architecture (Vanilla)
#
#

# In[1]:

from keras.models import Model
from keras.layers import (
    Conv2D,
    Dense,
    MaxPooling2D,
    Flatten,
    Input
)


# In[2]:

import random
import numpy as np
from gridworld import gameEnv


# In[3]:

# Architecture parameters
kernel_size = (9, 9)
pool_stride = (2, 2)
conv_activation = "tanh"
dense_activation = "relu"
output_activation = "linear"
lr = 0.1
optimiser = "adam"  # (default)


# In[4]:

class GridEnvDQNAgent(object):
    '''
    An agent which exposes `learn` and `demo` methods for learning GridEnv env
    and running a demo what it has learnt.
    '''

    def __init__(self, exploration_prob=1.0, exploration_prob_min=0.1, max_episodes=10000, max_episode_length=500, batch_size=20):
        '''
        Tunable parameters (partial list)
        Apart from these, there are optimiser types, learning rates, decay rates;
        model architecture parameters; reward disgestion mechanism as variables.
        '''
        self.exploration_prob = exploration_prob
        self.exploration_prob_min = exploration_prob_min
        self.max_episodes = max_episodes
        self.max_episode_length = max_episode_length
        self.batch_size = batch_size
        self.agent = self.build_DQN(input_dims=(84, 84, 3), number_of_actions=4)
        self.env = gameEnv(partial=False, size=5)
        # Set this to any other value to train agent with most recent few experiences P .
        self.pick_policy = "!random"
        self.experiences = []
        self.rewards = []

    def build_DQN(self, input_dims, number_of_actions):
        '''
        A Deep network which takes state as input & outputs action weights.
        Essentially this learns the Q table for solving CartPole, hence the name Deep-Q-Network.
        '''
        self.input_dims = input_dims
        input_ = Input(shape=input_dims)

        c1 = Conv2D(activation=conv_activation, filters=32,
                    kernel_size=kernel_size, padding="SAME")(input_)
        c1 = MaxPooling2D(strides=pool_stride, pool_size=pool_stride)(c1)

        c2 = Conv2D(activation=conv_activation, filters=128,
                    kernel_size=kernel_size, padding="SAME")(c1)
        c2 = MaxPooling2D(strides=pool_stride, pool_size=pool_stride)(c2)

        c3 = Conv2D(activation=conv_activation, filters=256,
                    kernel_size=kernel_size, padding="SAME")(c2)
        c3 = MaxPooling2D(strides=pool_stride, pool_size=pool_stride)(c3)

        flattened = Flatten()(c3)

        d1 = Dense(1024, activation=dense_activation)(flattened)
        d2 = Dense(512, activation=dense_activation)(d1)
        output_ = Dense(number_of_actions, activation=output_activation)(d2)

        agent_model = Model(inputs=[input_], outputs=[output_])

        agent_model.compile(
            optimizer=optimiser,
            loss="mse",
            metrics=["accuracy"]
        )

        print(agent_model.summary())

        return agent_model

    def get_greedy_action(self, state):
        '''
        Given a state, estimates action probability distribution using DQN (agent).
        Then it would return most valuable action with a probability of `exploration_prob`
        '''
        if random.random() <= self.exploration_prob:
            action = random.choice([0, 1, 2, 3])
        else:
            actions_pd = self.agent.predict(state.reshape(self.reshape_dims))
            action = np.argmax(actions_pd)

        return action

    def get_training_data(self, pick_policy):
        '''
        When invoked, returns at most self.batch_size number of experiences
        from self.experiences
        '''
        mini_batch_size = min(self.batch_size, len(self.experiences))
        if pick_policy == "random":
            # Pick randomly
            mini_batch = random.sample(
                population=self.experiences,
                k=mini_batch_size
            )
        else:
            # Pick last few
            mini_batch = self.experiences[-mini_batch_size:]

        trainX, trainY = [], []
        for x, y in mini_batch:
            trainX.append(x)
            trainY.append(y)

        return np.array(trainX), np.array(trainY)

    def decay_exploration_prob(self):
        '''
        Slowly decays the exploration probabilty with runtime
        '''
        self.exploration_prob -= 1. / self.max_episodes
        self.exploration_prob = max(self.exploration_prob, self.exploration_prob_min)

    def learn(self):
        '''
        Initialises env, plays & updates DQN to the optimal Q
        '''
        # Play a maximum of `max_episodes` number of games
        for game in range(1, self.max_episodes + 1):

            self.experiences = []
            current_state = self.env.reset()
            accum_reward = 0

            # Stock experiences into buffer of length max number of time units a game should sustain
            for time_unit in range(self.max_episode_length):

                # Take an action (exploratorily) and see how it rewards now
                action = self.get_greedy_action(state=current_state)
                next_state, reward, done = self.env.step(action)

                # Now update the action_pd responsible for above action by including
                # current reward and value we'd achieve by following optimal policy
                # from now on, given we took above action
                self.reshape_dims = tuple([-1]) + self.input_dims
                target_action_pd = self.agent.predict(current_state.reshape(self.reshape_dims))[0]
                target_action_pd[action] = reward + 0.99 * \
                    np.max(self.agent.predict(next_state.reshape(self.reshape_dims))[0])
                if done is True:  # In this env, done is never true.
                    target_action_pd[action] = -1

                # Stock the experiences
                self.experiences.append([current_state.tolist(), target_action_pd.tolist()])
                current_state = next_state
                accum_reward += reward

                # Check if it's time to update policy (train DQN)
                if time_unit % self.batch_size == 0 or done is True:
                    trainX, trainY = self.get_training_data(pick_policy=self.pick_policy)
                    self.agent.train_on_batch(trainX, trainY)

                # Exit if game's over
                if done is True:  # In this env, done is never true.
                    break

            self.rewards.append(accum_reward)
            self.decay_exploration_prob()

            stat_update_freq = 8
            if game % stat_update_freq == 0:
                avg_reward = np.array(self.rewards[-2:]).mean()
                print("Avg. reward over last {0:d} is {1:3.2f} | Last played game#{2:d} | e={3:1.4f}".format(
                    stat_update_freq, avg_reward, game, self.exploration_prob
                ))
                if avg_reward > 195.0:
                    print("Done solving... ")
                    print("""(CartPole-v0 defines "solving" as getting average reward
                    of 195.0 over 100 consecutive trials.)""")
                    break

    def demo(self):
        '''
        Run this to render the performance of trained model.
        Be sure to first train the model.
        '''
        total_reward = 0
        state = self.env.reset()

        done = False
        while not done:
            action = np.argmax(self.agent.predict(state.reshape(-1, 4))[0])
            state, reward, done = self.env.step(action)
            self.env.render()


# In[5]:

# RL Hyperparameters
exploration_prob = 1.0
exploration_prob_min = 0.1
max_episodes = 10000
max_episode_length = 50
batch_size = 50


# In[6]:

# Instantiate GridEnvDQNAgent object.
# This object should be trained afresh everytime a notebook is restarted
# Consider saving weights of trained agent to make demo work out of box
DQNAgent = GridEnvDQNAgent(
    max_episodes=max_episodes,
    exploration_prob_min=exploration_prob_min,
    batch_size=batch_size,
    max_episode_length=max_episode_length
)


# In[ ]:

# Explore, sync reward and learn
DQNAgent.learn()


# In[ ]:
