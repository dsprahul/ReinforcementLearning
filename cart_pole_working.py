import os
os.environ['KERAS_BACKEND'] = 'theano'

import gym
import random
import numpy as np
from keras.layers import Dense, Input, Flatten, Activation
from keras.models import Model
from keras.models import Sequential


epsilon = 1.0


def build_neural_agent_(input_dim):

    input_ = Input(shape=(4, 1))
    # reward = Input(shape=(1,))
    x = Dense(4, activation="relu")(input_)
    x = Dense(16, activation="relu")(x)
    x = Dense(8, activation="relu")(x)
    x = Flatten()(x)
    output = Dense(2, activation="linear")(x)

    agent = Model(inputs=input_, outputs=output)
    agent.compile(
        optimizer="adam",
        loss="mse",
        metrics=["accuracy"]
    )

    print agent.summary()
    return agent


def build_neural_agent(n_inputs, n_outputs):
    model = Sequential([
        Dense(8, batch_input_shape=(None, n_inputs)),
        Activation('relu'),
        Dense(16),
        Activation('relu'),
        Dense(n_outputs),
        Activation('linear')
    ])
    model.compile('adam', loss='mse')
    # if isfile(WEIGHT_FILE):
    #     print "[+] Loaded weights from file"
    #     model.load_weights(WEIGHT_FILE)

    model.summary()
    return model


def build_agent():

    agent = build_neural_agent(n_inputs=4, n_outputs=2)

    return agent


def get_action_proba(agent, state_, reward=0):

    action_proba = agent.predict(state_)

    return action_proba


def get_next_action(agent, state_, reward, greedy=False):

    action_proba = get_action_proba(reward=reward,
                                    state_=state_.reshape(-1, 4),
                                    agent=agent)
    # TODO implement greedy action picking
    global epsilon
    if random.random() <= epsilon:
        action = random.choice([1, 0])
    else:
        action = int(np.argmax(action_proba))

    return action, action_proba


def sync_reward(reward, action, action_vect, next_state, agent, done):

    _, next_action_vect = get_next_action(
        agent=agent,
        state_=next_state,
        reward=reward
    )
    # print action_vect, action_vect.shape
    # action_vect[0, action] = action_vect[0, action] +\
    #     (0.99 * (reward + 0.95 * (np.max(next_action_vect))))\
    #     - (0.99 * action_vect[0, action])
    # print action_vect[0, int(action)], action_vect[0, int(not action)], action_vect, action
    action_vect[0, int(action)] = reward + 0.99 * np.max(next_action_vect)

    if done is True:
        action_vect[0, int(action)] = -1
    # action_vect[0, int(action)] = min(1.0, action_vect[0, int(action)])
    # action_vect[0, int(not action)] = 1.0 - action_vect[0, int(action)]
    # print action_vect, next_action_vect
    return action_vect


def simple_exploration():

    episodes = 10000
    max_episode_len = 300
    buffer_len = 15

    # Init env
    env = gym.make("CartPole-v0")
    agent = build_agent()
    reward = 0

    avg_reward_over_episodes = []
    for episode_num in range(episodes):

        if epsilon >= 0.1:
            global epsilon
            epsilon -= 1. / episodes

        current_state = env.reset()
        total_reward = 0
        buffer_a = []
        buffer_r = []
        for _ in range(max_episode_len):
            action, action_vect = get_next_action(state_=current_state,
                                                  agent=agent,
                                                  reward=reward)
            next_state, reward, done, _ = env.step(action)
            if episode_num > 9500:
                env.render()
            total_reward += reward

            # How do we communicate reward with agent? TODO
            y_true = sync_reward(
                reward=reward,
                action=action,
                action_vect=action_vect,
                next_state=next_state,
                agent=agent,
                done=done
            )

            buffer_a.append(current_state)
            buffer_r.append(y_true.reshape(-1, 2))
            if len(buffer_a) >= buffer_len or done is True:
                buffer_a, buffer_r = np.array(buffer_a), np.array(buffer_r)
                agent.train_on_batch(x=buffer_a.reshape(-1, 4),
                                     y=buffer_r.reshape(-1, 2))

                buffer_a, buffer_r = [], []

            if done is True:
                break

            current_state = next_state

        avg_reward_over_episodes.append(total_reward)

        if episode_num % 500 == 0:
            print("{0:5d} --> Average accumulated reward: {1:.2f}\t Reward for this episode: {2:3d}\t Epsilon: {3:.2f}".format(
                episode_num, np.array(avg_reward_over_episodes)[
                    -100:].mean(), int(total_reward), epsilon
            ))


if __name__ == "__main__":
    simple_exploration()
