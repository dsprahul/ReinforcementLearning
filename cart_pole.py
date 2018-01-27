import os
os.environ['KERAS_BACKEND'] = 'theano'

import gym
import random
import numpy as np
from keras.layers import Dense, Input, Flatten
from keras.models import Model


def build_neural_agent(input_dim):

    input_ = Input(shape=(4, 1))
    # reward = Input(shape=(1,))
    x = Dense(4, activation="relu")(input_)
    x = Dense(16, activation="relu")(x)
    x = Dense(8, activation="relu")(x)
    x = Flatten()(x)
    output = Dense(2, activation="softmax")(x)

    agent = Model(inputs=input_, outputs=output)
    agent.compile(
        optimizer="adam",
        loss="mse",
        metrics=["accuracy"]
    )

    return agent


def build_agent():

    agent = build_neural_agent((1, 4, 1))

    return agent


def get_action_proba(agent, state_, reward=0):

    action_proba = agent.predict(state_)

    return action_proba


def get_next_action(agent, state_, reward, greedy=False):

    action_proba = get_action_proba(reward=reward,
                                    state_=state_.reshape(1, 4, 1),
                                    agent=agent)
    # TODO implement greedy action picking
    action = not int(np.argmax(action_proba)) if random.random(
    ) < 0.3 else int(np.argmax(action_proba))
    return action, action_proba


def sync_reward(reward, action, action_vect, next_state, agent):

    _, next_action_vect = get_next_action(
        agent=agent,
        state_=next_state,
        reward=reward
    )
    # print action_vect, action_vect.shape
    action_vect[0, action] = action_vect[0, action] +\
        (0.99 * (reward + 0.95 * (np.max(next_action_vect))))\
        - (0.99 * action_vect[0, action])

    return action_vect


def simple_exploration():

    episodes = 10000
    max_episode_len = 500
    buffer_len = 5

    # Init env
    env = gym.make("CartPole-v0")
    agent = build_agent()
    reward = 0

    avg_reward_over_episodes = []
    for episode_num in range(episodes):

        current_state = env.reset()
        total_reward = 0
        buffer_a = []
        buffer_r = []
        for _ in range(max_episode_len):
            action, action_vect = get_next_action(state_=current_state,
                                                  agent=agent,
                                                  reward=reward)
            next_state, reward, done, _ = env.step(action)
            # env.render()
            total_reward += reward

            # How do we communicate reward with agent? TODO
            y_true = sync_reward(
                reward=reward,
                action=action,
                action_vect=action_vect,
                next_state=next_state,
                agent=agent
            )

            buffer_a.append(current_state)
            buffer_r.append(y_true.reshape(-1, 2))
            if len(buffer_a) >= buffer_len or done is True:
                buffer_a, buffer_r = np.array(buffer_a), np.array(buffer_r)
                agent.fit(x=buffer_a.reshape(-1, 4, 1),
                          y=buffer_r.reshape(-1, 2),
                          verbose=False)

                buffer_a, buffer_r = [], []

            if done is True:
                break

            current_state = next_state

        avg_reward_over_episodes.append(total_reward)

        if episode_num % 100 == 0:
            print("Average accumulated reward {}".format(
                np.array(avg_reward_over_episodes).mean())
            )


if __name__ == "__main__":
    simple_exploration()
