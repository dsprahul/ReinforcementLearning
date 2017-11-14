import os
os.environ['KERAS_BACKEND'] = 'theano'

import gym
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

    return int(np.argmax(action_proba))


def loss_fn(y_true, y_pred):

    pass


def simple_exploration():

    episodes = 2000
    max_episode_len = 500

    # Init env
    env = gym.make("CartPole-v0")
    agent = build_agent()
    reward = 0

    avg_reward_over_episodes = []
    for episode_num in range(episodes):

        current_state = env.reset()
        total_reward = 0
        for _ in range(max_episode_len):
            action = get_next_action(state_=current_state,
                                     agent=agent,
                                     reward=reward)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # How do we communicate reward with agent? TODO

            if reward > 0:
                y_true = np.array([0.0, 0.0])
                y_true[action] = 1.0
            else:
                y_true = np.array([1.0, 1.0])
                y_true[action] = 0.0

            agent.fit(x=current_state.reshape(1, 4, 1),
                      y=y_true.reshape(-1, 2),
                      verbose=False)

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
