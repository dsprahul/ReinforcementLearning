import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import gym
import random
import numpy as np
from keras import backend as K
from keras.layers import Dense, Input, Flatten
from keras.models import Model
from functools import partial


def discount_rewards(r, gamma=0.99):
    """
    Takes 1d float array of rewards and computes discounted reward
    e.g. f([1, 1, 1], 0.99) -> [1, 0.99, 0.9801]
    """

    # discounted_rewards = []
    # for i, reward_tensor in enumerate(K.tf.unstack(r)):
    #     discounted_rewards.append(K.tf.scalar_mul(gamma ** i, reward_tensor))

    # r -= K.mean(r)
    # r /= (K.std(r) + 10**-8)

    return r  # K.tf.stack(discounted_rewards)


def discount_rewards_actual(r, gamma=0.99):
    """Takes 1d float array of rewards and computes discounted reward
    e.g. f([1, 1, 1], 0.99) -> [1, 0.99, 0.9801]
    """
    return np.array([val * (gamma ** i) for i, val in enumerate(r)])


def loss_fn(y_true, y_pred, rewards):
    """ Log-likelihood error function """

    # advantages = discount_rewards(rewards)
    advantages = rewards

    log_lik = K.tf.log(
        y_true * (y_true - y_pred) +
        (1 - y_true) * (y_true + y_pred)
    )
    loss = -K.tf.reduce_mean(log_lik * advantages)

    return loss


def get_loss_fn(reward_):
    """ Creating partial fn to adhere loss_fn API """
    return partial(loss_fn, rewards=reward_)


def build_neural_agent(input_dim):

    input_ = Input(shape=(4, 1))
    reward_ = Input(shape=(1,))

    x = Dense(4, activation="relu")(input_)
    x = Dense(16, activation="relu")(x)
    x = Dense(8, activation="relu")(x)
    x = Flatten()(x)
    output = Dense(2, activation="softmax")(x)

    agent = Model(inputs=[input_, reward_], outputs=output)
    agent.compile(
        optimizer="sgd",
        loss=get_loss_fn(reward_=reward_),
        metrics=["accuracy"]
    )

    return agent


def build_agent():

    agent = build_neural_agent((1, 4, 1))

    return agent


def get_action_proba(agent, state_, reward=0):

    action_proba = agent.predict(x=[state_, np.array([reward])])

    return action_proba


def get_next_action(agent, state_, reward, greedy=False):

    action_proba = get_action_proba(reward=reward,
                                    state_=state_.reshape(1, 4, 1),
                                    agent=agent)
    # TODO implement greedy action picking
    action = (not int(np.argmax(action_proba))) if (
        random.random() < 0.3) else int(np.argmax(action_proba))

    return action


def get_target(reward, action):

    if reward > 0:
        y_true = np.array([0.0, 0.0])
        y_true[action] = 1.0
    else:
        y_true = np.array([1.0, 1.0])
        y_true[action] = 0.0

    return y_true


def simple_exploration():

    episodes = 10000
    max_episode_len = 500
    buffer_len = 20

    # Init env
    env = gym.make("CartPole-v0")
    agent = build_agent()
    reward = 0

    avg_reward_over_episodes = []
    for episode_num in range(episodes):

        current_state = env.reset()
        total_reward = 0
        buffer_a = []
        buffer_t = []
        buffer_r = []
        for _ in range(max_episode_len):
            action = get_next_action(state_=current_state,
                                     agent=agent,
                                     reward=reward)
            next_state, reward, done, _ = env.step(action)
            # env.render()
            total_reward += reward

            # How do we communicate reward with agent? TODO
            y_true = get_target(reward=reward, action=action)

            buffer_a.append(current_state)
            buffer_t.append(y_true.reshape(-1, 2))
            buffer_r.append(reward)

            if len(buffer_a) >= buffer_len or done is True:

                buffer_a = np.array(buffer_a)
                np.random.shuffle(buffer_a)
                buffer_t = np.array(buffer_t)
                np.random.shuffle(buffer_t)
                buffer_r = np.array(buffer_r)
                np.random.shuffle(buffer_r)

                buffer_r = discount_rewards_actual(buffer_r)

                print agent.train_on_batch(x=[
                    buffer_a.reshape(-1, 4, 1),
                    buffer_r.reshape(-1, 1)
                ],
                    y=buffer_t.reshape(-1, 2))

                # buffer_a, buffer_t, buffer_r = [], [], []

            if done is True:
                break

            current_state = next_state

        avg_reward_over_episodes.append(total_reward)

        # if episode_num % 100 == 0:
        #     print("Average accumulated reward {}".format(
        #         np.array(avg_reward_over_episodes)[-100:].mean())
        #     )


if __name__ == "__main__":
    simple_exploration()
