
import gym
from gym.utils.play import play
from tqdm import tqdm

from PolicyGradientAgent import *


def get_keys_to_action(env):
    if hasattr(env, 'get_keys_to_action'):
        return env.get_keys_to_action()
    elif hasattr(env.unwrapped, 'get_keys_to_action'):
        return env.unwrapped.get_keys_to_action()
    else:
        # Customize keyboard actions
        return {(): 0,
                (ord('d'),): 1,
                (ord('s'),): 2,
                (ord('a'), ): 3}


if __name__ == '__main__':
    # Initalization
    epoch_num = 2
    episode_num = 2

    env_name = 'LunarLander-v2'
    env = gym.make(env_name)

    # Use the keyboard to play gym games
    #play(env, keys_to_action=get_keys_to_action(env))

    metadata = 'rgb_array' if 'rgb_array' in env.metadata['render.modes'] else 'human'

    agent = PolicyGradientAgent(env.observation_space.shape[0], env.action_space.n, lr=0.001)

    # Train agent
    epoch_progress = tqdm(range(epoch_num))
    for epoch in epoch_progress:
        epoch_rewards, epoch_log_probs = [], []

        for episode in range(episode_num):
            observation = env.reset()
            episode_reward = 0

            while True:
                env.render()
                action, log_prob = agent.sample(observation)
                observation, reward, done, info = env.step(action)

                episode_reward += reward

                if done:
                    epoch_rewards.append(episode_reward)
                    break

    env.close()

    pass
