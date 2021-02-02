
from datetime import datetime
import gym
from gym.utils.play import play
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from PolicyGradientAgent import *

def GetModelName(env_name):
    now = datetime.now()
    timeStr = now.strftime("%y%m%d_%H%M%S")

    return env_name + '_' + timeStr

def SaveModel(network, output_folder, modelName):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    torch.save(network.state_dict(), output_folder + '/' + modelName + '.pkl')

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epoch_num = 10000
    episode_num = 5
    now = datetime.now()

    env_name = 'LunarLander-v2'
    output_folder = 'results/PG_' + env_name + '_'+ now.strftime("%y%m%d_%H%M")
    logPath = output_folder + '/log'
    modelPath = output_folder + '/model'
    writer = SummaryWriter(logPath)

    env = gym.make(env_name)

    # Use the keyboard to play gym games
    #play(env, keys_to_action=get_keys_to_action(env))

    metadata = 'rgb_array' if 'rgb_array' in env.metadata['render.modes'] else 'human'
    observation_num = env.observation_space.shape[0]
    action_num = env.action_space.n
    agent = PolicyGradientAgent(observation_num, action_num, lr=0.001, device=device)
    #agent.network.load_state_dict(torch.load('results/PG_210202_1631/model/Model_210202_164109.pkl'))
    agent.network.train()

    # Train agent
    epoch_progress = tqdm(range(epoch_num))
    for epoch in epoch_progress:
        epoch_pg = torch.zeros(1, dtype=torch.float32, device=device)
        epoch_reward = 0

        for episode in range(episode_num):
            observation = env.reset()
            observation = torch.FloatTensor(observation).to(device)

            episode_reward = 0
            log_probs = torch.zeros(1, dtype=torch.float32, device=device)

            while True:
                #env.render()
                action, log_prob = agent.sample(observation)

                observation, reward, done, info = env.step(action)
                observation = torch.FloatTensor(observation).to(device)

                episode_reward += reward
                log_probs += log_prob

                if done:
                    epoch_pg += (log_probs * episode_reward)
                    epoch_reward += episode_reward
                    break

        if epoch > 0 and epoch % 50 == 0:
            modelName = GetModelName(env_name)
            SaveModel(agent.network, modelPath, modelName)

        avg_epoch_pg = epoch_pg / episode_num
        avg_epoch_reward = epoch_reward / episode_num

        agent.learn(epoch_pg)

        writer.add_scalar(logPath + '/Avg-Epoch-PG', -avg_epoch_pg.item(), epoch)
        writer.add_scalar(logPath + '/Avg-Epoch-Reward', avg_epoch_reward, epoch)

        epoch_progress.set_description(f"Avg-Epoch-PG: {-avg_epoch_pg.item(): 4.1f}, Avg-Epoch-Reward: {avg_epoch_reward: 4.1f}")

    # Test agent
    agent.network.eval()

    observation = env.reset()
    observation = torch.FloatTensor(observation).to(device)

    episode_reward = 0

    while True:
        env.render()
        action, _ = agent.sample(observation)

        observation, reward, done, info = env.step(action)
        observation = torch.FloatTensor(observation).to(device)

        episode_reward += reward

        if done:
            print('Episode Reward: ' + str(episode_reward))
            break

    env.close()

    pass
