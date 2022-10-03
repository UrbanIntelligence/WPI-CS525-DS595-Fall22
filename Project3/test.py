"""

### NOTICE ###
You DO NOT need to upload this file

"""

import argparse
import numpy as np
from environment import Environment
import time
from gym.wrappers.monitoring import video_recorder
from tqdm import tqdm

seed = 11037

def parse():
    parser = argparse.ArgumentParser(description="DS595/CS525 RL Project3")
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def test(agent, env, total_episodes=30, record_video=False):
    rewards = []
    env.seed(seed)
    if record_video:
        vid = video_recorder.VideoRecorder(env=env.env, path="test_vid.mp4")
    start_time = time.time()
    for i in tqdm(range(total_episodes)):
        frames= 0
        state = env.reset()
        agent.init_game_setting()
        episode_reward = 0.0

        #playing one game
        #frames = [state]
        terminated, truncated = False, False
        while not terminated and not truncated:
            frames += 1
            action = agent.make_action(state, test=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            #frames.append(state)
            if record_video:
                vid.capture_frame()
            if terminated or truncated:
                ###############################################################################
                ''' May not need to show this part when testing. (Just to show if stop because of 
                Time-limit, i.e., infinite state-action loops)
                ''' 
                if truncated is True:
                    print("Truncated: ", truncated)
                print(f"Episode {i+1} reward: {episode_reward}")
                ###############################################################################
                break

            env.env.close()
        rewards.append(episode_reward)

    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))
    print('rewards',rewards)
    print('running time',time.time()-start_time)

def run(args):
    '''
    render_mode: - None    --> no render is computed. (good when testing on many episodes)
                 - 'human' --> The environment is continuously rendered (human consumption)

    record_video: (bool) whether you need to record video
    '''
    env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True, render_mode=None)
    from agent_dqn import Agent_DQN
    agent = Agent_DQN(env, args)
    test(agent, env, total_episodes=100, record_video=False)


if __name__ == '__main__':
    args = parse()
    run(args)
