import numpy as np
from collections import deque

import torch
import torch.optim as optim

from common import params
from common.wrappers import make_atari, wrap
from common.replay_buffer import ReplayBuffer
from common.network import DQN, soft_update
from common.utils import plot, get_delta
from matplotlib import pyplot as plt
import gym
from datetime import datetime

def train():

    # Set seed
    np.random.seed(0)
    torch.manual_seed(0)

    # Initialize environments, 5 for Pong, 5 for Breakouut
    breakout_env = gym.vector.AsyncVectorEnv([
        lambda: wrap(make_atari(params.breakout)),
        lambda: wrap(make_atari(params.breakout)),
        lambda: wrap(make_atari(params.breakout)),
        lambda: wrap(make_atari(params.breakout)),
        lambda: wrap(make_atari(params.breakout))        
    ])

    pong_env = gym.vector.AsyncVectorEnv([
        lambda: wrap(make_atari(params.pong)),
        lambda: wrap(make_atari(params.pong)),
        lambda: wrap(make_atari(params.pong)),
        lambda: wrap(make_atari(params.pong)),
        lambda: wrap(make_atari(params.pong))
    ])

    num_actions = 4

    # Train the agent using DQN    
    breakout_buffer = ReplayBuffer(num_actions=num_actions, memory_len=params.mixed_buffer_size)
    pong_buffer = ReplayBuffer(num_actions=num_actions, memory_len=params.mixed_buffer_size)
    dqn = DQN(num_actions=num_actions).cuda()
    dqn_target = DQN(num_actions=num_actions).cuda()
    optimizer = optim.Adam(dqn.parameters(), lr=params.learning_rate)
    huber = torch.nn.HuberLoss()

    losses = []
    returns = []
    returns_pong_10 = deque(maxlen=10)
    returns_pong_50 = deque(maxlen=50)
    returns_breakout_10 = deque(maxlen=10)
    returns_breakout_50 = deque(maxlen=50)

    state_breakout = breakout_env.reset()
    state_pong = pong_env.reset()

    timesteps = 0
    episode = 0
    episode_pong = 0
    episode_breakout = 0
    ret_breakout = [0, 0, 0, 0, 0]
    ret_pong = [0, 0, 0, 0, 0]
    breakout_last_printed = -1
    pong_last_printed = -1
    to_train = 1
    t1 = datetime.now()

    while(True):

        # Epsilon decay
        epsilon = max(params.epsilon_lb, params.epsilon_ub - timesteps/ params.epsilon_decay)

        # Epsilon-driven action policy
        a_breakout = []
        a_pong = []
        for i in range(10):
            if i < 5:
                if np.random.choice([0,1], p=[1-epsilon,epsilon]) == 1:
                    a_breakout.append(np.random.randint(low=0, high=num_actions, size=1)[0])
                else:
                    net_out = dqn(state_breakout[i]).detach().cpu().numpy()
                    a_breakout.append(np.argmax(net_out))
            else:
                if np.random.choice([0,1], p=[1-epsilon,epsilon]) == 1:
                    a_pong.append(np.random.randint(low=0, high=num_actions, size=1)[0])
                else:
                    net_out = dqn(state_pong[i-5]).detach().cpu().numpy()
                    a_pong.append(np.argmax(net_out))
            
        
        # Perform chosen action and get next state
        next_state_breakout, r_breakout, done_breakout, info_breakout = breakout_env.step(a_breakout)
        next_state_pong, r_pong, done_pong, info_pong = pong_env.step(a_pong)
                       

        # Store transition
        for i in range(5):
            breakout_buffer.add(state_breakout[i], a_breakout[i], r_breakout[i], next_state_breakout[i], done_breakout[i])
        for i in range(5):
            pong_buffer.add(state_pong[i], a_pong[i], r_pong[i], next_state_pong[i], done_pong[i])
        

        state_breakout = next_state_breakout
        state_pong = next_state_pong
        timesteps = timesteps + 1

        # Update policy using temporal difference
        if (breakout_buffer.length() > params.breakout_minibatch_size and breakout_buffer.length() > params.update_after) and (pong_buffer.length() > params.pong_minibatch_size and pong_buffer.length() > params.update_after):
            optimizer.zero_grad()
            # Sample a minibatch randomly
            for i in range(2):
                if to_train:
                    mb_states, mb_a, mb_reward, mb_next_states, mb_done = breakout_buffer.sample_batch(batch_size=params.breakout_minibatch_size)
                else:
                    mb_states, mb_a, mb_reward, mb_next_states, mb_done = pong_buffer.sample_batch(batch_size=params.pong_minibatch_size)
                to_train = not to_train
                q_states = dqn(mb_states)
                q_next_states = dqn_target(mb_next_states)
                # Compute the targets
                targets = mb_reward + params.gamma * torch.max(q_next_states, dim=1)[0] * (1 - mb_done)
                # Compute the predictions
                predictions = (q_states * mb_a).sum(dim=1)
                # Update loss
                loss = huber(predictions, targets)
                loss.backward(retain_graph=False)
                optimizer.step()
                losses.append(loss.item())

                # Update target network
                soft_update(dqn, dqn_target, params.tau)
        
        # Update returns
        for i in range(5):
            if done_breakout[i] == True:
                returns.append(ret_breakout[i])
                returns_breakout_10.append(ret_breakout[i])
                returns_breakout_50.append(ret_breakout[i])
                ret_breakout[i] = 0
                episode+=1
                episode_breakout +=1
            else:
                ret_breakout[i] = ret_breakout[i] + int(r_breakout[i])
        
        for i in range(5):
            if done_pong[i] == True:
                returns.append(ret_pong[i])
                returns_pong_10.append(ret_pong[i])
                returns_pong_50.append(ret_pong[i])
                ret_pong[i] = 0
                episode+=1
                episode_pong += 1
            else:
                ret_pong[i] = ret_pong[i] + int(r_pong[i]) 

        # Backup
        if (episode_breakout % 100 == 0 or episode_pong % 10 == 0) and episode_pong != pong_last_printed and episode_breakout != breakout_last_printed and episode_breakout != 0 and episode_pong != 0:
            t1, h, m, s = get_delta(t1)
            print(f"Episode: {episode}\t Time: {int(h):02}:{int(m):02}:{int(s):02}")
            print(f"Pong Episodes: {episode_pong}\t Pong: {list(returns_pong_10)}\t Mean: {np.mean(returns_pong_50):.2f}")
            print(f"Breakout Episodes: {episode_breakout}\t Breakout: {list(returns_breakout_10)}\t Mean: {np.mean(returns_breakout_50):.2f}\n")
            torch.save(dqn.state_dict(), 'mixed_checkpoint.pth')
            if episode_breakout % 100 == 0:
                breakout_last_printed = episode_breakout
            if episode_pong % 10 == 0:
                pong_last_printed = episode_pong
        
        # Check goal
        if episode > 50:
            if(np.mean(returns_pong_50) >= 18 and np.mean(returns_breakout_50) >= 25):
                torch.save(dqn.state_dict(), 'new_mixed_model.pth')
                plot(timesteps, returns, losses)
                return

# Test function to visualize the trained model play
def test(env_id, trained_model = "Mixed/mixed_model.pth", rendered = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize environment
    test_env = make_atari(env_id, rendered)
    test_env = wrap(test_env, False, True, True, True)
    num_actions = 4

    # Load trained model
    model = DQN(num_actions)
    model.load_state_dict(torch.load(trained_model))
    model = model.to(device)
    model.eval()

    # Play an episode
    state = test_env.reset()
    done = False
    reward = 0
    
    if env_id == params.breakout:
        no_rew = 0
        timestep = 0
        while not done and timestep < 10000:
            timestep += 1
            net_out = model(state).detach().cpu().numpy()
            a = np.argmax(net_out)
            next_state, r, done, info = test_env.step(a)
            if r == 0:
                no_rew += 1
            if no_rew > 100:
                next_state, r, done, info = test_env.step(1) # spawn ball if game does not restart automatically
                no_rew = 0
            state = next_state
            reward += r
        print(f"The game ended with score: {int(reward)}. Completition percentage: {int(reward/108*100)}%")

    else:
        while not done:
            net_out = model(state).detach().cpu().numpy()
            a = np.argmax(net_out)
            next_state, r, done, info = test_env.step(a)
            state = next_state
            reward += r
        if reward > 0:
            print(f"Agent won. The game ended with score: {21-int(reward)} - 21")
        else:
            print(f"Agent lost. The game ended with score:: 21 - {21+int(reward)}")


# Performance function to get results of 100 episodes
def performance(env_id, trained_model= "Mixed/mixed_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize environment
    test_env = make_atari(env_id, False)
    test_env = wrap(test_env, False, True, True, True)
    num_actions = 4

    # Load trained model
    model = DQN(num_actions)
    model.load_state_dict(torch.load(trained_model))
    model = model.to(device)
    model.eval()

    rewards = []
    if env_id == params.breakout:
        dones = 0
        long = 0
        hi_score = 0
        lo_score = 108
        
        for i in range(100):
            
            if i % 10 == 0 and i != 0:
                print(i)
            
            state = test_env.reset()
            done = False
            reward = 0
            no_rew = 0
            timestep = 0
            while not done and timestep < 10000:
                timestep += 1
                net_out = model(state).detach().cpu().numpy()
                a = np.argmax(net_out)
                next_state, r, done, info = test_env.step(a)
                if r == 0:
                    no_rew += 1
                if no_rew > 100:
                    next_state, r, done, info = test_env.step(1) # spawn ball if game does not restart automatically
                    no_rew = 0
                state = next_state
                reward += r
            
            if done:
                dones += 1
            else:
                long += 1
            if reward > hi_score:
                hi_score = reward
            if reward < lo_score:
                lo_score = reward
            rewards.append(reward)
        total = sum(rewards)
        print(f"Mean Score: {int(total/100)}\t Max: {hi_score}\t Min: {lo_score}\t Dead: {dones}\t Timeouts: {long}")

        episodes = list(np.arange(100))
        fig = plt.figure(figsize = (10, 5))
        plt.bar(episodes, rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Scores")
        plt.title("E3: Breakout Episodes Results")
        plt.show()

        d = dict.fromkeys(list(np.arange(109)), 0)
        for j in range(100):
            d[rewards[j]]+=1
        scores = list(d.keys())
        freqs = list(d.values())
        fig = plt.figure(figsize = (10, 5))
        plt.bar(scores, freqs)
        plt.xlabel("Scores")
        plt.ylabel("Frequencies")
        plt.title("E3: Breakout Scores Frequencies")
        plt.show()
    
    else:
        win = 0
        lose = 0
        hi_score = -21
        lo_score = 21
        
        for i in range(100):
            
            if i % 10 == 0 and i != 0:
                print(i)
            
            state = test_env.reset()
            done = False
            reward = 0

            while not done:
                net_out = model(state).detach().cpu().numpy()
                a = np.argmax(net_out)
                next_state, r, done, info = test_env.step(a)
                state = next_state
                reward += r
            if reward > 0:
                win += 1
            else:
                lose += 1
            if reward < lo_score:
                lo_score = reward
            if reward > hi_score:
                hi_score = reward
            rewards.append(reward)
        print(f"Wins: {win}\t Loses: {lose}\t Best: {hi_score}\t Worst: {lo_score}\t")

        episodes = list(np.arange(100))
        fig = plt.figure(figsize = (10, 5))
        plt.bar(episodes, rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Scores")
        plt.title("E3: Pong Episodes Results")
        plt.show()

        d = dict.fromkeys(list(np.arange(-21,22)), 0)
        for j in range(100):
            d[rewards[j]]+=1
        scores = list(d.keys())
        freqs = list(d.values())
        fig = plt.figure(figsize = (10, 5))
        plt.bar(scores, freqs)
        plt.xlabel("Scores")
        plt.ylabel("Frequencies")
        plt.title("E3: Pong Scores Frequencies")
        plt.show()


if __name__ == "__main__":
    #train()
    test(params.pong, "Mixed/mixed_model.pth")
    #test(params.breakout, "Mixed/mixed_model.pth")
    #performance(params.pong, "Mixed/mixed_model.pth")
    #performance(params.breakout, "Mixed/mixed_model.pth")