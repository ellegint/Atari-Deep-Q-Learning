import numpy as np
from collections import deque

import torch
import torch.optim as optim

from common import params
from common.wrappers import make_atari, wrap_breakout
from common.replay_buffer import ReplayBuffer
from common.network import DQN, soft_update
from common.utils import plot, get_delta
from matplotlib import pyplot as plt
import gym
from datetime import datetime

def train(env_id):

    # Set seed
    np.random.seed(0)
    torch.manual_seed(0)

    # Initialize environments
    env = gym.vector.AsyncVectorEnv([
        lambda: wrap_breakout(make_atari(env_id)),
        lambda: wrap_breakout(make_atari(env_id)),
        lambda: wrap_breakout(make_atari(env_id)),
        lambda: wrap_breakout(make_atari(env_id)),
        lambda: wrap_breakout(make_atari(env_id)),
        lambda: wrap_breakout(make_atari(env_id)),
        lambda: wrap_breakout(make_atari(env_id)),
        lambda: wrap_breakout(make_atari(env_id)),
        lambda: wrap_breakout(make_atari(env_id)),
        lambda: wrap_breakout(make_atari(env_id)),

    ])

    num_actions = env.action_space[0].n

    
    backup_frequency = 50
    minibatch_size = params.breakout_minibatch_size
    buffer_size = params.breakout_buffer_size
    
    # Train the agent using DQN
    buffer = ReplayBuffer(num_actions=num_actions, memory_len=buffer_size) 
    dqn = DQN(num_actions=num_actions, input_size=80).cuda()
    dqn_target = DQN(num_actions=num_actions, input_size=80).cuda()
    optimizer = optim.Adam(dqn.parameters(), lr=params.learning_rate)
    huber = torch.nn.HuberLoss()

    losses = []
    returns = []
    returns_10 = deque(maxlen=10)
    returns_50 = deque(maxlen=50)
    ret = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    timesteps = 0
    episode = 0
    last_printed = -1

    start = datetime.now()
    t1 = datetime.now()

    state = env.reset()
    while(True):

        # Epsilon decay
        epsilon = max(params.epsilon_lb, params.epsilon_ub - timesteps/ params.epsilon_decay)

        # Epsilon-driven action policy
        a = []
        for i in range(10):
            if np.random.choice([0,1], p=[1-epsilon,epsilon]) == 1:
                a.append(np.random.randint(low=0, high=num_actions, size=1)[0])
            else:
                net_out = dqn(state[i]).detach().cpu().numpy()
                a.append(np.argmax(net_out))
        
        # Perform chosen action and get next state
        next_state, r, done, info = env.step(a)

        # Store transition
        for i in range(10):
            buffer.add(state[i], a[i], r[i], next_state[i], done[i])

        state = next_state
        timesteps = timesteps + 1        

        # Update policy using temporal difference
        if buffer.length() > minibatch_size and buffer.length() > params.update_after:
            optimizer.zero_grad()
            # Sample a minibatch randomly
            mb_states, mb_a, mb_reward, mb_next_states, mb_done = buffer.sample_batch(batch_size=minibatch_size)
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
        for i in range(10):
            ret[i] += int(r[i])
            if done[i] == True:
                returns.append(ret[i])
                returns_10.append(ret[i])
                returns_50.append(ret[i])
                ret[i] = 0
                episode+=1

        # Backup
        if episode % backup_frequency == 0 and episode != last_printed and episode!=0:
            t1, h, m, s = get_delta(t1)
            print(f"Episode: {episode}\t Timesteps: {timesteps}\t Rewards: {list(returns_10)}\t Mean: {np.mean(returns_50):.2f}\t Time: {int(h):02}:{int(m):02}:{int(s):02}")
            torch.save(dqn.state_dict(), 'breakout_checkpoint.pth')
            last_printed = episode
        
        # Check goal
        if np.mean(returns_50) >= 25:
            torch.save(dqn.state_dict(),'new_breakout_model.pth')
            end, h, m, s = get_delta(start)
            print(f"End of computation. Episodes played: {episode}\t Final Mean: {np.mean(returns_50):.2f}\t Total Time: {int(h):02}:{int(m):02}:{int(s):02}")
            plot(timesteps, returns, losses)
            return


# Test function to visualize the trained model play
def test(env_id, trained_model = "Breakout/breakout_model.pth", rendered = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize environment
    test_env = make_atari(env_id, rendered)
    test_env = wrap_breakout(test_env, False, True, True, True)
    num_actions = 4

    # Load trained model
    model = DQN(num_actions, input_size=80)
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
def performance(env_id, trained_model= "Breakout/breakout_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize environment
    test_env = make_atari(env_id, False)
    test_env = wrap_breakout(test_env, False, True, True, True)
    num_actions = 4

    # Load trained model
    model = DQN(num_actions, input_size=80)
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
        plt.title("E2: Breakout Episodes Results")
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
        plt.title("E2: Breakout Scores Frequencies")
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
        plt.title("E2: Pong Episodes Results")
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
        plt.title("E2: Pong Scores Frequencies")
        plt.show()


if __name__ == "__main__":
    #train(params.breakout)
    #test(params.pong, "Breakout/breakout_model.pth")
    test(params.breakout, "Breakout/breakout_model.pth")
    #performance(params.pong, "Breakout/breakout_model.pth")
    #performance(params.breakout, "Breakout/breakout_model.pth")