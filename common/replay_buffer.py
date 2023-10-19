from common.utils import Transition
import torch
import random
import numpy as np

# Replay Buffer class - stores transitions - state, action, reward, next_state, done
class ReplayBuffer():

    def __init__(self, num_actions, memory_len = 10000):
        self.memory_len = memory_len
        self.transition = []
        self.num_actions = num_actions

    def add(self, state, action, reward, next_state, done):
        if self.length() > self.memory_len:
            self.remove()
        self.transition.append(Transition(state, action, reward, next_state, done))

    def sample_batch(self, batch_size = 32):
        minibatch = random.sample(self.transition, batch_size)

        states_mb, a_, reward_mb, next_states_mb, done_mb = map(np.array, zip(*minibatch))

        mb_reward = torch.from_numpy(reward_mb).cuda()
        mb_done = torch.from_numpy(done_mb.astype(int)).cuda()

        a_mb = np.zeros((a_.size, self.num_actions))
        a_mb[np.arange(a_.size), a_] = 1
        mb_a = torch.from_numpy(a_mb).cuda()

        return states_mb, mb_a, mb_reward, next_states_mb, mb_done

    def length(self):
        return len(self.transition)

    def remove(self):
        self.transition.pop(0)