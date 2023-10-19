# (Hyper)parameters
breakout = "BreakoutNoFrameskip-v4"
pong = "PongNoFrameskip-v4"
num_episodes = 600 # number of episodes to run the algorithm
pong_buffer_size = (10 ** 5 * 3) # size of the buffer used in E1 (300k -> 8GB)
breakout_buffer_size = (10 ** 5 * 3)*3 # size of the buffer used in E2 (900k -> 24GB)
mixed_buffer_size = (10 ** 5 * 3)*3/2 # size of the buffer used in E3  (450k -> 12GB)
epsilon = 1.0 # initial probablity of selecting random action a, annealed over time
timesteps = 0 # counter for number of frames
pong_minibatch_size = 128
breakout_minibatch_size = 128
gamma = 0.99 # discount factor
tau = 1e-3 
learning_rate = 0.00001
update_after = 2000 # update after num time steps
epsilon_decay = 10**5
epsilon_ub = 1.0
epsilon_lb = 0.02
