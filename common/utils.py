from collections import namedtuple
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
from datetime import datetime

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

SET = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def plot(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()

# Function to calculate time intervals
def get_delta(t1):
    t2 = datetime.now()
    s = (t2 - t1).seconds
    h, re = divmod(s, 3600)
    m, s = divmod(re, 60)
    t1 = t2
    return t1, h, m, s
