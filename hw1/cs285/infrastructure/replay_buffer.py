from cs285.infrastructure.utils import *


#@ My comments start with #@

#@ ReplayBuffer Importance
#@ In learning, ReplayBuffer plays an important role in improving learning efficiency and stability 
#@ by storing and reusing situations experienced by agents. By re-learning randomly selected 
#@ past experiences in the learning process, agents can learn to respond to more diverse situations 
#@ and reduce the impact of temporal associations between samples on the learning process. 
#@ This method helps to increase the stability of learning, especially in complex environments, where data is sequentially dependent.

#@ Save rollouts(action => env => obs) and manage to use it for training
#@ Main purpose: improve learning efficiency and stability

class ReplayBuffer(object):

    #@ maximum size of experiences that buffer can store
    def __init__(self, max_size=1000000):

        self.max_size = max_size

        # store each rollout
        self.paths = []

        # store (concatenated) component arrays from each rollout
        #@ observations actions rewards next_observations terminals
        self.obs = None
        self.acs = None
        self.rews = None
        self.next_obs = None
        self.terminals = None

    #@ Returns the number of experiences that buffer stores (based on the length of the obs)
    def __len__(self):
        if self.obs:
            return self.obs.shape[0]
        else:
            return 0

    #@ Add new experiences('paths') to the buffer
    def add_rollouts(self, paths, concat_rew=True):

        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)
        
        #@ concatenate the signals in the 'paths'(obs acs rews nobs term) to the existing data

        # convert new rollouts into their component arrays, and append them onto
        # our arrays
        observations, actions, rewards, next_observations, terminals = (
            convert_listofrollouts(paths, concat_rew))
        
        #@ if the buffer size exceeds max_size, delete the oldest data and maintain the size (max_size) with slicing [-self.max_size]
        #@ The way we treat reward data differs according to the 'concat_rew = True or False'
        #@ concat_rew = True => simply concatenate to the array. False => we have options to handle
        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.rews = rewards[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            if concat_rew:
                self.rews = np.concatenate(
                    [self.rews, rewards]
                )[-self.max_size:]
            else:
                if isinstance(rewards, list):
                    self.rews += rewards
                else:
                    self.rews.append(rewards)
                self.rews = self.rews[-self.max_size:]
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_size:]

