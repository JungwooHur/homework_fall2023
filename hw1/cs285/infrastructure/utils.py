"""A
Some miscellaneous utility functions

Functions to edit:
    1. sample_trajectory
"""

from collections import OrderedDict
import cv2
import numpy as np
import time

from cs285.infrastructure import pytorch_util as ptu

#@ useful to automate the agent's performance evaluation, data collection, 
#@ and preprocessing process in algorithms. By applying the agent's behavior policy 
#@ to the environment, the data can be analyzed, and based on this, the agent's learning 
#@ process can be adjusted, and performance can be improved.

#@ samples the rollout that agent follows the policy at env
def sample_trajectory(env, policy, max_path_length, render=False):
    """Sample a rollout in the environment from a policy."""
    
    #@ init env and get init obs
    
    # initialize env for the beginning of a new rollout
    ob =  env.reset() # TODO: initial observation after resetting the env

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    
    #@ while loop: select action according to the policy, act => get next_obs rews term 
    steps = 0
    
    
    while True:

        # render image of the simulated env
        if render:
            if hasattr(env, 'sim'):
                img = env.sim.render(camera_name='track', height=500, width=500)[::-1]
            else:
                img = env.render()
            # image_obs.append(cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC))
            #@ 
            if img is not None:
                resized_img = cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
                image_obs.append(resized_img)

        #@---
        #@My code

        # TODO use the most recent ob to decide what to do
        ob_tensor = ptu.from_numpy(ob)
        sampled_action = policy.forward(ob_tensor)# HINT: this is a numpy array
        action_nparray = ptu.to_numpy(sampled_action)

        # TODO: take that action and get reward and next ob
        next_ob, rew, term, trun, info  = env.step(action_nparray)
        
        done = term or trun
        
        # TODO rollout can end due to done, or due to max_path_length
        steps += 1
        rollout_done = done or steps >= max_path_length
        
        # rollout_done = TODO # HINT: this is either 0 or 1
        
        #@---
        
        #@ record
        # record result of taking that action
        obs.append(ob)
        acs.append(sampled_action)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    #@ convert each components to array and return it as dictionary 
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}

#@ sample some trajectories. Collect data untill the number of steps reach min_timesteps_per_batch   
def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False):
    """Collect rollouts until we have collected min_timesteps_per_batch steps."""

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:

        #collect rollout
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)

        #count steps
        timesteps_this_batch += get_pathlength(path)

    return paths, timesteps_this_batch

#@ sample some trajectories. Collect data untill the number of paths reach max_path_length   
def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False):
    """Collect ntraj rollouts."""

    paths = []
    for i in range(ntraj):
        # collect rollout
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)
    return paths


########################################
########################################

#@ convert rollouts to seperate arrays
def convert_listofrollouts(paths, concat_rew=True):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    return observations, actions, rewards, next_observations, terminals


########################################
########################################
            
#@ compute metrics ex) mean standard deviation max min returns(rewards), mean episode length
#@ return the metrics to form of OrderedDict
def compute_metrics(paths, eval_paths):
    """Compute metrics for logging."""

    # returns, for logging
    train_returns = [path["reward"].sum() for path in paths]
    eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

    # episode lengths, for logging
    train_ep_lens = [len(path["reward"]) for path in paths]
    eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

    # decide what to log
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs


############################################
############################################

#@ return given path's reward length 
def get_pathlength(path):
    return len(path["reward"])
