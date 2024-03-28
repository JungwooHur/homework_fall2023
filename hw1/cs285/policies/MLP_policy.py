"""
Defines a pytorch policy as the agent's actor

Functions to edit:
    2. forward
    3. update
"""

#@ My comments start with #@.

#@ Defines MLP policy that outputs continuous actions using Supervised Learning

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy

#@ Consturcting a feedforward neural network with given arguments

#@ arguments:
#@ input size, output size: size of the NN's input output Layer
#@ n_layers: the number of hidden layers
#@ size: designate each layers' number of neuruns

def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        #@ Use tanh activation function for hidden layers 
        layers.append(nn.Tanh())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))

    mlp = nn.Sequential(*layers)
    return mlp

#@ Defines MLP based policy that maps observations to continuous actions
#@ The class works as PyTorch model with BasePolicy and nn.Module 
class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to continuous
    actions.

    Attributes
    ----------
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """
    #@ initialize main parameters, such as learning rate, dimensions of ob and ac, size of the layers
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline
        
        #@ mean_net: NN that predicts mean of the action relative to the observation 
        self.mean_net = build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.size,
        )
        self.mean_net.to(ptu.device)
        
        #@ logstd is (Parameter) the log standard deviation of the action which can be trained
        #@ It is used for modeling the action distribution in continuous action space
        self.logstd = nn.Parameter(

            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)
        #@ Adam optimizer used for updating parameters?
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )

    #@ Save the trained model as a file
    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    #@ Defines the forward pass of the network. Get the observation and ouputs the action distribution
    #@ observation => mean_net => make action distribution with results and logstd.
    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        # TODO: implement the forward pass of the network.
        # You can return anything you want, but you should be able to differentiate
        # through it. For example, you can return a torch.FloatTensor. You can also
        # return more flexible objects, such as a
        # `torch.distributions.Distribution` object. It's up to you!
        
        #@---
        #@My code
        mean = self.mean_net(observation)
        std = torch.exp(self.logstd)
        
        act_dist = distributions.Normal(mean, std)
        sampled_action = act_dist.sample()
        
        
        return sampled_action
        #@---

    
    #@ Updates policy with observations and paired actions. The Supervised Learning
    #@ process that agent mimics expert's actions
    def update(self, observations, actions):
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        #@ Define the loss function, update the parameters in NN with back-propagation?
        #@ Normally we minimize the difference between estimated action and the expert's action
        
        # TODO: update the policy and return the loss
        
        #@---
        #@My code
        
        
        #@ predict action with NN
        obs_tensor = ptu.from_numpy(observations)
        acs_tensor = ptu.from_numpy(actions)
        
        estimated_actions = self.mean_net(obs_tensor)
        
        #@ compute loss with Mean Squared Error  => Mean[(estimated actions - expert's actions)^2]
        loss_fn = nn.MSELoss()
        loss = loss_fn(estimated_actions, acs_tensor)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
        #@---
