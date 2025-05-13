import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
from typing import Sequence


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    #activation: str = "tanh"
    activation: str = "leaky_relu" 
    num_layers: int = 3
    num_nodes: int = 64
    out_activation: str = "hard_tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "tanh":
            activation = nn.tanh
        elif self.activation == "leaky_relu":
            activation = nn.leaky_relu
        else:
            raise ValueError("Invalid activation function")
        
        # Output activation:

        if self.out_activation == "hard_tanh":
            out_activation = nn.hard_tanh
        elif self.out_activation == "relu":
            out_activation = nn.relu
        elif self.out_activation == "clipped_relu":
            out_activation = nn.clipped_relu
        else:
            raise ValueError("Invalid output activation function.")
    
        actor_mean = x
        for _ in range(self.num_layers):
            actor_mean = nn.Dense(self.num_nodes, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
            actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

        actor_mean = out_activation(actor_mean)
        # actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        # actor_mean = activation(actor_mean)
        # actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        # actor_mean = activation(actor_mean)
        # actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        # actor_mean = activation(actor_mean)
        # actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        # actor_mean = activation(actor_mean)
        actor_logstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(loc=actor_mean, scale_diag=jnp.exp(actor_logstd))

        critic = x
        for _ in range(self.num_layers):
            critic = nn.Dense(self.num_nodes, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
            critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        
        # critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        # critic = activation(critic)
        # critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        # critic = activation(critic)
        # critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        # critic = activation(critic)
        # critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)