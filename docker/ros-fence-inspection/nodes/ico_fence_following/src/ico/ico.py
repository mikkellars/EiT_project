
"""Class for ICO learning node
"""


import random
import math


class ICO():
    def __init__(self, lr:float = 0.1, weight_predic:float = random.uniform(0.0, 0.1), activation_func:str = 'tanh'):
        self.weight_reflex = 1.0
        self.weight_predic = weight_predic
        self.x_reflex = 0.0
        self.x_predic = 0.0
        self.lr = lr
        self.activation_func = activation_func
        self.output = 0.0

    def run_and_learn(self, x_reflex:float, x_predic:float) -> float:
        """Propagates through the network and updates the weight.
        Args:
            x_reflex (float): The reflex signal input
            x_predic (float): The predictive signal input
        Returns:
            output (float): The propagated value
        """
        output = self.forward_prop(x_reflex, x_predic)
        if x_reflex != 0:
            self.update_weight(x_reflex)

        # Updates input signals 
        self.x_reflex = x_reflex
        self.x_predic = x_predic

        return output

    def reset_network(self):
        """Resets the weights and signals back to default values
        """
        self.weight_reflex = 1.0
        self.weight_predic = random.random()
        self.x_reflex = 0.0
        self.x_predic = 0.0

    def forward_prop(self, x_reflex:float, x_predic:float) -> float:
        """Propagates through the network with the learned weights
        Args:
            x_reflex (float): The reflex signal input
            x_predic (float): The predictive signal input
        Returns:
            y (float): The propagated value
        """
        u = self.weight_reflex * x_reflex + self.weight_predic * x_predic
        if self.activation_func is 'tanh':
            self.output = math.tanh(u)
        elif self.activation_func is 'sigmoid':
            self.output = self.__sigmoid(u)
        else:
            ValueError('Argument only support tanh or sigmoid')
        
        return self.output

    def update_weight(self, x_reflex_new:float):
        """Updates the weights for the predictive signals.
        The network only learns when the reflex signal is on
        Args:
            x_reflex_new (float): The new reflex signal as input
        """
        self.weight_predic += self.lr * self.x_predic * (x_reflex_new - self.x_reflex)

    @staticmethod
    def __sigmoid(input:float) -> float:
        return 1.0/(1.0+math.exp(-input))