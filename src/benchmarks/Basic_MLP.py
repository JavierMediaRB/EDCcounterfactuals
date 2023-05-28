# -*- coding: utf-8 -*-
"""
Created on Sun May 7 2023

@author: javier mediavilla relaño
"""

import numpy
import torch


class Basic_MLP(torch.nn.Module):
    """
    This class implements tools to find the counterfactual samples on the decissions of a Neural Network classifier
    implemented in pytorch. The expected procedure is as follows:
        1. Train your Neural Network in torch on a different proccess.
        2. Initialize this class with the same size and architecture fo your previuosly trained Neural Network.
        3. Load the weigths of your pretrained model into this class with the function 'load_state_dict()' of torch.
        4. Explain any sample that you want with the function 'explica_muestra()' of this class (one sample explained
           on each call).
        5. Use the rest of auxiliar functions to measure any metric (savings, distance, etc.) on the finded
           counterfactual samples.

    [Warning]: The code is only adapted to binary classification when the labels are [-1, +1], specifically the input
               data, MUST have label '-1' for the mayority class, and '+1' for the minority class.

    [Warning]: In this code, the output probabilities of the model are understood as the probability of
               a sample belonging to class 1, that is, the minority class. The probability is expresed in float format
               in the rage [0, 1].
    """

    def __init__(self,
                 name_dataset: str,
                 pretrained_weights: object,
                 input_size: int,
                 hidden_size: int,
                 n_out: int = 1) -> None:
        """
        Inicialization of the Neural Network for the specified size.

        First all the elements of the Newral Network are defined with specified size. Then the NN is fill in with the
        pretrained weights of the Neural Network that we want to explain. Finally the gradient memory of all the weights
        of the NN are turn off in order to preserve the original weights values during the optimization process of
        searching the counterfactual.

        Args:
            name_dataset (str): The name of the dataset where to perform the explanation.
            pretrained_weights (object): The pretrained weights of the Neural Network that we want to explain its
                                         decission by finding the counterfactual samples.
            input_size (int): The number of dimensions of the input samples.
            hidden_size (int): The number of neurons in the hidden layer of the Neural Network.
            n_out (int): The number of outputs of the Neural Network. Default to 1.
        """
        super(Basic_MLP, self).__init__()

        self.eps = 0.000000000000001  # Used in ratios for numerical stability.
        self.n_out = n_out
        self.name_dataset = name_dataset

        # Function to initilice the weights.
        def init_weights(m) -> None:
            if type(m) == torch.nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)

        # First, declarating the conections of the layer
        self.hidden0 = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size))
        self.hidden0.apply(init_weights)  # Secondly, initializing the weigths

        # Third, declarating the conections of the second layer
        self.out = torch.nn.Sequential(torch.nn.Linear(hidden_size, self.n_out))
        self.out.apply(init_weights)  # Finally, initializing the weigths of the second layer

        self.load_state_dict(pretrained_weights, strict=False)

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        """
        This function evaluates the Neural Network for a specified input (not necessarily the counterfactual sample).

        [Warning]: The code is only adapted to binary classification when the labels are [-1, +1], specifically the
                   input data, MUST have label '-1' for the mayority class, and '+1' for the minority class.

        [Warning]: In this code, the output probabilities of the model are understood as the probability of
                   a sample belonging to class 1, that is, the minority class. The probability is expresed in float
                   format in the rage [0, 1].
        Args:
            input (torch.Tensor): The samples to evaluate the Neural Network.

        Returns:
            o (torch.Tensor): The output of the Neural Network evaluated on 'input'.
        """
        if x_input.dtype == torch.float64:
            x_input = x_input.float()
        elif x_input.dtype == torch.float32:
            x_input = x_input
        else:
            x_input = torch.from_numpy(x_input).float()

        o = torch.relu(self.hidden0(x_input))  # Relu activation on hidden layer
        o = torch.tanh(self.out(o))  # Tanh activation on output layer
        return o

    def forward_2_outputs(self, x_input: torch.Tensor) -> torch.Tensor:
        """
        This function evaluates the Neural Network for a specified input (not necessarily the counterfactual sample).

        [Warning]: The code is only adapted to binary classification when the labels are [-1, +1], specifically the
                   input data, MUST have label '-1' for the mayority class, and '+1' for the minority class.

        [Warning]: In this code, the output probabilities of the model are understood as the probability of
                   a sample belonging to class 1, that is, the minority class. The probability is expresed in float
                   format in the rage [0, 1].
        Args:
            input (torch.Tensor): The samples to evaluate the Neural Network.

        Returns:
            o (torch.Tensor): The output of the Neural Network evaluated on 'input'. This time as a 2 dimention output
                              where both outputs are complementary probabilities, i.e., [1 - prob, prob].
        """
        # Eval the model
        o = self.forward(x_input=x_input)

        # ##### adapt the output to mach a 2 dim output ##### #
        o_reshape = torch.reshape(o, (o.shape[0], 1))
        o_0_to_1 = (o_reshape + 1) / 2
        cat_o_2_outputs = torch.cat((1 - o_0_to_1, o_0_to_1), 1)
        # ################################################### #

        return cat_o_2_outputs

    def forward_2_outputs_numpy(self, x_input: torch.Tensor) -> torch.Tensor:
        """
        This function evaluates the Neural Network for a specified input (not necessarily the counterfactual sample).

        [Warning]: The code is only adapted to binary classification when the labels are [-1, +1], specifically the
                   input data, MUST have label '-1' for the mayority class, and '+1' for the minority class.

        [Warning]: In this code, the output probabilities of the model are understood as the probability of
                   a sample belonging to class 1, that is, the minority class. The probability is expresed in float
                   format in the rage [0, 1].
        Args:
            input (torch.Tensor): The samples to evaluate the Neural Network.

        Returns:
            o (torch.Tensor): The output of the Neural Network evaluated on 'input'. This time as a 2 dimention output
                              where both outputs are complementary probabilities, i.e., [1 - prob, prob].
        """
        # Eval the model
        o = self.forward(x_input=x_input).detach().numpy().flatten()

        # ##### adapt the output to mach a 2 dim output ##### #
        o = (o + 1) / 2
        result = numpy.array([o, 1 - o]).transpose()
        # ################################################### #

        return result

    def predict_class(self, x_input: torch.Tensor) -> numpy.ndarray:
        """
        Evaluates the Neural Network on the 'x_input' variables and predict the decision using as frontier the value 0.0
        The possible predictions are [0, 1], not [-1, 1].

        [Warning]: The code is only adapted to binary classification when the labels are [-1, +1], specifically the
                   input data, MUST have label '-1' for the mayority class, and '+1' for the minority class.

        Args:
            x_input (torch.Tensor): The input samples to evaluate the Neural Network.

        Returns:
            y_pred (numpy.ndarray): The decisions of the x_input samples.
        """
        frontera = 0.0
        o = self.forward(x_input=x_input).detach().flatten().numpy()
        y_pred = o > frontera
        y_pred = y_pred * 1.0

        # To convert the pred class into the same range of the labels, i.e., [-1, +1]
        y_pred = ((2 * y_pred) - 1).astype(int)

        return y_pred

    def predict_class_2_output(self, x_input):
        """
        TODO: ...

        Args:

        Returns:

        """
        frontera = 0.0
        o = self.forward(x_input=x_input).detach().flatten().numpy()
        y_pred = o > frontera
        y_pred = y_pred * 1.0

        # To convert the pred class into the same range of the labels, i.e., [-1, +1]
        y_pred = ((2 * y_pred) - 1).astype(int)
        res = numpy.array([y_pred * -1, y_pred]).transpose()

        return res
