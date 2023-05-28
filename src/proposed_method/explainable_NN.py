# -*- coding: utf-8 -*-
"""
Created on Sun May 7 2023

@author: javier mediavilla relaño
"""

import matplotlib.pyplot
import numpy
import torch
import typing


class explainable_NN(torch.nn.Module):
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

        super(explainable_NN, self).__init__()

        self.eps = 0.000000000000001  # Used in ratios for numerical stability.
        self.n_out = n_out
        self.explainable_sample = torch.nn.parameter.Parameter(torch.from_numpy(numpy.zeros(input_size)), True)
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
        self.gradients_off()

    def forward(self) -> torch.Tensor:
        """
        This function evaluates the Neural Network on the counterfactual sample.

        Returns:
            o (torch.Tensor): The output of the Neural Network evaluated on 'input'.
        """
        o = torch.relu(self.hidden0(self.explainable_sample.float()))  # Relu activation on hidden layer
        o = torch.tanh(self.out(o))  # Tanh activation on output layer
        return o

    def forward_with_no_final_activation(self) -> torch.Tensor:
        """
        The aim of this function is to know the output of the Neural Network before the final output activation.

        Returns:
            o (torch.Tensor): The output of the Neural Network evaluated on 'x_input'.
        """
        o = torch.relu(self.hidden0(self.explainable_sample.float()))  # Relu activation on hidden layer
        o = self.out(o)  # No activation on output layer
        return o

    def eval_samples(self, x_input: torch.Tensor) -> torch.Tensor:
        """
        This function evaluates the Neural Network for a specified x_input (not necessarily the counterfactual sample).

        [Warning]: The code is only adapted to binary classification when the labels are [-1, +1], specifically the
                   x_input data, MUST have label '-1' for the mayority class, and '+1' for the minority class.

        [Warning]: In this code, the output probabilities of the model are understood as the probability of
                   a sample belonging to class 1, that is, the minority class. The probability is expresed in float
                   format in the rage [0, 1].
        Args:
            x_input (torch.Tensor): The samples to evaluate the Neural Network.

        Returns:
            o (torch.Tensor): The output of the Neural Network evaluated on 'x_input'.
        """

        if x_input.dtype == torch.float64:
            x_input = x_input.float()
        elif x_input.dtype == torch.float32:
            x_input = x_input
        else:
            x_input = torch.from_numpy(x_input).float()

        #  ###### Evaluate the Neural Network ###### #
        o = torch.relu(self.hidden0(x_input))  # Relu activation on hidden layer
        o = torch.tanh(self.out(o))  # Tanh activation on output layer
        # ########################################## #

        return o

    def set_explainable_sample(self, value_tensor: torch.Tensor) -> None:
        """
        Set the variable explainable_sample with the value of the 'value_tensor' input as a torch tensor with the param
        'requires_grad' activated.

        Args:
            value_tensor (torch.Tensor): An original sample to find its counterfactual.
        """
        self.explainable_sample = torch.nn.parameter.Parameter(value_tensor.detach().clone(), True)

    def gradients_off(self, verbose: int = 0):
        """
        Turn off the coumputable gradients of all the parameters of the Newral Network. After this function usually must
        call the 'function set_explainable_sample()', so the only paramers that can compute gradients on the Neural
        Network are the varaibles on the sample where to find its counterfactual.

        Args:
            verbose (int): Depending on the number diferent level information is printed.
                       [0: no prints; 1: high level information; 2: full detail]. Default to 1.
        """
        if verbose > 1:
            print('Transforming the model into explainable...')
            print('requires_grad of the parameters Before transformation:')
            for param in self.parameters():
                print('\t', param.requires_grad)

        for param in self.parameters():
            param.requires_grad = False

        if verbose > 1:
            print('requires_grad of the parameters After transformation:')
            for param in self.parameters():
                print('\t', param.requires_grad)
            print('DONE\n')

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
        o = self.eval_samples(x_input=x_input).detach().flatten().numpy()
        y_pred = o > frontera
        y_pred = y_pred * 1.0

        # To convert the pred class into the same range of the labels, i.e., [-1, +1]
        y_pred = ((2 * y_pred) - 1).astype(int)

        return y_pred

    def predict_with_costs(self,
                           tensor_output: torch.Tensor,
                           Qneutral: float,
                           cost_c01: torch.Tensor,
                           cost_c10: torch.Tensor,
                           cost_c11: torch.Tensor,
                           cost_c00: torch.Tensor) -> torch.Tensor:
        """
        This function applies the example dependent Bayes cost frontier on the output values of a model evaluation.

        [Warning]: The code is only adapted to binary classification when the labels are [-1, +1], specifically the
                   input data, MUST have label '-1' for the mayority class, and '+1' for the minority class.

        Args:
            tensor_output (torch.Tensor): The output of a model evaluation.
            Qneutral (float): A value that indicates the amount of rebalance made during the training process.
            cost_c01 (torch.Tensor): The cost 01 for the x_input samples
            cost_c10 (torch.Tensor): The cost 10 for the x_input samples
            cost_c11 (torch.Tensor): The cost 11 for the x_input samples
            cost_c00 (torch.Tensor): The cost 00 for the x_input samples

        Returns:
            tensor_y_pred (torch.Tensor): The dicisions for the Bayes cost frontier.
        """

        # ###### Variables ###### #
        C01 = cost_c01.detach().numpy().flatten()
        C10 = cost_c10.detach().numpy().flatten()
        C11 = cost_c11.detach().numpy().flatten()
        C00 = cost_c00.detach().numpy().flatten()

        if len(C11) == 0:
            C11 = numpy.zeros(len(C01))

        if len(C00) == 0:
            C00 = numpy.zeros(len(C01))
        # ####################### #

        # ###### Compute the costs frontier and decide with it ###### #
        tensor_P = tensor_output.view(len(tensor_output))

        Q = 1. * Qneutral * (C10 - C00 + self.eps)/(1. * (C01 - C11 + self.eps))
        nu = (Q-1)/(1. * (Q+1))
        dd = numpy.isnan(nu)
        nu[dd] = 0.0

        tensor_nu = torch.from_numpy(nu)
        tensor_y_pred = tensor_P.double() > tensor_nu.double()
        tensor_y_pred = (2 * tensor_y_pred.double()) - 1
        # ########################################################### #

        return tensor_y_pred

    def bayes_discriminant(self,
                           tensor_output: torch.Tensor,
                           RBI: float,
                           cost_policy_function: typing.Callable,
                           mean_tr: numpy.ndarray,
                           std_tr: numpy.ndarray,
                           extra_params: typing.Dict[str, object] = {}) -> torch.Tensor:
        """
        This function computes the Bayes discriminant ussing only operations that are compatible with the gradient
        memory of torch, in ordet to be able to compute the gradients of the Bayes discriminant on the dimensions
        of the optimizable counterfactual variable.

        Args:
            tensor_output (torch.Tensor): The previously evaluated output of the Neural Network.
            RBI (float): A value that indicates the amount of rebalance made during the training process.
            cost_policy_function (typing.Callable): A function that computes the 4 cost values following the ecuations
                                                    of the specified dataset.
            mean_tr (numpy.ndarray): Contains the mean for each varable of the original train set. The standard
                                     desviation used when the data was originally standarized.
            std_tr (numpy.ndarray): Contains the standard desviation for each variable of the original train set. The
                                standard desviation used when the data was originally standarized.
            extra_params (typing.Dict[str, object]): A dict of extra params for the cost_policy_function(). Default
                                                     to {}.

        Returns:
            tensor_d (torch.Tensor): The value of the Bayes discriminant for the evaluated sample.
        """

        cost_policy_result = cost_policy_function(name_dataset=self.name_dataset,
                                                  x_input=self.explainable_sample,
                                                  mean_tr=mean_tr,
                                                  std_tr=std_tr,
                                                  extra_params=extra_params)
        cost_c01 = cost_policy_result['cost_c01']
        cost_c10 = cost_policy_result['cost_c10']
        cost_c11 = cost_policy_result['cost_c11']
        cost_c00 = cost_policy_result['cost_c00']

        tensor_f = cost_c00 - cost_c10
        tensor_g = cost_c10 - cost_c00 + cost_c01 - cost_c11
        tensor_pr1 = torch.div(1, (1 + torch.div((1-tensor_output), torch.mul(RBI, (1+tensor_output)) + self.eps)))

        tensor_d = tensor_f.double() + torch.mul(tensor_g.double(), tensor_pr1.double())
        tensor_d = tensor_d.float()
        return tensor_d

    def bayes_discriminant_control_grad(self,
                                        tensor_output: torch.Tensor,
                                        RBI: float,
                                        cost_policy_function: typing.Callable,
                                        mean_tr: numpy.ndarray,
                                        std_tr: numpy.ndarray,
                                        extra_params: typing.Dict[str, object] = {}) -> typing.List[torch.Tensor]:
        """
        This function computes the Bayes discriminant ussing only operations that are compatible with the gradient
        memory of torch, in ordet to be able to compute the gradients of the Bayes discriminant on the dimensions
        of the optimizable counterfactual variable. Also anulates the gradients comming from the terms g and f to
        control the effects of high values of the gradient produced by high values of the cost policy.

        Args:
            tensor_output (torch.Tensor): The previously evaluated output of the Neural Network.
            RBI (float): A value that indicates the amount of rebalance made during the training process.
            cost_policy_function (typing.Callable): A function that computes the 4 cost values following the ecuations
                                                    of the specified dataset.
            mean_tr (numpy.ndarray): Contains the mean for each varable of the original train set. The standard
                                     desviation used when the data was originally standarized.
            std_tr (numpy.ndarray): Contains the standard desviation for each variable of the original train set. The
                                standard desviation used when the data was originally standarized.
            extra_params (typing.Dict[str, object]): A dict of extra params for the cost_policy_function(). Default
                                                     to {}.

        Returns:
            tensor_d (torch.Tensor): The value of the Bayes discriminant for the evaluated sample.
            f (torch.Tensor): The value of the term f of Bayes discriminant for the evaluated sample.
            g (torch.Tensor): The value of the term g of Bayes discriminant for the evaluated sample.
            grad_regularization (torch.Tensor): A value proportional to the cost policy values. Used to regularized
                                                high values of the gradient caused by high values of the cost policy.
        """

        cost_policy_result = cost_policy_function(name_dataset=self.name_dataset,
                                                  x_input=self.explainable_sample,
                                                  mean_tr=mean_tr,
                                                  std_tr=std_tr,
                                                  extra_params=extra_params)
        cost_c01 = cost_policy_result['cost_c01']
        cost_c10 = cost_policy_result['cost_c10']
        cost_c11 = cost_policy_result['cost_c11']
        cost_c00 = cost_policy_result['cost_c00']
        amt_no_standarized = cost_policy_result['amt_no_standarized']

        tensor_f = cost_c00 - cost_c10
        tensor_g = cost_c10 - cost_c00 + cost_c01 - cost_c11
        tensor_pr1 = torch.div(1, (1 + torch.div((1-tensor_output), torch.mul(RBI, (1+tensor_output)) + self.eps)))

        f = tensor_f.double().detach().flatten().numpy()
        g = tensor_g.double().detach().flatten().numpy()
        grad_regularization = amt_no_standarized.double().detach().numpy()

        f = torch.from_numpy(f)
        g = torch.from_numpy(g)
        tensor_d = f + torch.mul(tensor_pr1.double(), g)

        tensor_d = tensor_d.float()
        f = f.float()
        g = g.float()

        return tensor_d, f, g, grad_regularization

    def filter_dimensions_restrictions(self, dimensions_restrictions: numpy.ndarray) -> None:
        """
        This function anulates the gradients of the dimensions of the counterfactual sample acording to the specified
        dimension restrictions.

        Args:
            dimensions_restrictions (numpy.ndarray): The restrictions on witch dimensions can almacenate gradients and
                                                     consequently change its value during the counterfactual searching
                                                     process.
        """

        for ii in range(len(dimensions_restrictions)):
            if dimensions_restrictions[ii] == 0:
                self.explainable_sample.grad[ii] = 0

    def explica_muestra(self,
                        input_sample: torch.Tensor,
                        mean_tr: numpy.ndarray,
                        std_tr: numpy.ndarray,
                        lr_for_regularization: float,
                        lr_for_discriminant: float,
                        epochs: int,
                        cost_policy_function: typing.Callable,
                        RB: float,
                        IR: float,
                        dimensions_restrictions: numpy.ndarray,
                        margen_nu: float,
                        convergence_criteria: float,
                        convergence_criteria_reg: float,
                        regularzation_frontier: float = 0.9,
                        extra_params: typing.Dict[str, object] = {},
                        zero_grad_regularization: bool = False,
                        figure: bool = False,
                        verbose: int = 1) -> typing.Dict[str, torch.Tensor]:
        """
        This function search a counterfactual sample for the specified 'input_sample'. The process is as follows:
            1. Previously during the innit the model has been initilized with pretrained weights and then turned off
               all the gradients of the Newral Network.
            2. Set the input_sample as the explainable_sample variable, with is the input to the Newral Network and
               have its gradient memory activated, so we can compute the gradients on the explainable_sample dimensions
            3. Check if the initial value of the sample is contained in the saturated zone of the tanh ouput activation
            4. If its the case of step 3, proceed the optimization computing the gradients ignoring the activation
               function to prevent the gradientes to be zero until the sample is outside the saturated zone.
            5. Initialize the SGD optimicer with torch.
            6. Continue with the optimization in orther to minimize the square of the Bayes discriminant.
            7. During the optimization process the gradients on the restricted dimensions will set to 0 with the
               function filter_dimensions_restrictions() before recompute the new values of explainable_sample.
            7. The learning rate is regularized by an amount proportional to the value of the cost policy (the amount),
               to prevent high values of the gradient on high values of aount, for example, a client that try to get a
               very high loand will lead to this gradient problem, and therefore need of this regularization.
            8. If at any time the progress in the minimization does not progress more than the convergence_criteria
               on the mean of the last 40 iterations, the optimization porcess will be considered finished and no more
               epoch are computed.
            9. One the optimization is done the variable explainable_sample must contain the value of the
               counterfactual sample.
            10. Finally, measure the final state, and plot the optimization progress if 'figure' is set to True.

        [Warning]: The code is only adapted to binary classification when the labels are [-1, +1], specifically the
                   input data, MUST have label '-1' for the mayority class, and '+1' for the minority class.

        Args:
            input_sample (torch.Tensor): The sample for which we want to find the counterfactual.
            mean_tr (numpy.ndarray): Contains the mean for each varable of the original train set. The standard
                                     desviation used when the data was originally standarized.
            std_tr (numpy.ndarray): Contains the standard desviation for each variable of the original train set. The
                                    standard desviation used when the data was originally standarized.
            lr_for_regularization (float): The learning rate for the zero gradient regularization.
            lr_for_discriminant (float): The learning rate for optimizing the Bayes discriminant.
            epochs (int): The maximun number of epoch used to find each counterfactual. If the Bayes discriminant is
                          minimized before the max epoch with an error of "convergence_criteria", the process is
                          stopped there.
            cost_policy_function (typing.Callable): A functions that computes the 4 values of the cost policy for the
                                                    specified dataset.
            RB (float): The amount of rebalance used during the training of the Neural Network.
            IR (float): The train imbalance of the data.
            dimensions_restrictions (numpy.ndarray): The restrictions on witch dimensions can almacenate gradients and
                                                     consequently change its value during the counterfactual searching
                                                     process.
            margen_nu (float): The extra margin when minimizing the Bayes discriminant to find a counterfactual.
            convergence_criteria (float): The minimal error to consider that a optimization process has reached the
                                          objetive during the counterfactual optimization.
            convergence_criteria_regularization (float): The minimal error to consider that a optimization process
                                                         has reached the objetive during the zero grad regularization.
            regularzation_frontier (float): The value to consider the saturated zone on the tanh activation function.
                                            Default to 0.9.
            extra_params (typing.Dict[str, object]): Contais the extra parameters that might be needed for the process.
                                                     Defatul to {}.
            zero_grad_regularization (bool): If true activates the regularization on saturated zones of the output
                                             activation function. Default to False.
            figure (bool): If true plots aditional information of the process. Default to False.
            verbose (int): Depending on the number diferent level information is printed.
                       [0: no prints; 1: high level information; 2: full detail]. Default to 1.

        Returns:
            original_sample (numpy.ndarray): The original input_sample
            contrafactual_sample (numpy.ndarray): The cost counterfactual sample respect to the orignal input_sample
            estado_final (int): The number of epoch needed to complete the counterfactual search
            discriminant_list (numpy.ndarray): The values of the discriminant along the optimization process
            g_list (typing.List[torch.Tensor]): The values of the term 'g' of the discriminant along the optimization
            f_list (typing.List[torch.Tensor]): The values of the term 'g' of the discriminant along the optimization
        """

        # ###### Initialices the explainable_sample parameter with the input_sample ###### #
        self.set_explainable_sample(input_sample)
        if verbose > 2:
            print('Added sample:\n', input_sample)
        # ################################################################################ #

        # ###### Initilize the optimizers ###### #
        if zero_grad_regularization:

            output_before_activation = self.forward_with_no_final_activation()
            if numpy.abs(output_before_activation.detach().numpy()) > regularzation_frontier:
                lr = float(lr_for_regularization)

            else:  # Use the normal function
                lr = float(lr_for_discriminant)
                zero_grad_regularization = False
        else:
            lr = float(lr_for_discriminant)

        optim = torch.optim.SGD(self.parameters(), lr=lr)
        ##########################################

        RBI = RB
        discriminant_list = numpy.zeros(epochs + 1)
        f_list = []
        g_list = []
        salidas = numpy.zeros(epochs + 1)
        C01 = numpy.zeros(epochs + 1)
        C10 = numpy.zeros(epochs + 1)
        C11 = numpy.zeros(epochs + 1)
        C00 = numpy.zeros(epochs + 1)
        lr_reg_list = numpy.zeros(epochs + 1)

        for ee in range(epochs):

            # Set to zero the gradients for the optimizer
            optim.zero_grad()

            # Evaluates the Neural Network
            output = self.forward()

            # ###### Compute the gradients for the square of the Bayes discriminant with the optimizer ###### #
            if zero_grad_regularization:
                output_before_activation = self.forward_with_no_final_activation()
                distance_regularization = output_before_activation - regularzation_frontier
                if numpy.abs(distance_regularization.detach().numpy()) > convergence_criteria_reg:
                    d = distance_regularization
                    lr = lr_for_regularization
                    optim.param_groups[0]['lr'] = float(lr)

                else:
                    d, f, g, lr_reg = self.bayes_discriminant_control_grad(tensor_output=output,
                                                                           RBI=RBI,
                                                                           cost_policy_function=cost_policy_function,
                                                                           mean_tr=mean_tr,
                                                                           std_tr=std_tr,
                                                                           extra_params=extra_params)
                    f_list.append(f.double().detach().numpy())
                    g_list.append(g.double().detach().numpy())
                    optim.param_groups[0]['lr'] = float(lr_for_discriminant / lr_reg)
                    zero_grad_regularization = False
            else:
                d, f, g, lr_reg = self.bayes_discriminant_control_grad(tensor_output=output,
                                                                       RBI=RBI,
                                                                       cost_policy_function=cost_policy_function,
                                                                       mean_tr=mean_tr,
                                                                       std_tr=std_tr,
                                                                       extra_params=extra_params)
                f_list.append(f.double().detach().numpy())
                g_list.append(g.double().detach().numpy())
                optim.param_groups[0]['lr'] = float(lr_for_discriminant / lr_reg)
            error = (d + margen_nu)**2
            error.backward()

            # #### Saving control variables #### #
            salidas[ee] = output
            discriminant_list[ee] = d
            cost_policy_result = cost_policy_function(name_dataset=self.name_dataset,
                                                      x_input=self.explainable_sample,
                                                      mean_tr=mean_tr,
                                                      std_tr=std_tr,
                                                      extra_params=extra_params)
            cost_c01 = cost_policy_result['cost_c01']
            cost_c10 = cost_policy_result['cost_c10']
            cost_c11 = cost_policy_result['cost_c11']
            cost_c00 = cost_policy_result['cost_c00']
            lr_reg = cost_policy_result['amt_no_standarized']

            C01[ee] = cost_c01.detach().numpy().flatten()
            C10[ee] = cost_c10.detach().numpy().flatten()
            C11[ee] = cost_c11.detach().numpy().flatten()
            C00[ee] = cost_c00.detach().numpy().flatten()
            lr_reg_list[ee] = lr_reg.detach().numpy().flatten()
            # ################################## #
            # ############################################################################################### #

            # ##### Check convergence criteria ##### #
            if ee > 40:  # Start checking after the epoch 40
                last_optim_changes = numpy.abs(discriminant_list[ee] - discriminant_list[ee-10:ee])
                if numpy.mean(last_optim_changes) < convergence_criteria:
                    break
            # ###################################### #

            # Cancel the gradients on the restricted dimensions
            self.filter_dimensions_restrictions(dimensions_restrictions=dimensions_restrictions)

            # #### print the value of the gradients #### #
            if verbose > 2:
                print('Gradient iteration ', ee, ':')
                for param in self.parameters():
                    print(param.grad)
            # ########################################## #

            # Update the values of the dimensions if explainable_sample with the computed gradients and the optimizer
            optim.step()

        # ############ Measure the final state ############ #
        output = self.forward()
        salidas[ee+1] = output
        d, f, g, lr_reg = self.bayes_discriminant_control_grad(tensor_output=output,
                                                               RBI=RBI,
                                                               cost_policy_function=cost_policy_function,
                                                               mean_tr=mean_tr,
                                                               std_tr=std_tr,
                                                               extra_params=extra_params)
        discriminant_list[ee+1] = d
        f_list.append(f.double().detach().numpy())
        g_list.append(g.double().detach().numpy())

        cost_policy_result = cost_policy_function(name_dataset=self.name_dataset,
                                                  x_input=self.explainable_sample,
                                                  mean_tr=mean_tr,
                                                  std_tr=std_tr,
                                                  extra_params=extra_params)
        cost_c01 = cost_policy_result['cost_c01']
        cost_c10 = cost_policy_result['cost_c10']
        cost_c11 = cost_policy_result['cost_c11']
        cost_c00 = cost_policy_result['cost_c00']
        lr_reg = cost_policy_result['amt_no_standarized']

        C01[ee+1] = cost_c01.detach().numpy().flatten()
        C10[ee+1] = cost_c10.detach().numpy().flatten()
        C11[ee+1] = cost_c11.detach().numpy().flatten()
        C00[ee+1] = cost_c00.detach().numpy().flatten()
        lr_reg_list[ee+1] = lr_reg.detach().numpy().flatten()
        # ################################################# #

        # ############ Final results ############ #
        contrafactual_sample = self.explainable_sample.detach().numpy()
        original_sample = input_sample.detach().numpy()
        movimiento_muestra = original_sample - contrafactual_sample
        if verbose > 1:
            print('[Results Info] Distance from the original sample to the counterfactual:\n', movimiento_muestra)

        if figure:
            n_columnas = 4
            n_filas = 2
            fig, axs = matplotlib.pyplot.subplots(2, 4, figsize=(4 * n_columnas, 3.5 * n_filas))
            axs[0, 0].plot(numpy.abs(movimiento_muestra[:ee+1]))
            axs[0, 0].set_title('Movement in input dimensions')
            axs[0, 0].grid(color='grey', linestyle='-', linewidth=0.5)

            axs[1, 0].plot(salidas[:ee+1])
            axs[1, 0].set_title('Net Output -> o(x)')
            axs[1, 0].grid(color='grey', linestyle='-', linewidth=0.5)

            axs[0, 1].plot(discriminant_list[:ee+1])
            axs[0, 1].set_title('Discriminant -> d(x)')
            axs[0, 1].grid(color='grey', linestyle='-', linewidth=0.5)

            axs[0, 2].plot(C01[:ee+1])
            axs[0, 2].set_title('C01')
            axs[0, 2].grid(color='grey', linestyle='-', linewidth=0.5)

            axs[1, 2].plot(C10[:ee+1])
            axs[1, 2].set_title('C10')
            axs[1, 2].grid(color='grey', linestyle='-', linewidth=0.5)

            axs[0, 3].plot(C00[:ee+1])
            axs[0, 3].set_title('C00')
            axs[0, 3].grid(color='grey', linestyle='-', linewidth=0.5)

            axs[1, 3].plot(C11[:ee+1])
            axs[1, 3].set_title('C11')
            axs[1, 3].grid(color='grey', linestyle='-', linewidth=0.5)

            fig.show()
        # ####################################### #

        final_epoch = ee
        g_list = numpy.array(g_list)
        f_list = numpy.array(f_list)

        result = {'original_sample': original_sample,
                  'contrafactual_sample': contrafactual_sample,
                  'final_epoch': final_epoch,
                  'discriminant_list': discriminant_list,
                  'g_list': g_list,
                  'f_list': f_list,
                  }
        return result
