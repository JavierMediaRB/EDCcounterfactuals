import numpy
import torch
import typing
from src.cost_policy.functions_from_costcla_repository import _creditscoring_costmat, _creditscoring_costmat_pytorch


def funcion_costes_numpy(name_dataset: str,
                         x_input: numpy.ndarray,
                         mean_tr: numpy.ndarray,
                         std_tr: numpy.ndarray,
                         extra_params: typing.Dict[str, object] = {}) -> typing.Dict[str, numpy.ndarray]:
    """
    This function computes the value of each of the 4 terms of the cost policy for the samples in 'x_input' following
    the cost formulation of the specified Dataset.

    Args:
        name_dataset (str): The name of the dataset. Must be one of the supported datasets names.
        x_input (numpy.ndarray): The standarized samples for compute its cost policy values.
        mean_tr (numpy.ndarray): Contains the mean for each varable of the original train set. The standard desviation
                                 used when the data was originally standarized.
        str_tr (numpy.ndarray): Contains the standard desviation for each variable of the original train set. The
                                standard desviation used when the data was originally standarized.
        extra_params (typing.Dict[str: object]): Contais the extra parameters that might be needed for the cost policy
                                                 of a certain dataset.

    Returns:
        result (typing.Dict[str: object]): A Dictionary containing the following variables:
            - cost_c01 (numpy.ndarray): The cost 01 for the x_input samples
            - cost_c10 (numpy.ndarray): The cost 10 for the x_input samples
            - cost_c11 (numpy.ndarray): The cost 11 for the x_input samples
            - cost_c00 (numpy.ndarray): The cost 00 for the x_input samples
            - amt_no_standarized (numpy.ndarray): The non standarized 'amount' variable for the x_input samples
    """

    cost_c01 = None
    cost_c10 = None
    cost_c11 = None
    cost_c00 = None
    amt_no_standarized = None

    if name_dataset == 'CCF':

        dim_amt = 28
        std = std_tr[dim_amt]
        mean = mean_tr[dim_amt]
        amt = x_input[:, dim_amt]
        amt_no_standarized = (std * amt) + mean

        cost_c01 = 0.75 * amt_no_standarized
        cost_c10 = 0.05 * amt_no_standarized
        cost_c11 = numpy.zeros(len(amt_no_standarized))
        cost_c00 = numpy.zeros(len(amt_no_standarized))

    elif name_dataset == 'CS1':

        dim_amt = 4  # MonthlyIncome
        dim_debt = 3  # DebtRatio
        amt = x_input[:, dim_amt]
        amt_no_standarized = (std_tr[dim_amt] * amt) + mean_tr[dim_amt]
        debt = x_input[:, dim_debt]
        debt_no_standarized = (std_tr[dim_debt] * debt) + mean_tr[dim_debt]

        cost_mat_recreated = _creditscoring_costmat(income=amt_no_standarized,
                                                    debt=debt_no_standarized,
                                                    pi_1=extra_params['pi_1'],
                                                    cost_mat_parameters=extra_params)

        cost_c10 = cost_mat_recreated[:, 0]
        cost_c01 = cost_mat_recreated[:, 1]
        cost_c11 = cost_mat_recreated[:, 2]
        cost_c00 = cost_mat_recreated[:, 3]

    elif name_dataset == 'CS2':

        dim_amt = 10  # PERSONAL_NET_INCOME
        amt = x_input[:, dim_amt]
        amt_no_standarized = (std_tr[dim_amt] * amt) + mean_tr[dim_amt]
        amt_no_standarized = amt_no_standarized * 0.33
        debt = numpy.zeros(x_input.shape[0])

        cost_mat_recreated = _creditscoring_costmat(income=amt_no_standarized,
                                                    debt=debt,
                                                    pi_1=extra_params['pi_1'],
                                                    cost_mat_parameters=extra_params)

        cost_c10 = cost_mat_recreated[:, 0]
        cost_c01 = cost_mat_recreated[:, 1]
        cost_c11 = cost_mat_recreated[:, 2]
        cost_c00 = cost_mat_recreated[:, 3]

    elif name_dataset == 'HMEQ':

        dim_amt = 0
        std = std_tr[dim_amt]
        mean = mean_tr[dim_amt]
        amt = x_input[:, dim_amt]
        amt_no_standarized = (std * amt) + mean

        cost_c01 = 0.75 * amt_no_standarized
        cost_c10 = 0.15 * amt_no_standarized
        cost_c11 = numpy.zeros(len(amt_no_standarized))
        cost_c00 = numpy.zeros(len(amt_no_standarized))

    elif name_dataset == 'MKT':

        dim_amt = 1
        std = std_tr[dim_amt]
        mean = mean_tr[dim_amt]
        amt = x_input[:, dim_amt]
        amt_no_standarized = (std * amt) + mean
        c_a = 1
        weighted_amt_no_standarized = amt_no_standarized * 0.00615
        pos_c_a = weighted_amt_no_standarized < c_a

        cost_c01 = weighted_amt_no_standarized.copy()
        cost_c01[pos_c_a] = c_a
        cost_c10 = numpy.zeros(len(amt_no_standarized)) + c_a
        cost_c11 = numpy.zeros(len(amt_no_standarized)) + c_a
        cost_c00 = numpy.zeros(len(amt_no_standarized))

    elif name_dataset == 'TCC':

        dim_amt = 1
        std = std_tr[dim_amt]
        mean = mean_tr[dim_amt]
        amt = x_input[:, dim_amt]
        amt_no_standarized = (std * amt) + mean

        cost_c01 = 12.0 * amt_no_standarized
        cost_c10 = 3.0 * amt_no_standarized
        cost_c11 = numpy.zeros(len(amt_no_standarized))
        cost_c00 = numpy.zeros(len(amt_no_standarized))

    else:
        print("[Error]: The indicated 'name_dataset' is not supported.")

    result = {'cost_c01': cost_c01,
              'cost_c10': cost_c10,
              'cost_c11': cost_c11,
              'cost_c00': cost_c00,
              'amt_no_standarized': amt_no_standarized}

    return result


def funcion_costes_torch(name_dataset: str,
                         x_input: torch.Tensor,
                         mean_tr: numpy.ndarray,
                         std_tr: numpy.ndarray,
                         extra_params: typing.Dict[str, object] = {}) -> typing.Dict[str, torch.Tensor]:
    """
    This function computes the value of each of the 4 terms of the cost policy for the samples in 'x_input' following
    the cost formulation of the specified Dataset. In this function all the operations computed over the tensor
    'x_input' are compatible with the computation of gradients in the process of finding the optimal counterfactual.

    Args:
        name_dataset (str): The name of the dataset. Must be one of the supported datasets names.
        x_input (torch.Tensor): The standarized samples for compute its cost policy values.
        mean_tr (numpy.ndarray): Contains the mean for each varable of the original train set. The standard desviation
                                 used when the data was originally standarized.
        str_tr (numpy.ndarray): Contains the standard desviation for each variable of the original train set. The
                                standard desviation used when the data was originally standarized.
        extra_params (typing.Dict[str: object]): Contais the extra parameters that might be needed for the cost policy
                                                 of a certain dataset.

    Returns:
        result (typing.Dict[str: object]): A Dictionary containing the following variables:
            - cost_c01 (torch.Tensor): The cost 01 for the x_input samples
            - cost_c10 (torch.Tensor): The cost 10 for the x_input samples
            - cost_c11 (torch.Tensor): The cost 11 for the x_input samples
            - cost_c00 (torch.Tensor): The cost 00 for the x_input samples
            - amt_no_standarized (torch.Tensor): The non standarized 'amount' variable for the x_input samples
    """

    cost_c01 = None
    cost_c10 = None
    cost_c11 = None
    cost_c00 = None
    amt_no_standarized = None

    if name_dataset == 'CCF':

        dim_amt = 28
        std = std_tr[dim_amt]
        mean = mean_tr[dim_amt]
        amt = x_input[dim_amt]
        amt_no_standarized = torch.mul(std, amt) + mean

        cost_c01 = torch.mul(0.75, amt_no_standarized)
        cost_c10 = torch.mul(0.05, amt_no_standarized)
        cost_c11 = torch.from_numpy(numpy.zeros(1))
        cost_c00 = torch.from_numpy(numpy.zeros(1))

    elif name_dataset == 'CS1':

        dim_amt = 4  # MonthlyIncome
        dim_debt = 3  # DebtRatio
        amt = x_input[dim_amt]
        amt_no_standarized = torch.mul(std_tr[dim_amt], amt) + mean_tr[dim_amt]
        debt = x_input[dim_debt]
        debt_recuperado = torch.mul(std_tr[dim_debt], debt) + mean_tr[dim_debt]

        cost_c10, cost_c01, cost_c11, cost_c00 = _creditscoring_costmat_pytorch(income=amt_no_standarized,
                                                                                debt=debt_recuperado,
                                                                                pi_1=extra_params['pi_1'],
                                                                                cost_mat_parameters=extra_params)

    elif name_dataset == 'CS2':

        dim_amt = 10  # PERSONAL_NET_INCOME
        amt = x_input[dim_amt]
        amt_no_standarized = torch.mul(std_tr[dim_amt], amt) + mean_tr[dim_amt]
        amt_no_standarized = torch.mul(amt_no_standarized, 0.33)
        debt = torch.from_numpy(numpy.zeros(1))

        cost_c10, cost_c01, cost_c11, cost_c00 = _creditscoring_costmat_pytorch(income=amt_no_standarized,
                                                                                debt=debt,
                                                                                pi_1=extra_params['pi_1'],
                                                                                cost_mat_parameters=extra_params)

    elif name_dataset == 'HMEQ':

        dim_amt = 0
        std = std_tr[dim_amt]
        mean = mean_tr[dim_amt]
        amt = x_input[dim_amt]
        amt_no_standarized = torch.mul(std, amt) + mean

        cost_c01 = torch.mul(0.75, amt_no_standarized)
        cost_c10 = torch.mul(0.15, amt_no_standarized)
        cost_c11 = torch.from_numpy(numpy.zeros(1))
        cost_c00 = torch.from_numpy(numpy.zeros(1))

    elif name_dataset == 'MKT':

        dim_amt = 1
        std = std_tr[dim_amt]
        mean = mean_tr[dim_amt]
        amt = x_input[dim_amt]
        c_a = 1
        amt_no_standarized = torch.mul(std, amt) + mean
        weighted_amt_no_standarized = amt_no_standarized * 0.00615
        pos_c_a = weighted_amt_no_standarized < c_a

        cost_c01 = torch.from_numpy(numpy.zeros(1)) + weighted_amt_no_standarized
        cost_c01[pos_c_a] = c_a
        cost_c10 = torch.from_numpy(numpy.zeros(1)) + c_a
        cost_c11 = torch.from_numpy(numpy.zeros(1)) + c_a
        cost_c00 = torch.from_numpy(numpy.zeros(1))

    elif name_dataset == 'TCC':

        dim_amt = 1
        std = std_tr[dim_amt]
        mean = mean_tr[dim_amt]
        amt = x_input[dim_amt]
        amt_no_standarized = torch.mul(std, amt) + mean

        cost_c01 = torch.mul(12.0, amt_no_standarized)
        cost_c10 = torch.mul(3.0, amt_no_standarized)
        cost_c11 = torch.from_numpy(numpy.zeros(1))
        cost_c00 = torch.from_numpy(numpy.zeros(1))

    else:
        print("[Error]: The indicated 'name_dataset' is not supported.")

    result = {'cost_c01': cost_c01,
              'cost_c10': cost_c10,
              'cost_c11': cost_c11,
              'cost_c00': cost_c00,
              'amt_no_standarized': amt_no_standarized}

    return result
