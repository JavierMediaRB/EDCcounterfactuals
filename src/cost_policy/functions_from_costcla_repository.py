"""
This file contains a copy of some of the cost functions needed for the CS1 and CS2 cost policy formulation. This
functions are an exact copy of the ones in the CostCla repository (https://pypi.org/project/costcla/). The only aim to
copy the functions instead of just importing them from the costcla library is to preserve the reproducibility of the
experiments of this paper in the case of a future change in the code of the costcla repository (in case than anyone try
to reproduce them with a diferent version of the costcla repository)

References
    ----------
    .. [1] A. Correa Bahnsen, D.Aouada, B, Ottersten,
           "Example-Dependent Cost-Sensitive Logistic Regression for Credit Scoring",
           in Proceedings of the International Conference on Machine Learning and Applications,
           , 2014.
"""

import numpy
import torch


def _creditscoring_costmat(income, debt, pi_1, cost_mat_parameters):
    """ Private function to calculate the cost matrix of credit scoring models.

    Parameters
    ----------
    income : array of shape = [n_samples]
        Monthly income of each example

    debt : array of shape = [n_samples]
        Debt ratio each example

    pi_1 : float
        Percentage of positives in the training set

    References
    ----------
    .. [1] A. Correa Bahnsen, D.Aouada, B, Ottersten,
           "Example-Dependent Cost-Sensitive Logistic Regression for Credit Scoring",
           in Proceedings of the International Conference on Machine Learning and Applications,
           , 2014.

    Returns
    -------
    cost_mat : array-like of shape = [n_samples, 4]
        Cost matrix of the classification problem
        Where the columns represents the costs of: false positives, false negatives,
        true positives and true negatives, for each example.
    """
    def calculate_a(cl_i, int_, n_term):
        """ Private function """
        return cl_i * ((int_ * (1 + int_) ** n_term) / ((1 + int_) ** n_term - 1))

    def calculate_pv(a, int_, n_term):
        """ Private function """
        return a / int_ * (1 - 1 / (1 + int_) ** n_term)

    # Calculate credit line Cl
    def calculate_cl(k, inc_i, cl_max, debt_i, int_r, n_term):
        """ Private function """
        cl_k = k * inc_i
        A = calculate_a(cl_k, int_r, n_term)
        Cl_debt = calculate_pv(inc_i * min(A / inc_i, 1 - debt_i), int_r, n_term)
        return min(cl_k, cl_max, Cl_debt)

    # calculate costs
    def calculate_cost_fn(cl_i, lgd):
        return cl_i * lgd

    def calculate_cost_fp(cl_i, int_r, n_term, int_cf, pi_1, lgd, cl_avg):
        a = calculate_a(cl_i, int_r, n_term)
        pv = calculate_pv(a, int_cf, n_term)
        r = pv - cl_i
        r_avg = calculate_pv(calculate_a(cl_avg, int_r, n_term), int_cf, n_term) - cl_avg
        cost_fp = r - (1 - pi_1) * r_avg + pi_1 * calculate_cost_fn(cl_avg, lgd)

        return max(0, cost_fp)

    v_calculate_cost_fp = numpy.vectorize(calculate_cost_fp)
    v_calculate_cost_fn = numpy.vectorize(calculate_cost_fn)
    v_calculate_cl = numpy.vectorize(calculate_cl)

    # Parameters
    k = cost_mat_parameters['k']
    int_r = cost_mat_parameters['int_r']
    n_term = cost_mat_parameters['n_term']
    int_cf = cost_mat_parameters['int_cf']
    lgd = cost_mat_parameters['lgd']
    cl_max = cost_mat_parameters['cl_max']

    cl = v_calculate_cl(k, income, cl_max, debt, int_r, n_term)
    if cost_mat_parameters['cl_avg'] is None:
        cl_avg = cl.mean()
        cost_mat_parameters['cl_avg'] = cl_avg
    else:
        cl_avg = cost_mat_parameters['cl_avg']

    n_samples = income.shape[0]
    cost_mat = numpy.zeros((n_samples, 4))  # cost_mat[FP,FN,TP,TN]
    cost_mat[:, 0] = v_calculate_cost_fp(cl, int_r, n_term, int_cf, pi_1, lgd, cl_avg)
    cost_mat[:, 1] = v_calculate_cost_fn(cl, lgd)
    cost_mat[:, 2] = 0.0
    cost_mat[:, 3] = 0.0

    return cost_mat


def _creditscoring_costmat_pytorch(income, debt, pi_1, cost_mat_parameters):
    """ This function is an adaptation of the original function in order to work with torch tensors

    Private function to calculate the cost matrix of credit scoring models.

    Parameters
    ----------
    income : array of shape = [n_samples]
        Monthly income of each example

    debt : array of shape = [n_samples]
        Debt ratio each example

    pi_1 : float
        Percentage of positives in the training set

    References
    ----------
    .. [1] A. Correa Bahnsen, D.Aouada, B, Ottersten,
           "Example-Dependent Cost-Sensitive Logistic Regression for Credit Scoring",
           in Proceedings of the International Conference on Machine Learning and Applications,
           , 2014.

    Returns
    -------
    cost_mat : array-like of shape = [n_samples, 4]
        Cost matrix of the classification problem
        Where the columns represents the costs of: false positives, false negatives,
        true positives and true negatives, for each example.
    """
    def calculate_a(cl_i, int_, n_term):
        """ Private function """
        return cl_i * ((int_ * (1 + int_) ** n_term) / ((1 + int_) ** n_term - 1))

    def calculate_pv(a, int_, n_term):
        """ Private function """
        return a / int_ * (1 - 1 / (1 + int_) ** n_term)

    # Calculate credit line Cl
    def calculate_cl(k, inc_i, cl_max, debt_i, int_r, n_term):
        """ Private function """
        cl_k = k * inc_i
        A = calculate_a(cl_k, int_r, n_term)
        Cl_debt = calculate_pv(inc_i * torch.min(A / inc_i, 1 - debt_i), int_r, n_term)
        result = torch.min(cl_k, torch.tensor(cl_max, dtype=torch.float64))
        result = torch.min(result, Cl_debt)
        return result

    # calculate costs
    def calculate_cost_fn(cl_i, lgd):
        return cl_i * lgd

    def calculate_cost_fp(cl_i, int_r, n_term, int_cf, pi_1, lgd, cl_avg):
        a = calculate_a(cl_i, int_r, n_term)
        pv = calculate_pv(a, int_cf, n_term)
        r = pv - cl_i
        r_avg = calculate_pv(calculate_a(cl_avg, int_r, n_term), int_cf, n_term) - cl_avg
        cost_fp = r - (1 - pi_1) * r_avg + pi_1 * calculate_cost_fn(cl_avg, lgd)

        return torch.max(torch.tensor(0.0, dtype=torch.float64), cost_fp)

    v_calculate_cost_fp = calculate_cost_fp
    v_calculate_cost_fn = calculate_cost_fn
    v_calculate_cl = calculate_cl

    # Parameters
    k = cost_mat_parameters['k']
    int_r = cost_mat_parameters['int_r']
    n_term = cost_mat_parameters['n_term']
    int_cf = cost_mat_parameters['int_cf']
    lgd = cost_mat_parameters['lgd']
    cl_max = cost_mat_parameters['cl_max']

    cl = v_calculate_cl(k, income, cl_max, debt, int_r, n_term)
    cl_avg = cost_mat_parameters['cl_avg']

    tensor_C10 = v_calculate_cost_fp(cl, int_r, n_term, int_cf, pi_1, lgd, cl_avg)
    tensor_C01 = v_calculate_cost_fn(cl, lgd)
    tensor_C11 = torch.from_numpy(numpy.zeros(1))
    tensor_C00 = torch.from_numpy(numpy.zeros(1))
    return tensor_C10, tensor_C01, tensor_C11, tensor_C00
