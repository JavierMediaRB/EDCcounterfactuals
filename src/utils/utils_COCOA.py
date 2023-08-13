import numpy
import torch
import tqdm
import typing
import pandas
from src.cost_policy.cost_policy_for_datasets import cost_function_numpy, cost_function_torch
from src.cocoa_method.cocoa_NN import cocoa_NN


def search_counterfactuals(name_dataset: str,
                           cocoa_NN_model: cocoa_NN,
                           loaded_dataset: typing.Dict[str, object],
                           params: typing.Dict[str, object],
                           figure: bool = False,
                           verbose: int = 1,
                           small_test: bool = False
                           ) -> typing.Dict[str, object]:
    """
    This function performs the pipeline process to generate a set of counterfactual samples for the COCOA method.
    The process have the following consecutive steps:

        1. Load the needed params for the experiment
        2. Load the data from the variable 'loaded_dataset'
        3. Set the optimization restrictions (if aplicable)
        4. Compute an initial prediction on the test data with the 'explainable_NN_model' model to select only the
           samples that are originally predicted in the minoritary class.
        5. If the param 'small_test' is True, select just the first 2 samples to reduce the computational cost. This is
           usefull to debug the code if necessary.
        6. With a for loop perform the searching of counterfactuals sample by sample.
        7. Compute some aditional variables like the costs of the founded counterfactual sample, and store them.

    [Warning]: Part of the function returns are probabilies. In this code, the probability is understood as the prob of
               a sample belonging to class 1, that is, the minority class.

    Args:
        name_dataset (str): The name of the dataset to access to the conf params
        cocoa_NN_model (explainable_NN): A NN model from the class explainable_NN.
        loaded_dataset (typing.Dict[str, object]): A dictionary containing the dataset data with the variables:
                                                     - 'tensor_x_ts': tensor_x_ts,
                                                     - 'tensor_y_ts': tensor_y_ts,
                                                     - 'tensor_Cs01': tensor_Cs01,
                                                     - 'tensor_Cs10': tensor_Cs10,
                                                     - 'tensor_Cs11': tensor_Cs11,
                                                     - 'tensor_Cs00': tensor_Cs00,
                                                     - 'tensor_x_tr': tensor_x_tr,
                                                     - 'tensor_y_tr': tensor_y_tr,
                                                     - 'tensor_Cr01': tensor_Cr01,
                                                     - 'tensor_Cr10': tensor_Cr10,
                                                     - 'tensor_Cr11': tensor_Cr11,
                                                     - 'tensor_Cr00': tensor_Cr00,
                                                     - 'IR_tr': IR_tr,
                                                     - 'mean_tr': mean_tr,
                                                     - 'std_tr': std_tr,
        params (typing.Dict[str, object]): Contais the parameters that might be needed for the process. Defatul to {}.
        figure (bool): If True displays a set of informative plots for the counterfactual search for each sample.
                       Defaults to False. [WARNING]: If the number of samples is large this pots could saturate
                       your cpu resources.
        verbose (int): Depending on the number diferent level information is printed.
                       [0: no prints; 1: high level information; 2: full detail]. Default to 1.
        small_test (bool): If True, the samples used to compute the counterfactuals are reduced to 2. In orther to
                           reduce the computational cost to debug the code. Defatul to False.

    Returns:
        original_samples (numpy.ndarray): The Xts data only for the samples originally classified as the minoritary
                                          class.
        lr_regularization (float): The regularization used for the optimization process.
        o_original_samples (torch.Tensor): The output of the model for the original samples.
        y_pred_original_samples (numpy.ndarray): The predicted class of the model for the original samples.
        counterfactual_samples (numpy.ndarray): Contains the computed counterfactual samples.
        o_counterfactual_samples (torch.Tensor): The output of the model for the counterfactual samples.
        y_pred_counterfactual_samples (numpy.ndarray): The predicted class of the model for the counterfactual samples.
        cost_matrix (pandas.DataFrame): Contains the example-dependent costs for the original and counterfactual samples
        y_true_orignal_samples (numpy.ndarray): The Yts data only for the samples originally classified as the
                                                minoritary class.
        success_vars (typing.Dict[str, object]): A Dictionary with relevant variables needed to compute the KPIs to
                                                 measure the performance of the counterfactuals.
            list_discriminant_final (typing.List[float]): The final value of the Bayes discriminant.
            list_g_final (typing.List[float]): The final value of the term 'g' of the Bayes discriminant.
            list_f_final (typing.List[float]): The final value of the term 'f' of the Bayes discriminant.
            list_g_inicial (typing.List[float]): The initial value of the term 'g' of the Bayes discriminant.
            list_f_inicial (typing.List[float]): The initial value of the term 'f' of the Bayes discriminant.
    """
    if verbose > 0:
        print('\n[Process] START counterfactual search')
    # ######## Params ######## #
    activate_movement_in_amount = params[name_dataset]['activate_movement_in_amount']
    epochs = params[name_dataset]['epochs']
    threshold_nu = params[name_dataset]['threshold_nu']
    convergence_criteria = params[name_dataset]['convergence_criteria']
    convergence_criteria_reg = params[name_dataset]['convergence_criteria_reg']
    regularzation_frontier = params[name_dataset]['regularzation_frontier']
    lr_for_regularization = params[name_dataset]['lr_for_regularization']
    lr_for_discriminant = params[name_dataset]['lr_for_discriminant']
    zero_grad_regularization = params[name_dataset]['zero_grad_regularization']
    RB = params[name_dataset]['RB']
    # ######################## #

    # ######## loaded dataset ######## #
    tensor_x_ts = loaded_dataset['tensor_x_ts']
    tensor_y_ts = loaded_dataset['tensor_y_ts']
    tensor_x_tr = loaded_dataset['tensor_x_tr']
    IR_tr = loaded_dataset['IR_tr']
    mean_tr = loaded_dataset['mean_tr']
    std_tr = loaded_dataset['std_tr']
    # ############################### #

    # ########## Restricciones de la optimizacion ########## #
    dimensiones = numpy.ones(tensor_x_tr.shape[1])
    dimensiones[params[name_dataset]['pos_dimension_amount']] = activate_movement_in_amount
    # ###################################################### #

    if verbose > 1:
        print('[Params Info] activate_movement_in_amount   = ', activate_movement_in_amount)
        print('[Params Info] epochs                        = ', epochs)
        print('[Params Info] threshold_nu                  = ', threshold_nu)
        print('[Params Info] convergence_criteria          = ', convergence_criteria)
        print('[Params Info] convergence_criteria_reg      = ', convergence_criteria_reg)
        print('[Params Info] regularzation_frontier        = ', regularzation_frontier)
        print('[Params Info] lr_for_regularization         = ', lr_for_regularization)
        print('[Params Info] lr_for_discriminant           = ', lr_for_discriminant)
        print('[Params Info] zero_grad_regularization      = ', zero_grad_regularization)
        print('[Params Info] RB                            = ', RB)

    key_counter = "counter"
    key_orig = "orig"

    # ######### Initial prediction ######### #
    y_pred_orig = cocoa_NN_model.predict_class(x_input=tensor_x_ts).astype(int)
    tensor_x_ts_filtered = tensor_x_ts
    Xts_filtered = tensor_x_ts.detach().numpy().copy()
    Yts_filtered = tensor_y_ts.detach().numpy().copy()
    if params['filter_original_minoritary_predictions']:

        tensor_x_ts_filtered = tensor_x_ts_filtered[numpy.where(y_pred_orig == 1)[0], :]
        Xts_filtered = Xts_filtered[numpy.where(y_pred_orig == 1)[0], :]
        Yts_filtered = Yts_filtered[numpy.where(y_pred_orig == 1)[0]]
    # ###################################### #

    n_samples = tensor_x_ts_filtered.shape[0]
    if small_test:
        # Reduce the test sample to 2 sample length. Warning, this will not compute the counterfactuals
        # on 2 samples for debugging processes.
        n_samples = 2
        Yts_filtered = Yts_filtered[:n_samples]
        Xts_filtered = Xts_filtered[:n_samples, :]
        tensor_x_ts_filtered = tensor_x_ts_filtered[:n_samples, :]
    list_orgi_samples = numpy.array(tensor_x_ts[0:0+n_samples, :])

    cost_matrix = pandas.DataFrame(numpy.zeros((n_samples, 8)), columns=[f"{key_counter}_c00",
                                                                         f"{key_counter}_c11",
                                                                         f"{key_counter}_c01",
                                                                         f"{key_counter}_c10",
                                                                         f"{key_orig}_c00",
                                                                         f"{key_orig}_c11",
                                                                         f"{key_orig}_c01",
                                                                         f"{key_orig}_c10"])

    list_counter_samples = []
    list_final_epoch = []
    list_discriminant_final = []
    list_g_final = []
    list_f_final = []
    list_g_initial = []
    list_f_initial = []

    for index_sample in tqdm.tqdm(range(n_samples)):
        counter_search_result = cocoa_NN_model.explain_sample(input_sample=tensor_x_ts_filtered[index_sample, :],
                                                              mean_tr=mean_tr,
                                                              std_tr=std_tr,
                                                              lr_for_regularization=lr_for_regularization,
                                                              lr_for_discriminant=lr_for_discriminant,
                                                              epochs=epochs,
                                                              cost_policy_function=cost_function_torch,
                                                              RB=RB,
                                                              IR=IR_tr,
                                                              dimensions_restrictions=dimensiones,
                                                              threshold_nu=threshold_nu,
                                                              convergence_criteria=convergence_criteria,
                                                              convergence_criteria_reg=convergence_criteria_reg,
                                                              regularzation_frontier=regularzation_frontier,
                                                              extra_params=params[name_dataset]['extra_params'],
                                                              zero_grad_regularization=zero_grad_regularization,
                                                              figure=figure,
                                                              verbose=verbose)

        contrafactual_sample = counter_search_result['contrafactual_sample']
        final_epoch = counter_search_result['final_epoch']
        discriminant_list = counter_search_result['discriminant_list']
        g_list = counter_search_result['g_list']
        f_list = counter_search_result['f_list']

        # #### Append results #### #
        discriminant_final = discriminant_list[final_epoch]
        g_final = g_list[len(g_list) - 1]
        f_final = f_list[len(f_list) - 1]
        g_inicial = g_list[0]
        f_inicial = f_list[0]

        list_counter_samples.append(contrafactual_sample)
        list_final_epoch.append(final_epoch)
        list_discriminant_final.append(discriminant_final)
        list_g_final.append(g_final)
        list_f_final.append(f_final)
        list_g_initial.append(g_inicial)
        list_f_initial.append(f_inicial)
        # ######################## #

    # #### Compute the model outputs and predictions for original anc counterfactual samples #### #
    list_counter_samples = numpy.array(list_counter_samples)
    tensor_counter_samples = torch.from_numpy(list_counter_samples)

    o_pred_counter = cocoa_NN_model.eval_samples(x_input=tensor_counter_samples)
    y_pred_counter = cocoa_NN_model.predict_class(x_input=torch.from_numpy(list_counter_samples).float())
    y_pred_counter = y_pred_counter.astype(int)
    o_pred_orig = cocoa_NN_model.eval_samples(x_input=tensor_x_ts_filtered)
    y_pred_original = cocoa_NN_model.predict_class(x_input=tensor_x_ts_filtered).astype(int)
    # ########################################################################################### #

    # #### Compute the costs of the original and counterfactual sample #### #
    # #### counterfactual cost matrix #### #
    counterfactual_cost_policy_result = cost_function_numpy(name_dataset=name_dataset,
                                                             x_input=list_counter_samples,
                                                             mean_tr=mean_tr,
                                                             std_tr=std_tr,
                                                             extra_params=params[name_dataset]['extra_params'])
    cost_c01 = counterfactual_cost_policy_result['cost_c01']
    cost_c10 = counterfactual_cost_policy_result['cost_c10']
    cost_c11 = counterfactual_cost_policy_result['cost_c11']
    cost_c00 = counterfactual_cost_policy_result['cost_c00']
    lr_reg = counterfactual_cost_policy_result['amt_no_standarized']

    cost_matrix[f"{key_counter}_c10"] = cost_c00
    cost_matrix[f"{key_counter}_c11"] = cost_c11
    cost_matrix[f"{key_counter}_c01"] = cost_c01
    cost_matrix[f"{key_counter}_c10"] = cost_c10
    # #################################### #

    # #### original cost matrix #### #
    original_cost_policy_result = cost_function_numpy(name_dataset=name_dataset,
                                                       x_input=list_orgi_samples,
                                                       mean_tr=mean_tr,
                                                       std_tr=std_tr,
                                                       extra_params=params[name_dataset]['extra_params'])
    cost_c01 = original_cost_policy_result['cost_c01']
    cost_c10 = original_cost_policy_result['cost_c10']
    cost_c11 = original_cost_policy_result['cost_c11']
    cost_c00 = original_cost_policy_result['cost_c00']
    lr_reg = original_cost_policy_result['amt_no_standarized']

    cost_matrix[f"{key_orig}_c10"] = cost_c00
    cost_matrix[f"{key_orig}_c11"] = cost_c11
    cost_matrix[f"{key_orig}_c01"] = cost_c01
    cost_matrix[f"{key_orig}_c10"] = cost_c10
    # ############################## #
    # ##################################################################### #

    # ####### results ####### #
    results = {'original_samples': Xts_filtered,
               'lr_regularization': lr_reg,
               'o_original_samples': o_pred_orig,
               'y_pred_original_samples': y_pred_original,
               'counterfactual_samples': list_counter_samples,
               'o_counterfactual_samples': o_pred_counter,
               'y_pred_counterfactual_samples': y_pred_counter,
               'cost_matrix': cost_matrix,
               'y_true_orignal_samples': Yts_filtered,
               'success_vars': {'list_d_final': list_discriminant_final,
                                'list_g_final': list_g_final,
                                'list_f_final': list_f_final,
                                'list_g_inicial': list_g_initial,
                                'list_f_inicial': list_f_initial}}
    # ####################### #

    if verbose > 0:
        print('[Process] DONE counterfactual search')
    return results
