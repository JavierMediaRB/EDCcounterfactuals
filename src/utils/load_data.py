import numpy
import torch
import tqdm
import typing
import pandas
from src.cost_policy.cost_policy_for_datasets import funcion_costes_numpy, funcion_costes_torch
from src.proposed_method.explainable_NN import explainable_NN


def load_data(name_dataset: str,
              path_data: str,
              verbose: int = 1) -> typing.Dict[str, object]:
    """
    This function load the train/test data and the cost of the original samples (not the counterfactuals)
    for the selected dataset. Also this function transform the data into torch tensor format.

    [Warning]: The input data, MUST have label '-1' for the mayority class, and '+1' for the minority class.

    Args:
        name_dataset (str): The name of the dataset to load.
        path_data (str): The path where to find the data.
        verbose (int): Depending on the number diferent level information is printed.
                       [0: no prints; 1: high level information; 2: full detail]. Default to 1.

    Returns:
        tensor_x_ts (torch.Tensor): The X matrix for the test set.
        tensor_y_ts (torch.Tensor): The Y vector for the test set.
        tensor_Cs01 (torch.Tensor): The cost_01 vector for the test set.
        tensor_Cs10 (torch.Tensor): The cost_10 vector for the test set.
        tensor_Cs11 (torch.Tensor): The cost_11 vector for the test set.
        tensor_Cs00 (torch.Tensor): The cost_00 vector for the test set.
        tensor_x_tr (torch.Tensor): The X matrix for the train set.
        tensor_y_tr (torch.Tensor): The Y vector for the train set.
        tensor_Cr01 (torch.Tensor): The cost_01 vector for the train set.
        tensor_Cr10 (torch.Tensor): The cost_10 vector for the train set.
        tensor_Cr11 (torch.Tensor): The cost_11 vector for the train set.
        tensor_Cr00 (torch.Tensor): The cost_00 vector for the train set.
        IR_tr (float): The imbalance of the train set.
        mean_tr (numpy.ndarray): The mean of the original (not standard yet) train set used to standarize the data.
        std_tr (numpy.ndarray): The std of the original (not standard yet) train set used to standarize the data.
    """
    # ######## PATHS ######## #
    path_dataset = f"{path_data}{name_dataset}_dataset/"
    file_load_data = "train_test_data.npz"
    file_load_standarizacion = "mean_std_of_standarization.npz"
    # ####################### #

    # ########### LOADING THE DATA ########### #
    if verbose > 0:
        print('\n[Process] START data loading')
    data = numpy.load(path_dataset + file_load_data)
    standard = numpy.load(path_dataset + file_load_standarizacion)

    Ytr = data['Datos_Y_tr']
    Xtr = data['Datos_X_tr']
    Ctr = data['Cost_XY_tr']  # Cost_XY_tr[cost_FP, cost_FN, cost_TP, cost_TN]

    Yts = data['Datos_Y_ts']
    Xts = data['Datos_X_ts']
    Cts = data['Cost_XY_ts']  # Cost_XY_ts[cost_FP, cost_FN, cost_TP, cost_TN]

    mean_tr = standard['mean_tr']
    std_tr = standard['std_tr']

    Cr10 = Ctr[:, 0]
    Cr01 = Ctr[:, 1]
    Cr11 = Ctr[:, 2]
    Cr00 = Ctr[:, 3]

    Cs10 = Cts[:, 0]
    Cs01 = Cts[:, 1]
    Cs11 = Cts[:, 2]
    Cs00 = Cts[:, 3]

    tensor_x_ts = torch.from_numpy(Xts)
    tensor_y_ts = torch.from_numpy(Yts)
    tensor_Cs01 = torch.from_numpy(Cs01)
    tensor_Cs10 = torch.from_numpy(Cs10)
    tensor_Cs11 = torch.from_numpy(Cs11)
    tensor_Cs00 = torch.from_numpy(Cs00)

    tensor_x_tr = torch.from_numpy(Xtr)
    tensor_y_tr = torch.from_numpy(Ytr)
    tensor_Cr01 = torch.from_numpy(Cr01)
    tensor_Cr10 = torch.from_numpy(Cr10)
    tensor_Cr11 = torch.from_numpy(Cr11)
    tensor_Cr00 = torch.from_numpy(Cr00)

    IR_tr = len(numpy.where(Ytr == -1)[0])/len(numpy.where(Ytr == 1)[0])
    IR_ts = len(numpy.where(Yts == -1)[0])/len(numpy.where(Yts == 1)[0])
    # ######################################## #

    if verbose > 0:
        eps = 0.0000000000000001
        P0tr = len(numpy.where(Ytr == -1)[0])
        P1tr = len(numpy.where(Ytr == 1)[0])
        P0ts = len(numpy.where(Yts == -1)[0])
        P1ts = len(numpy.where(Yts == 1)[0])
        Qc_tr = numpy.round(numpy.mean((Cr01 - Cr00) / (Cr10 - Cr11 + eps)), 1)
        Qc_ts = numpy.round(numpy.mean((Cs01 - Cs00) / (Cs10 - Cs11 + eps)), 1)
        tot_c01 = numpy.concatenate([Cs01, Cr01])
        tot_c00 = numpy.concatenate([Cs00, Cr00])
        tot_c10 = numpy.concatenate([Cs10, Cr10])
        tot_c11 = numpy.concatenate([Cs11, Cr11])
        Qc_tot = numpy.round(numpy.mean((tot_c01 - tot_c00) / (tot_c10 - tot_c11 + eps)), 1)
        tot_ir = numpy.round((P0tr + P0ts)/(P1tr + P1ts), 1)
        tr_ir = numpy.round(IR_tr, 1)
        ts_ir = numpy.round(IR_ts, 1)
        len_tot = len(Ytr) + len(Yts)
        n_dims = len(Xtr[1, :])

        tr_prop = int(numpy.round((len(Ytr) / len_tot)*100))
        ts_prop = int(numpy.round((len(Yts) / len_tot)*100))

        print(f"[Data Info] Imbalance ratio (IR) = P0/P1 = N0/N1 || total: {tot_ir}\t || tr: {tr_ir}\t || ts: {ts_ir}")
        print(f"[Data Info] mean(1/Qc = (c01 - c11)/(c10 - c00)) || total: {Qc_tot}\t || tr: {Qc_tr}\t || ts: {Qc_ts}")
        print(f"[Data Info] shape: {n_dims} dims for {len_tot} samples || Train/Test = {tr_prop}/{ts_prop} [%]")

    if verbose > 0:
        print('[Process] DONE data loading')

    results = {'tensor_x_ts': tensor_x_ts,
               'tensor_y_ts': tensor_y_ts,
               'tensor_Cs01': tensor_Cs01,
               'tensor_Cs10': tensor_Cs10,
               'tensor_Cs11': tensor_Cs11,
               'tensor_Cs00': tensor_Cs00,
               'tensor_x_tr': tensor_x_tr,
               'tensor_y_tr': tensor_y_tr,
               'tensor_Cr01': tensor_Cr01,
               'tensor_Cr10': tensor_Cr10,
               'tensor_Cr11': tensor_Cr11,
               'tensor_Cr00': tensor_Cr00,
               'IR_tr': IR_tr,
               'mean_tr': mean_tr,
               'std_tr': std_tr,
               }

    return results


def search_counterfactuals(name_dataset: str,
                           explainable_NN_model: explainable_NN,
                           loaded_dataset: typing.Dict[str, object],
                           params: typing.Dict[str, object],
                           figure: bool = False,
                           verbose: int = 1,
                           ) -> typing.Dict[str, object]:
    """
    ... TODO...

    [Warning]: Part of the function returns are probabilies. In this code, the probability is understood as the prob of
               a sample belonging to class 1, that is, the minority class.
    Args:
        name_dataset (str):
        loaded_dataset (typing.Dict[str, object]):
        params (typing.Dict[str, object]):
        figure (bool):
        verbose (int): Depending on the number diferent level information is printed.
                       [0: no prints; 1: high level information; 2: full detail]. Default to 1.

    Returns:
        Xts_filtered ():
        lr_reg ():
        o_pred_orig ():
        y_pred_original ():
        list_counter_samples ()
        o_pred_counter ():
        y_pred_counter ():
        cost_matrix ():
        Yts_filtered ():
        success_vars ():
            list_discriminant_final ():
            list_g_final ():
            list_f_final ():
            list_g_inicial ():
            list_f_inicial ():

    """
    if verbose > 0:
        print('\n[Process] START counterfactual search')
    # ######## Params ######## #
    activate_movement_in_amount = params[name_dataset]['activate_movement_in_amount']
    epochs = params[name_dataset]['epochs']
    margen_nu = params[name_dataset]['margen_nu']
    convergence_criteria = params[name_dataset]['convergence_criteria']
    convergence_criteria_reg = params[name_dataset]['convergence_criteria_reg']
    regularzation_frontier = params[name_dataset]['regularzation_frontier']
    lr_for_regularization = params[name_dataset]['lr_for_regularization']
    lr_for_discriminant = params[name_dataset]['lr_for_discriminant']
    zero_grad_regularization = params[name_dataset]['zero_grad_regularization']
    RB = params[name_dataset]['RB']
    # ######################## #

    # ######## loaded dataset ######## #
    # TODO: No se si me gusta, ya que estoy obligando a esta forma de pasar el dato...
    tensor_x_ts = loaded_dataset['tensor_x_ts']
    tensor_y_ts = loaded_dataset['tensor_y_ts']
    tensor_x_tr = loaded_dataset['tensor_x_tr']
    # tensor_y_tr = loaded_dataset['tensor_y_tr']
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
        print('[Params Info] margen_nu                     = ', margen_nu)
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
    y_pred_orig = explainable_NN_model.predict_class(x_input=tensor_x_ts).astype(int)
    tensor_x_ts_filtered = tensor_x_ts
    Xts_filtered = tensor_x_ts.detach().numpy().copy()
    Yts_filtered = tensor_y_ts.detach().numpy().copy()
    if params['filter_original_minoritary_predictions']:

        tensor_x_ts_filtered = tensor_x_ts_filtered[numpy.where(y_pred_orig == 1)[0], :]
        Xts_filtered = Xts_filtered[numpy.where(y_pred_orig == 1)[0], :]
        Yts_filtered = Yts_filtered[numpy.where(y_pred_orig == 1)[0]]
    # ###################################### #

    n_samples = tensor_x_ts_filtered.shape[0]
    # n_samples = 2
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
        counter_search_result = explainable_NN_model.explica_muestra(input_sample=tensor_x_ts_filtered[index_sample, :],
                                                                     mean_tr=mean_tr,
                                                                     std_tr=std_tr,
                                                                     lr_for_regularization=lr_for_regularization,
                                                                     lr_for_discriminant=lr_for_discriminant,
                                                                     epochs=epochs,
                                                                     cost_policy_function=funcion_costes_torch,
                                                                     RB=RB,
                                                                     IR=IR_tr,
                                                                     dimensions_restrictions=dimensiones,
                                                                     margen_nu=margen_nu,
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

    o_pred_counter = explainable_NN_model.eval_samples(input=tensor_counter_samples)
    y_pred_counter = explainable_NN_model.predict_class(x_input=torch.from_numpy(list_counter_samples).float())
    y_pred_counter = y_pred_counter.astype(int)
    o_pred_orig = explainable_NN_model.eval_samples(input=tensor_x_ts_filtered)
    y_pred_original = explainable_NN_model.predict_class(x_input=tensor_x_ts_filtered).astype(int)
    # ########################################################################################### #

    # #### Compute the costs of the original and counterfactual sample #### #
    # #### counterfactual cost matrix #### #
    counterfactual_cost_policy_result = funcion_costes_numpy(name_dataset=name_dataset,
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
    original_cost_policy_result = funcion_costes_numpy(name_dataset=name_dataset,
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
