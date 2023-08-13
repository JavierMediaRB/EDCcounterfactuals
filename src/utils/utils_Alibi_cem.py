import numpy
import torch
import typing
import pandas
import tqdm
import matplotlib.pyplot
from src.cost_policy.cost_policy_for_datasets import cost_function_numpy
from src.benchmarks.Basic_MLP import Basic_MLP
from alibi.explainers import CEM


def search_counterfactuals_Alibi_CEM(name_dataset: str,
                                     NN_model: Basic_MLP,
                                     loaded_dataset: typing.Dict[str, object],
                                     params: typing.Dict[str, object],
                                     figure: bool = False,
                                     verbose: int = 1,
                                     small_test: bool = False
                                     ) -> typing.Dict[str, object]:
    """
    This function performs the pipeline process to generate a set of counterfactual samples for the CEM method.
    The process have the following consecutive steps:

        1. Load the needed params for the experiment
        2. Load the data from the variable 'loaded_dataset'
        3. Set the optimization restrictions (if aplicable)
        4. Compute an initial prediction on the test data with the 'NN_model' model to select only the
           samples that are originally predicted in the minoritary class.
        5. If the param 'small_test' is True, select just the first 2 samples to reduce the computational cost. This is
           usefull to debug the code if necessary.
        6. With a for loop perform the searching of counterfactuals sample by sample, by using the Alibi library.
        7. Compute some aditional variables like the costs of the founded counterfactual sample, and store them.

    [Warning]: Part of the function returns are probabilies. In this code, the probability is understood as the prob of
               a sample belonging to class 1, that is, the minority class.

    Args:
        name_dataset (str): The name of the dataset to access to the conf params
        NN_model (Basic_MLP): A NN model from the class Basic_MLP.
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
        figure (bool): In this function is useless. We keep this variable to maintain a common code structure for all
                       the counterfactual methods.
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
    """

    if verbose > 0:
        print('\n[Process] START counterfactual search')

    # ######## loaded dataset ######## #
    tensor_x_ts = loaded_dataset['tensor_x_ts']
    tensor_y_ts = loaded_dataset['tensor_y_ts']
    tensor_x_tr = loaded_dataset['tensor_x_tr']
    # tensor_y_tr = loaded_dataset['tensor_y_tr']
    # IR_tr = loaded_dataset['IR_tr']
    mean_tr = loaded_dataset['mean_tr']
    std_tr = loaded_dataset['std_tr']
    # ############################### #

    # ######## Params ######## #
    Xtr = tensor_x_tr.detach().numpy()
    # [mode]: 'PN' (pertinent negative) or 'PP' (pertinent positive)
    mode = params[name_dataset]['mode']
    # [Kappa]: minimum difference needed between the prediction probability for the perturbed instance on the
    # class predicted by the original instance and the max probability on the other classes
    # in order for the first loss term to be minimized
    kappa = params[name_dataset]['kappa']
    # [beta]: weight of the L1 loss term
    beta = params[name_dataset]['beta']
    # [c_init]: initial weight c of the loss term encouraging to predict a different class (PN) or
    # the same class (PP) for the perturbed instance compared to the original instance to be explained
    c_init = params[name_dataset]['c_init']
    # [c_steps]: nb of updates for c
    c_steps = params[name_dataset]['c_steps']
    # [max_iterations]: nb of iterations per value of c
    max_iterations = params[name_dataset]['max_iterations']
    # [clip]: gradient clipping
    min_clip = params[name_dataset]['min_clip']
    max_clip = params[name_dataset]['max_clip']
    clip = (min_clip, max_clip)
    # [lr_init]: initial learning rate
    lr_init = params[name_dataset]['lr_init']
    # [no_info_type]: 'median' or 'mean' value by feature supported.
    no_info_type = params[name_dataset]['no_info_type']
    # [shape]: instance shape
    shape = (1,) + Xtr.shape[1:]
    # [feature_range]: feature range for the perturbed instance
    # can be either a float or array of shape (1xfeatures)
    feature_range = (Xtr.min(axis=0).reshape(shape)-.1, Xtr.max(axis=0).reshape(shape)+.1)
    # ######################## #

    if verbose > 1:
        print('[Params Info] mode                  = ', mode)
        print('[Params Info] kappa                 = ', kappa)
        print('[Params Info] beta                  = ', beta)
        print('[Params Info] c_init                = ', c_init)
        print('[Params Info] c_steps               = ', c_steps)
        print('[Params Info] max_iterations        = ', max_iterations)
        print('[Params Info] min_clip              = ', min_clip)
        print('[Params Info] max_clip              = ', max_clip)
        print('[Params Info] lr_init               = ', lr_init)
        print('[Params Info] no_info_type          = ', no_info_type)
        print('[Params Info] NN                    = ', params[name_dataset]['NN'])

    # ########### Initialize and fit CEM explainer ########### #
    cem = CEM(predict=NN_model.forward_2_outputs_numpy,
              mode=mode,
              shape=shape,
              kappa=kappa,
              beta=beta,
              feature_range=feature_range,
              max_iterations=max_iterations,
              c_init=c_init,
              c_steps=c_steps,
              learning_rate_init=lr_init,
              clip=clip)
    # we need to define what feature values contain the least info wrt predictions here we will naively assume that
    # the feature-wise median contains no info; domain knowledge helps!
    cem.fit(train_data=Xtr, no_info_type=no_info_type)
    # ######################################################## #

    key_counter = "counter"
    key_orig = "orig"

    # ######### Initial prediction ######### #
    y_pred_orig = NN_model.predict_class(x_input=tensor_x_ts).astype(int)
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

    not_found = 0
    counterfactuals = Xts_filtered.copy() * 0.0
    for index_sample in tqdm.tqdm(range(n_samples)):

        sample_to_explain = Xts_filtered[index_sample].reshape((1,) + Xts_filtered[index_sample].shape)
        explanation = cem.explain(sample_to_explain)

        # #### Record results #### #
        if explanation.PN_pred is not None:
            counterfactuals[index_sample, :] = explanation.PN[0]
        else:
            counterfactuals[index_sample, :] = Xts_filtered[index_sample, :]
            not_found = not_found + 1
        # ######################### #

    # #### Compute the model outputs and predictions for original anc counterfactual samples #### #
    list_counter_samples = counterfactuals
    tensor_counter_samples = torch.from_numpy(list_counter_samples)

    o_pred_counter = NN_model.forward(x_input=tensor_counter_samples)
    y_pred_counter = NN_model.predict_class(x_input=torch.from_numpy(list_counter_samples).float())
    y_pred_counter = y_pred_counter.astype(int)
    o_pred_orig = NN_model.forward(x_input=tensor_x_ts_filtered)
    y_pred_original = NN_model.predict_class(x_input=tensor_x_ts_filtered).astype(int)
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

    # ########### Plot counterfactuals resume ########### #
    label_df = pandas.DataFrame(numpy.array(y_pred_counter), columns=['counterfactual_label'])
    label_df['original_label'] = numpy.array(y_pred_original)

    o_pred_count_np = NN_model.forward_2_outputs(x_input=tensor_counter_samples).detach().numpy()
    found_counterfactual = (numpy.array(o_pred_count_np)[:, 1] < numpy.array(o_pred_count_np)[:, 0])
    sample_near_to_frontier = (numpy.abs(numpy.array(o_pred_count_np)[:, 1] - numpy.array(o_pred_count_np)[:, 0]) < 0.1)
    found_min_counter = (found_counterfactual * sample_near_to_frontier).sum()

    found = len(numpy.where(label_df['counterfactual_label'] != label_df['original_label'])[0])
    not_found = n_samples - found
    print(f"Not found counter: {not_found}/{n_samples}")
    print(f"Found counter:     {found}/{n_samples}. From those {found_min_counter} have a minimal distance")

    diff_df = pandas.DataFrame(Xts_filtered - list_counter_samples).round(4).abs()
    diferecia_promedio = diff_df.mean()[diff_df.mean() != 0]
    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(diferecia_promedio.index, diferecia_promedio.values)
    matplotlib.pyplot.title('Mean Diff for each variable:')
    matplotlib.pyplot.show()

    matplotlib.pyplot.figure()
    matplotlib.pyplot.hist(numpy.array(o_pred_count_np)[:, 0], bins=20)
    matplotlib.pyplot.title('Counter model outputs: \nProb=1 (far away counter) | Prob=0 (No counter)')
    matplotlib.pyplot.show()
    # ################################################### #

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
               }
    # ####################### #

    if verbose > 0:
        print('[Process] DONE counterfactual search')
    return results
