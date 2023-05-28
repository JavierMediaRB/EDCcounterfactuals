import numpy
import torch
import typing
import pandas
import tqdm
import matplotlib.pyplot
from src.cost_policy.cost_policy_for_datasets import funcion_costes_numpy
from src.benchmarks.Basic_MLP import Basic_MLP
from alibi.explainers import CounterfactualProto


def search_counterfactuals_Alibi_Proto(name_dataset: str,
                                       NN_model: Basic_MLP,
                                       loaded_dataset: typing.Dict[str, object],
                                       params: typing.Dict[str, object],
                                       figure: bool = False,
                                       verbose: int = 1,
                                       small_test: bool = False
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
        small_test (bool): If True, the test data will be croped in order to reduce the computational cost of the
                           experiments for debugging processes.

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

    # ######## loaded dataset ######## #
    # TODO: No se si me gusta, ya que estoy obligando a esta forma de pasar el dato...
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
    use_kdtree = params[name_dataset]['use_kdtree']
    theta = params[name_dataset]['theta']
    max_iterations = params[name_dataset]['max_iterations']
    c_init = params[name_dataset]['c_init']
    c_steps = params[name_dataset]['c_steps']
    feature_range = (Xtr.min(axis=0), Xtr.max(axis=0))
    shape = (Xtr[0].reshape((1,) + Xtr[0].shape)).shape
    feature_range = (Xtr.min(axis=0), Xtr.max(axis=0))
    # ######################## #

    if verbose > 1:
        print('[Params Info] use_kdtree            = ', use_kdtree)
        print('[Params Info] theta                 = ', theta)
        print('[Params Info] max_iterations        = ', max_iterations)
        print('[Params Info] c_init                = ', c_init)
        print('[Params Info] c_steps               = ', c_steps)
        print('[Params Info] NN                    = ', params[name_dataset]['NN'])

    # ########### Initialize and fit CEM explainer ########### #
    cf = CounterfactualProto(predict=NN_model.forward_2_outputs_numpy,
                             shape=shape,
                             use_kdtree=use_kdtree,
                             theta=theta,
                             max_iterations=max_iterations,
                             feature_range=feature_range,
                             c_init=c_init,
                             c_steps=c_steps)
    cf.fit(Xtr)
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
        explanation = cf.explain(sample_to_explain)

        # #### Record results #### #
        if explanation.cf is not None:
            counterfactuals[index_sample, :] = explanation.cf['X'][0]
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
