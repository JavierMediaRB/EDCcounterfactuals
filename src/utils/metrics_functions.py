import pandas
import numpy
import typing


def compute_probabilistic_cost_for_class(cost_mat: numpy.ndarray,
                                         prob_objetive_counter: numpy.ndarray,
                                         prob_objetive_orig: numpy.ndarray,
                                         y_pred_orig: numpy.ndarray,
                                         y_true_orig: numpy.ndarray,
                                         objetive_class: int = -1,
                                         oposite_class: int = 1,
                                         ) -> typing.Tuple[pandas.DataFrame, typing.Dict[str, object]]:
    """
    TODO: Description...

    [Warning]: The code is only adapted to binary classification when the labels are [-1, +1], specifically the input
               data, MUST have label '-1' for the mayority class, and '+1' for the minority class. Also, this fucntion
               expect probabilities as input, i.e., probabilities as float in the range [0, 1].

    Args:
        cost_mat (numpy.ndarray):
        prob_objetive_counter (numpy.ndarray):
        prob_objetive_orig (numpy.ndarray):
        y_pred_orig (numpy.ndarray):
        y_true (numpy.ndarray):
        objetive_class (int): Default to -1.
        oposite_class (int): Default to +1.

    Returns:
        df_result ():
            prob_cost_orig ():
            prob_cost_counter_dummy ():
            prob_cost_counter ():

        info ():
            true_cost_orig ():
            total_prob_cost_orig ():
            true_cost_counter_dummy ():
            total_prob_cost_counter_dummy ():
            total_prob_cost_counter ():
    """
    str_objetive_class = str(int((objetive_class + 1) / 2))
    str_oposite_class = str(int((oposite_class + 1) / 2))

    filter_objetive_class_samples = (y_pred_orig == objetive_class)
    filter_success = ((y_pred_orig == objetive_class) & (y_true_orig == objetive_class))
    filter_failure = ((y_pred_orig == objetive_class) & (y_true_orig == oposite_class))

    key_columns = 'orig'
    key1 = f"{key_columns}_c{str_objetive_class}{str_objetive_class}"
    key2 = f"{key_columns}_c{str_objetive_class}{str_oposite_class}"
    key3 = f"{key_columns}_c{str_oposite_class}{str_oposite_class}"
    key4 = f"{key_columns}_c{str_oposite_class}{str_objetive_class}"

    true_cost_orig = cost_mat[key1].iloc[filter_success].sum() + cost_mat[key2].iloc[filter_failure].sum()
    prob_cost_orig = prob_objetive_orig * cost_mat[key1] + (1 - prob_objetive_orig) * cost_mat[key2]
    total_prob_cost_orig = prob_cost_orig[filter_objetive_class_samples].sum()
    true_cost_counter_dummy = cost_mat[key3].iloc[filter_success].sum() + cost_mat[key4].iloc[filter_failure].sum()
    prob_cost_counter_dummy = prob_objetive_orig * cost_mat[key4] + (1 - prob_objetive_orig) * cost_mat[key3]
    total_prob_cost_counter_dummy = prob_cost_counter_dummy[filter_objetive_class_samples].sum()

    key_columns = 'counter'
    key1 = f"{key_columns}_c{str_oposite_class}{str_objetive_class}"
    key2 = f"{key_columns}_c{str_oposite_class}{str_oposite_class}"

    prob_cost_counter = prob_objetive_counter * cost_mat[key1] + (1 - prob_objetive_counter) * cost_mat[key2]
    total_prob_cost_counter = prob_cost_counter[filter_objetive_class_samples].sum()

    df_result = {'prob_cost_orig': prob_cost_orig,
                 'prob_cost_counter_dummy': prob_cost_counter_dummy,
                 'prob_cost_counter': prob_cost_counter,
                 }

    info = {'true_cost_orig': true_cost_orig,
            'total_prob_cost_orig': total_prob_cost_orig,
            'true_cost_counter_dummy': true_cost_counter_dummy,
            'total_prob_cost_counter_dummy': total_prob_cost_counter_dummy,
            'total_prob_cost_counter': total_prob_cost_counter
            }

    df_result = pandas.DataFrame(df_result)
    return df_result, info


def compute_probabilistic_cost(cost_mat: numpy.ndarray,
                               pred_prob_counter: numpy.ndarray,
                               pred_prob_orig: numpy.ndarray,
                               y_pred_orig: numpy.ndarray,
                               y_true_orig: numpy.ndarray,
                               one_class: int = 1,
                               zero_class: int = -1,
                               verbose: bool = True) -> pandas.DataFrame:
    """
    TODO: Description...

    [Warning]: The code is only adapted to binary classification when the labels are [-1, +1], specifically the input
               data, MUST have label '-1' for the mayority class, and '+1' for the minority class. Also, this fucntion
               expect probabilities as input, i.e., probabilities as float in the range [0, 1].
               The probability is understood as the prob of a sample belonging to class 1, that is, the minority class.
    Args:
        cost_mat (numpy.ndarray):
        pred_prob_counter (numpy.ndarray):
        pred_prob_orig (numpy.ndarray):
        y_pred_orig (numpy.ndarray):
        y_true (numpy.ndarray):
        one_class (int): Default to 1.
        zero_class (int): Defatul to 0.
        verbose (bool): Defatult to True.

    Returns:
        result (pandas.DataFrame):
            pred_prob_orig ():
            pred_prob_counter ():
            y_true ():
            y_pred_orig ():
            c11_orig ():
            c00_orig ():
            c10_orig ():
            c01_orig ():
            prob_cost_orig ():
            prob_cost_counter_dummy ():
            prob_cost_counter_proposed ():
    """
    df_result_class0, info_class0 = compute_probabilistic_cost_for_class(cost_mat=cost_mat,
                                                                         prob_objetive_counter=(1 - pred_prob_counter),
                                                                         prob_objetive_orig=(1 - pred_prob_orig),
                                                                         y_pred_orig=y_pred_orig,
                                                                         y_true_orig=y_true_orig,
                                                                         objetive_class=-1,
                                                                         oposite_class=1)

    df_result_class1, info_class1 = compute_probabilistic_cost_for_class(cost_mat=cost_mat,
                                                                         prob_objetive_counter=pred_prob_counter,
                                                                         prob_objetive_orig=pred_prob_orig,
                                                                         y_pred_orig=y_pred_orig,
                                                                         y_true_orig=y_true_orig,
                                                                         objetive_class=1,
                                                                         oposite_class=-1)

    filter_samples_pred_as_one_class = (y_pred_orig == one_class)
    # filter_samples_pred_as_zero_class = (y_pred_orig == zero_class)

    merge_result = df_result_class0
    merge_result.loc[filter_samples_pred_as_one_class, :] = df_result_class1.loc[filter_samples_pred_as_one_class, :]

    var1 = 'total_prob_cost_orig'
    total_prob_cost_orig = info_class1[var1] + info_class0[var1]
    true_cost_orig = info_class1['true_cost_orig'] + info_class0['true_cost_orig']
    var2 = 'total_prob_cost_counter_dummy'
    total_prob_cost_counter_dummy = info_class1[var2] + info_class0[var2]
    true_cost_counter_dummy = info_class1['true_cost_counter_dummy'] + info_class0['true_cost_counter_dummy']
    total_prob_cost_counter = info_class1['total_prob_cost_counter'] + info_class0['total_prob_cost_counter']

    if verbose:
        aux1 = f"Total probabilistic_cost = {total_prob_cost_orig} | true_cost = {true_cost_orig}"
        aux2 = f"Total probabilistic_cost = {total_prob_cost_counter_dummy} | true_cost = {true_cost_counter_dummy}"
        aux3 = f"Total probabilistic_cost = {total_prob_cost_counter} | true_cost = ¡Can't compute it !"

        print(f"Original Costs:            \n \t {aux1}")
        print(f"Cunterfactual Dummy Costs: \n \t {aux2}")
        print(f"Counterfactual Costs:      \n \t {aux3}")

    result = {'pred_prob_orig': pred_prob_orig,
              'pred_prob_counter': pred_prob_counter,
              'y_true_orig': y_true_orig,
              'y_pred_orig': y_pred_orig,
              'c11_orig': cost_mat["orig_c11"],
              'c00_orig': cost_mat["orig_c00"],
              'c10_orig': cost_mat["orig_c10"],
              'c01_orig': cost_mat["orig_c01"],
              'prob_cost_orig': merge_result['prob_cost_orig'],
              'prob_cost_counter_dummy': merge_result['prob_cost_counter_dummy'],
              'prob_cost_counter_proposed': merge_result['prob_cost_counter']
              }

    result = pandas.DataFrame(result)

    return result


def compute_metrics(method_name_list: typing.List[str],
                    dataset_name_list: typing.List[str],
                    load_experiments_path: str,
                    verbose: bool = False,
                    ) -> typing.Dict[str, typing.Dict[str, float]]:
    """
    TODO: Description...

    [Warning]: The code is only adapted to binary classification when the labels are [-1, +1], specifically the input
               data, MUST have label '-1' for the mayority class, and '+1' for the minority class. Also, this fucntion
               expect to load probabilities results, i.e., probabilities as float in the range [0, 1].
               The probability is understood as the prob of a sample belonging to class 1, that is, the minority class.


    Args:
        method_name_list (typing.List[str]):
        dataset_name_list (typing.List[str]):
        load_experiments_path (str):
        verbose (bool):  Default to False.
    Returns:
        global_mean_distance ():
        global_succeed_ratio ():
        global_counter_savings ():
        global_results_dict ():
    """
    global_results_dict = {}
    global_counter_savings = {}
    global_succeed_ratio = {}
    global_mean_distance = {}
    for index_model in range(len(method_name_list)):
        model_name = method_name_list[index_model]
        path_save_counter = f"{load_experiments_path}/{model_name}/"

        model_results_dict = {}
        counter_savings = {}
        succeed_ratio = {}
        mean_distance = {}
        for name_dataset in dataset_name_list:

            # ### Load counterfactual search results ### #
            if verbose:
                print(f"\n\nDataset: {name_dataset}")

            file_save_counter = f"counterfactual_results_{name_dataset}.npy"
            direccion_resultado = path_save_counter + file_save_counter
            load_file = numpy.load(direccion_resultado, allow_pickle=True)

            orig_samples = load_file.item()['original_samples']
            o_orig_samples = load_file.item()['o_original_samples']
            y_pred_orig_samples = load_file.item()['y_pred_original_samples']
            counter_samples = load_file.item()['counterfactual_samples']
            o_counter_samples = load_file.item()['o_counterfactual_samples']
            # y_pred_counter_samples = load_file.item()['y_pred_counterfactual_samples']
            cost_mat = load_file.item()['cost_matrix']
            y_true_orig_samples = load_file.item()['y_true_orignal_samples']
            if model_name == 'Proposed':
                success_vars = load_file.item()['success_vars']
            # ########################################## #

            # ##### filter minoritary orig preds ##### #
            filter = y_pred_orig_samples == 1

            cost_mat = cost_mat[filter]
            o_counter_samples = o_counter_samples[filter]
            o_orig_samples = o_orig_samples[filter]
            y_pred_orig_samples = y_pred_orig_samples[filter]
            y_true_orig_samples = y_true_orig_samples[filter]
            # ######################################## #

            # ########## Measure cost impact ########## #
            pred_prob_counter = (o_counter_samples.detach().flatten().numpy() + 1) / 2
            pred_prob_orig = (o_orig_samples.detach().flatten().numpy() + 1) / 2
            results = compute_probabilistic_cost(cost_mat=cost_mat,
                                                 pred_prob_counter=pred_prob_counter,
                                                 pred_prob_orig=pred_prob_orig,
                                                 y_pred_orig=(y_pred_orig_samples).astype(int),
                                                 y_true_orig=(y_true_orig_samples).astype(int),
                                                 one_class=1,
                                                 zero_class=-1,
                                                 verbose=verbose)

            # Add y_pred_counter variable
            pred_class_0_1 = (results['pred_prob_counter'] >= 0.5) * 1  # TODO: Esto solo está bien si TODOS los metodos dan o in[-1, 1] y por tanto al combertir tengo prob in[0, 1]

            # To convert the output model [-1, 1] into probability rante [0, 1]
            results['y_pred_counter'] = (pred_class_0_1 * 2) - 1
            # ######################################### #

            # ################ Final metrics ################ #
            # #### Counter Savings #### #
            cost_dummy = results['prob_cost_counter_dummy'].sum()  # El mismo dummy para todos los benchmarks
            cost_model = results['prob_cost_counter_proposed'].sum()  # El coste de las deciones de cada modelo testado
            dataset_counter_savings = cost_dummy / (cost_model + cost_dummy)
            # ######################### #

            # #### Succeed ratio #### #
            if model_name == 'Proposed':
                eps_success = 0.3

                if name_dataset == 'CS2':
                    margen_nu = -0.5
                else:
                    margen_nu = 0.0

                list_d_final = success_vars['list_d_final']
                list_g_final = success_vars['list_g_final']
                list_f_final = success_vars['list_f_final']
                list_g_inicial = success_vars['list_g_inicial']
                list_f_inicial = success_vars['list_f_inicial']
                # list_success = success_vars['list_success']

                cond1 = (numpy.abs(numpy.array(list_d_final) + margen_nu) <= eps_success)
                cond2 = (numpy.abs(list_g_final) > eps_success)
                cond3 = (numpy.abs(list_f_final) > eps_success)
                cond4 = ((numpy.abs(list_f_inicial) < eps_success) * (numpy.abs(list_g_inicial) < eps_success))

                success = cond1 * (cond2 + cond3 + cond4)
                dataset_succeed_ratio = success.mean()
            else:
                num_fails = (results['y_pred_counter'] == 1).sum()
                dataset_succeed_ratio = (len(results) - num_fails) / len(results)
            # ####################### #

            # Distance counter-orig
            dataset_mean_distance = numpy.sqrt(((counter_samples - orig_samples)**2).sum(axis=1)).mean()

            if verbose:
                print(f"The counterfactual savings: {counter_savings} for {name_dataset} dataset ")
            # ############################################## #

            mean_distance[name_dataset] = numpy.round(dataset_mean_distance, 2)
            succeed_ratio[name_dataset] = numpy.round(dataset_succeed_ratio * 100, 2)
            counter_savings[name_dataset] = numpy.round(dataset_counter_savings * 100, 2)
            model_results_dict[name_dataset] = results

        global_mean_distance[model_name] = mean_distance
        global_succeed_ratio[model_name] = succeed_ratio
        global_counter_savings[model_name] = counter_savings
        global_results_dict[model_name] = model_results_dict

        final_results = {'global_mean_distance': global_mean_distance,
                         'global_succeed_ratio': global_succeed_ratio,
                         'global_counter_savings': global_counter_savings,
                         'global_results_dict': global_results_dict}

    return final_results
