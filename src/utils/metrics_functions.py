import pandas
import numpy
import sklearn.neighbors
import typing
from src.utils.load_data import load_data


def compute_probabilistic_cost_for_class(cost_mat: numpy.ndarray,
                                         prob_objetive_counter: numpy.ndarray,
                                         prob_objetive_orig: numpy.ndarray,
                                         y_pred_orig: numpy.ndarray,
                                         y_true_orig: numpy.ndarray,
                                         objetive_class: int = -1,
                                         oposite_class: int = 1,
                                         ) -> typing.Tuple[pandas.DataFrame, typing.Dict[str, object]]:
    """
    Compute the probabilistic costs for the samples that match a given objetve class and optisite class.

    This compute the probabilistic costs for original and counterfactual samples. When searching from counterfactuals
    the objetive is to find a sample that classify as the oposite class from the originally predicted class
    (objetive_class). Therefore, in this function we call "oposite class" to the counterfactual class, i.e., if the
    original class (objetive_class) is 0, the objetive class will be 1.

    This functions also computes the costs for the dummy coutnerfactuals, with are the result of directly switch the
    predicted class with no perturbation fo the original sample. This will produce a high cost counterfactulas that are
    used as benchmarking. As the position of this counterfactuals are the same as the original samples we know the
    resulting true labes, so we could compute the true cost of the decission for that samples, not just the
    probabilistic ones.

    The process have the following steps:
        1. Filter the samples that are failures and success of finding the specified counterfacual flip (indicated with
           objetive_class, and oposite_class values).
        2. Computes the true costs for the original and dummy counterfactual samples.
        3. Computes the probabilistic costs for the method counterfactual samples. As the position of the counterfactual
           sample is not the same as the original sample, we dont know the we dont know the true labels, so the costs
           of the decisions must be computed based on the probabilies.

    [Warning]: The code is only adapted to binary classification when the labels are [-1, +1], specifically the input
               data, MUST have label '-1' for the mayority class, and '+1' for the minority class. Also, this fucntion
               expect probabilities as input, i.e., probabilities as float in the range [0, 1].

    Args:
        cost_mat (numpy.ndarray): The example-dependent cost matrix for each sample.
        prob_objetive_counter (numpy.ndarray): Probability of the objetive class for the original sample.
        prob_objetive_orig (numpy.ndarray): Probability of the objetive class for the counterfactual sample.
        y_pred_orig (numpy.ndarray): The class prediction for the original sample.
        y_true (numpy.ndarray): The true class for the original sample.
        objetive_class (int): The objetive class. Default to -1.
        oposite_class (int): The oposite class. Default to +1.

    Returns:
        df_result (pandas.DataFrame): A pandas.DataFrame containing the following variables:
            prob_cost_orig (numpy.ndarray): The probabilistic cost for each original sample.
            prob_cost_counter_dummy (numpy.ndarray): The probabilistic cost for each counterfactual (computed with the
                                                     dummy method) sample.
            prob_cost_counter (numpy.ndarray): The probabilistic cost for each counterfactual (computed with the
                                               corresponding method to measure its performance) sample.

        info (typing.Dict[str, object]): A Dictionary containing the following variables:
            true_cost_orig (float): The sum of the TRUE costs of the original samples.
            total_prob_cost_orig (float): The sum of the PROBA costs of the original samples.
            true_cost_counter_dummy (float): The sum of the TRUE costs of the counter samples.
            total_prob_cost_counter_dummy (float): The sum of the PORBA costs of the dummy counter samples.
            total_prob_cost_counter (float): The sum of the PROBA costs of the counter samples.
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
    key3 = f"{key_columns}_c{str_oposite_class}{str_oposite_class}"
    key4 = f"{key_columns}_c{str_oposite_class}{str_objetive_class}"

    prob_cost_counter = prob_objetive_counter * cost_mat[key4] + (1 - prob_objetive_counter) * cost_mat[key3]
    total_prob_cost_counter = prob_cost_counter[filter_objetive_class_samples].sum()

    df_result = {'prob_cost_orig': prob_cost_orig,
                 'prob_cost_counter_dummy': prob_cost_counter_dummy,
                 'prob_cost_counter': prob_cost_counter,
                 }
    df_result = pandas.DataFrame(df_result)

    info = {'true_cost_orig': true_cost_orig,
            'total_prob_cost_orig': total_prob_cost_orig,
            'true_cost_counter_dummy': true_cost_counter_dummy,
            'total_prob_cost_counter_dummy': total_prob_cost_counter_dummy,
            'total_prob_cost_counter': total_prob_cost_counter
            }

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
    Computes the probabilistic cost for a set of orginal, and counterfactual samples. Also computes the metrics for
    the dummy counterfactuals.

    The process have the following steps:
        1. Comput the probabilistic costs for all the samples depending on the original predicted label. Therefore,
           the function 'compute_probabilistic_cost_for_class' is called 2 times, one per "oposite class" option (binary
           classification problem).
        2. Merge and sum the costs results of both "oposite class" options.
        3. If Verbose is True prints some usefull information of the resulting metrics.

    [Warning]: The code is only adapted to binary classification when the labels are [-1, +1], specifically the input
               data, MUST have label '-1' for the mayority class, and '+1' for the minority class. Also, this fucntion
               expect probabilities as input, i.e., probabilities as float in the range [0, 1].
               The probability is understood as the prob of a sample belonging to class 1, that is, the minority class.
    Args:
        cost_mat (numpy.ndarray): The example-dependent cost matrix for each sample.
        pred_prob_counter (numpy.ndarray): The probability of the minoritary class for each counterfactual sample.
        pred_prob_orig (numpy.ndarray): The probability of the minoritary class for each original sample.
        y_pred_orig (numpy.ndarray): The predicted class of the original samples.
        y_true_orig (numpy.ndarray): The true label for each orignal sample.
        one_class (int): The label corresponding to the minoritary class. Default to 1.
        zero_class (int): The label corresponding to the mayoritary class.  Defatul to 0.
        verbose (bool): If True prints the resulting metrics for the original and conterfactual an ddummy counterfactual
                        samples. Defatult to True.

    Returns:
        result (pandas.DataFrame):
            pred_prob_orig (numpy.ndarray): The input probability of the minoritary class for each original sample.
            pred_prob_counter (numpy.ndarray): The input probability of the minoritary class for each counterfactual
                                               sample.
            y_pred_orig (numpy.ndarray): The input predicted class of the original samples.
            y_true_orig (numpy.ndarray): The input true label for each orignal sample.
            c11_orig (numpy.ndarray): The cost c11 for the original samples.
            c00_orig (numpy.ndarray): The cost c00 for the original samples.
            c10_orig (numpy.ndarray): The cost c10 for the original samples.
            c01_orig (numpy.ndarray): The cost c01 for the original samples.
            prob_cost_orig (numpy.ndarray): The merged probabilistic cost for each original sample.
            prob_cost_counter_dummy (numpy.ndarray): The merged probabilistic cost for each counter sample.
            prob_cost_counter_proposed (numpy.ndarray): The merged probabilistic cost for each dummy counter sample.
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
              'y_pred_orig': y_pred_orig,
              'y_true_orig': y_true_orig,
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
    Compute all the metrics for all the specified datasets and counterfactual methods in the input lists.

    The process have the following steps for each dataset and counterfactual method specified in the input lists :
        1. Load the precomputed counterfactual results.
        2. Filter the samples that are originally predicted as minoritary class.
        3. Call the function 'compute_probabilistic_cost' to compute the probabilistic costs.
        4. Compute the rest of the metrics:
                - The Probabilistic savings (based on the probabilistic costs)
                - The success ratio (proporton of samples that meet the objetive of the counterfactual search): For
                  traditional counterfactual methods is to find a sample thar flips the decission and for the
                  cost counterfactual methods, as the proposed, is to find a counterfactual sample that flips the
                  bayes cost-based decision.
                - The distance between the original and counterfactual sample
                - The distance between the counterfactual and the nearest plausible sample. The set of plausible
                  samples are the original samples belonging to the 0 class. That means plausible "states" for a
                  real world sample.

    [Warning]: The code is only adapted to binary classification when the labels are [-1, +1], specifically the input
               data, MUST have label '-1' for the mayority class, and '+1' for the minority class. Also, this fucntion
               expect to load probabilities results, i.e., probabilities as float in the range [0, 1].
               The probability is understood as the prob of a sample belonging to class 1, that is, the minority class.

    Args:
        method_name_list (typing.List[str]): The list of the names of the counterfactual methods to compute its
                                             performance.
        dataset_name_list (typing.List[str]): The list of the names of the datasets to compute its performance.
        load_experiments_path (str): The path where to find the precomputed counterfactual results for each method and
                                     dataset.
        verbose (bool): If True prints the results of the computed metrics. Default to False.

    Returns:
        global_mean_plsusible_distance (typing.Dict[str, float]): The global mean distance metric between the
                                                                  counterfactual to the nearest plausible sample.
        global_mean_orgi_distance (typing.Dict[str, float]): The global mean distance metric from counterfactual to the
                                                             orginal sample.
        global_succeed_ratio (typing.Dict[str, float]): The global succeed ratio metric.
        global_counter_savings (typing.Dict[str, float]): The global counter savings metric.
        global_results_dict (typing.Dict[str, float]): All the results from the 'compute_probabilistic_cost' function.
    """
    global_results_dict = {}
    global_counter_savings = {}
    global_succeed_ratio = {}
    global_mean_orig_distance = {}
    global_mean_plausible_distance = {}
    for index_model in range(len(method_name_list)):
        model_name = method_name_list[index_model]
        path_save_counter = f"{load_experiments_path}/{model_name}/"

        model_results_dict = {}
        counter_savings = {}
        succeed_ratio = {}
        mean_orig_distance = {}
        mean_plausible_distance = {}
        for name_dataset in dataset_name_list:

            # ######## Load orginal dataset ######## #
            path_data = "../data/datasets/"
            loaded_dataset = load_data(name_dataset=name_dataset,
                                       path_data=path_data,
                                       verbose=verbose)

            x_ts = loaded_dataset['tensor_x_ts']
            y_ts = loaded_dataset['tensor_y_ts']
            # ###################################### #

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
            # [WARNING]: The values of o_counter_samples must be in the range [-1, 1]
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
            pred_class_0_1 = (results['pred_prob_counter'] >= 0.5) * 1

            # To convert the output model [-1, 1] into probability rante [0, 1]
            results['y_pred_counter'] = (pred_class_0_1 * 2) - 1
            # ######################################### #

            # ################ Final metrics ################ #
            # #### Counter Savings #### #
            cost_dummy = results['prob_cost_counter_dummy'].sum()  # The same dummy for all the benchmarks and proposed.
            cost_model = results['prob_cost_counter_proposed'].sum()  # The cost of the decisions of each tested model.
            dataset_counter_savings = (cost_dummy - cost_model) / cost_dummy  # cost_dummy / (cost_model + cost_dummy)
            # ######################### #

            # #### Succeed ratio #### #
            if model_name == 'Proposed':
                eps_cost_objetive_function = 0.3

                ### cost boundary Succed Ratio ###
                if name_dataset == 'CS2':
                    margen_nu = -0.5
                else:
                    margen_nu = 0.0

                list_d_final = success_vars['list_d_final']
                list_g_final = success_vars['list_g_final']
                list_f_final = success_vars['list_f_final']
                list_g_inicial = success_vars['list_g_inicial']
                list_f_inicial = success_vars['list_f_inicial']

                # Compute if the counterfactual cross the cost boundary, or equivalently,
                # minimize the objetive function with a tolerance.
                cond1 = (numpy.abs(numpy.array(list_d_final) + margen_nu) <= eps_cost_objetive_function)
                cond2 = (numpy.abs(list_g_final) > eps_cost_objetive_function)
                cond3 = (numpy.abs(list_f_final) > eps_cost_objetive_function)
                cond4 = ((numpy.abs(list_f_inicial) < eps_cost_objetive_function) * (numpy.abs(list_g_inicial) < eps_cost_objetive_function))

                success = cond1 * (cond2 + cond3 + cond4)
                dataset_succeed_ratio = success.mean()
                ##################################

            else:
                ### Prob boundary Succed Ratio ###
                eps_prob_boundary = 0.05
                pred_class_0_1 = (results['pred_prob_counter'] >= 0.5) * 1
                num_fails = (pred_class_0_1 == 1).sum()
                dataset_succeed_ratio = (len(results) - num_fails) / len(results)
                ##################################
            # ####################### #

            # # Distance counter-orig # #
            dataset_mean_orig_distance = numpy.sqrt(((counter_samples - orig_samples)**2).sum(axis=1)).mean()
            dataset_std_orig_distance = numpy.sqrt(((counter_samples - orig_samples)**2).sum(axis=1)).std()
            # ######################### #

            # # Distance counter-nearest_plausible # #
            orgi_samples_class_0 = x_ts[y_ts == -1]
            neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=1)
            neigh.fit(X=orgi_samples_class_0)
            neigh_dist, neigh_ind = neigh.kneighbors(X=counter_samples)
            dataset_mean_plausible_distance = neigh_dist.mean()
            dataset_std_plausible_distance = neigh_dist.std()
            # ###################################### #

            if verbose:
                print(f"The counterfactual savings: {counter_savings} for {name_dataset} dataset ")
            # ############################################## #

            mean_plausible_distance[name_dataset] = (str(numpy.round(dataset_mean_plausible_distance, 2)) + u" \u00B1 "
                                                     + str(numpy.round(dataset_std_plausible_distance, 2)))
            mean_orig_distance[name_dataset] = (str(numpy.round(dataset_mean_orig_distance, 2)) + u" \u00B1 "
                                                + str(numpy.round(dataset_std_orig_distance, 2)))
            succeed_ratio[name_dataset] = numpy.round(dataset_succeed_ratio * 100, 2)
            counter_savings[name_dataset] = numpy.round(dataset_counter_savings * 100, 2)
            model_results_dict[name_dataset] = results

        global_mean_plausible_distance[model_name] = mean_plausible_distance
        global_mean_orig_distance[model_name] = mean_orig_distance
        global_succeed_ratio[model_name] = succeed_ratio
        global_counter_savings[model_name] = counter_savings
        global_results_dict[model_name] = model_results_dict

        final_results = {'global_mean_plausible_distance': global_mean_plausible_distance,
                         'global_mean_orig_distance': global_mean_orig_distance,
                         'global_succeed_ratio': global_succeed_ratio,
                         'global_counter_savings': global_counter_savings,
                         'global_results_dict': global_results_dict}

    return final_results
