import numpy
import torch
import typing
import pandas
import matplotlib.pyplot
from src.cost_policy.cost_policy_for_datasets import cost_function_numpy
from src.benchmarks.Basic_MLP import Basic_MLP
from src.benchmarks.GRACE_method.main_args_parser import arguments
from src.benchmarks.GRACE_method.utils import get_constraints
from src.benchmarks.GRACE_method.selector import FeatureSelector
from src.benchmarks.GRACE_method.grace import test_grace


class trainer_custom:
    """
    Auxuliar class to connect the Basic_MLP class used in the experiments to the functions in the GRACE method.

    Params:
        model (Basic_MLP): The model to connect to the GRACE class.
    """

    def __init__(self, model: Basic_MLP):
        self.model = model

    def predict(self, sample: torch.Tensor) -> numpy.ndarray:
        """
        A function that call the 'predict_class' function from the Basic_MLP class. It just evaluates the model
        and predicts the class for a input sample.

        Args:
            sample (torch.Tensor): The input sample data.

        Returns:
            y_pred (numpy.ndarray): The predicted class.
        """

        y_pred = self.model.predict_class(sample)
        return y_pred


def test_aux_custom(train_data: numpy.ndarray,
                    test_data: numpy.ndarray,
                    model: Basic_MLP,
                    scaler: object,
                    args: arguments
                    ) -> typing.Dict[str, object]:
    """
    This function connect the Basic_MLP class method to GRACE methods by means of the auxiliar 'trainer_custom' class.

    Args:
        train_data (numpy.ndarray): The input train data.
        test_data (numpy.ndarray): The input test data.
        model (Basic_MLP): The model to connect to the GRACE class.
        scaler (object): Defatul to None.
        args (arguments): The object class 'arguments' from GRACE containing the experiment params.

    Returns:
        original_samples (numpy.ndarray): The original input test data.
        o_original_samples (torch.Tensor): The model outputs for the test data.
        y_pred_original_samples (numpy.ndarray): The model class predictions for the test data.
        counterfactual_samples (numpy.ndarray): The computed counterfactual samples.
        o_counterfactual_samples (torch.Tensor): The output of the model for the counterfactual samples.
        y_pred_counterfactual_samples (numpy.ndarray): The model class predictions for the counterfactual samples.
    """

    # Instanciate the trainer_custom class with the Basic_MLP class model
    trainer = trainer_custom(model)

    # configurations for generating explanation
    num_feat = train_data.shape[1]
    bound_min, bound_max, bound_type = get_constraints(train_data)
    alphas = args.alpha * numpy.ones(num_feat) if args.alpha > 0 else numpy.std(train_data, axis=0)
    feature_selector = FeatureSelector(train_data, args.gen_gamma) if args.gen_gamma > 0.0 else None

    print(feature_selector)

    results = test_grace(model,
                         trainer,
                         test_data,
                         args,
                         method="Naive",
                         scaler=scaler,
                         bound_min=bound_min,
                         bound_max=bound_max,
                         bound_type=bound_type,
                         alphas=alphas,
                         feature_selector=feature_selector)

    orig_test_samples = test_data
    original_test_prediction = results['original_label']
    contrafactual_test_samples = results['counterfactual_x']
    counterfactual_test_prediction = results['counterfactual_label']
    counterfactual_output_model = results['counter_probs']
    o_orig_ts = model.forward_2_outputs(torch.autograd.Variable(torch.from_numpy(test_data)))
    o_orig_ts = o_orig_ts.data.cpu().detach().numpy().flatten()

    results = {'original_samples': orig_test_samples,
               'o_original_samples': o_orig_ts,
               'y_pred_original_samples': original_test_prediction,
               'counterfactual_samples': contrafactual_test_samples,
               'o_counterfactual_samples': counterfactual_output_model,
               'y_pred_counterfactual_samples': counterfactual_test_prediction,
               }

    return results


def search_counterfactuals_GRACE(name_dataset: str,
                                 NN_model: Basic_MLP,
                                 loaded_dataset: typing.Dict[str, object],
                                 params: typing.Dict[str, object],
                                 figure: bool = False,
                                 verbose: int = 1,
                                 small_test: bool = False
                                 ) -> typing.Dict[str, object]:
    """
    This function performs the pipeline process to generate a set of counterfactual samples for the GRACE method.
    The process have the following consecutive steps:

        1. Load the needed params for the experiment
        2. Load the data from the variable 'loaded_dataset'
        3. Set the optimization restrictions (if aplicable)
        4. Compute an initial prediction on the test data with the 'NN_model' model to select only the
           samples that are originally predicted in the minoritary class.
        5. If the param 'small_test' is True, select just the first 2 samples to reduce the computational cost. This is
           usefull to debug the code if necessary.
        6. Call the GRACE method to perform the searching of counterfactuals.
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

    # ######## Params ######## #
    scaler = None
    lr = params[name_dataset]['lr']
    gen_gamma = params[name_dataset]['gen_gamma']
    gen_max_features = params[name_dataset]['gen_max_features']
    gen_max_iter = params[name_dataset]['gen_max_iter']
    gen_overshoot = params[name_dataset]['gen_overshoot']
    explain_units = params[name_dataset]['explain_units']

    args = arguments(lr=lr,
                     gen_gamma=gen_gamma,
                     # El numero de features con las que busca la opti a la vez.
                     gen_max_features=gen_max_features,
                     # Si pongo mas ejecuta N veces la opti hasta que encuentra un candidato, pero si no puede fallar.
                     gen_max_iter=gen_max_iter,
                     # Estas dos anteriores las detiene cuando encuentra un contrafactual.
                     gen_overshoot=gen_overshoot,
                     explain_units=explain_units)
    # ######################## #

    # ######## loaded dataset ######## #
    tensor_x_ts = loaded_dataset['tensor_x_ts']
    tensor_y_ts = loaded_dataset['tensor_y_ts']
    tensor_x_tr = loaded_dataset['tensor_x_tr']
    # tensor_y_tr = loaded_dataset['tensor_y_tr']
    # IR_tr = loaded_dataset['IR_tr']
    mean_tr = loaded_dataset['mean_tr']
    std_tr = loaded_dataset['std_tr']
    # ############################### #

    if verbose > 1:

        print('[Params Info] lr                    = ', lr)
        print('[Params Info] gen_gamma             = ', gen_gamma)
        print('[Params Info] gen_max_features      = ', gen_max_features)
        print('[Params Info] gen_max_iter          = ', gen_max_iter)
        print('[Params Info] gen_overshoot         = ', gen_overshoot)
        print('[Params Info] explain_units         = ', explain_units)
        print('[Params Info] NN                    = ', params[name_dataset]['NN'])

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

    counter_search_result = test_aux_custom(train_data=tensor_x_tr.detach().numpy().copy(),
                                            test_data=Xts_filtered,
                                            model=NN_model,
                                            scaler=scaler,
                                            args=args)  # test the trained model with a generation method

    # #### Compute the model outputs and predictions for original anc counterfactual samples #### #
    # list_counter_samples = numpy.array(list_counter_samples)
    list_counter_samples = counter_search_result['counterfactual_samples']

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
