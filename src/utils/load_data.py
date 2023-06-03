import numpy
import torch
import typing


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
