"""
This file contains a copy of some of the functions needed for use the GRACE method. This
functions are an exact copy of the ones in the GRACE_KDD20 repository (https://github.com/lethaiq/GRACE_KDD20/tree/master). The only aim to
copy these functions is to preserve the reproducibility of the experiments.

References
    ----------
    @article{le2019grace,
             title={GRACE: Generating Concise and Informative Contrastive Sample to Explain Neural Network Model's Prediction},
             author={Thai Le and Suhang Wang and Dongwon Lee},
             year={2019},
             journal={Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '20)},
             doi={10.1145/3394486.3403066}
             isbn={978-1-4503-7998-4/20/08}
             }
"""

import torch
import numpy as np
from tqdm import tqdm as tqdm
import time
from torch.autograd import Variable

def generate(x, model, gen_model, args, scaler=None, trainer=None, **kargs):

    ini_probs = model.forward_2_outputs(Variable(x)).data.cpu().numpy().flatten()
    #print(f"Initial probs: {np.round(ini_probs, 2)}")

    for j in range(args.gen_max_features):
        #print(f"New number of features search: N features = {j+1}")
        lb_org, lb_new, r, x_adv, feats_idx, nb_iter = gen_model(x=x,
                                                                 num_feat=j+1,
                                                                 net=model,
                                                                 overshoot=args.gen_overshoot,
                                                                 max_iter=args.gen_max_iter,
                                                                 bound_min=kargs["bound_min"],
                                                                 bound_max=kargs["bound_max"],
                                                                 bound_type=kargs["bound_type"],
                                                                 alphas=kargs["alphas"],
                                                                 feature_selector=kargs['feature_selector'])
        if scaler:
            x_adv = scaler.inverse_transform(x_adv.reshape(1, -1))[0]
            lb_new = trainer.predict(x_adv)
        if lb_org != lb_new:
            break
    
    fin_probs = model.forward_2_outputs(Variable(torch.from_numpy(x_adv))).data.cpu().numpy().flatten()
    #print(f"Final probs  : {np.round(fin_probs, 2)}\n")
    time.sleep(0.1)
    return lb_org, lb_new, x_adv, feats_idx, fin_probs


def test_grace(model, trainer, test_x, args, method="Naive", scaler=None, **kargs):
    if method == "Naive":
        from src.benchmarks.GRACE_method.methods_refactored import NaiveGradient

        gen_model = NaiveGradient

    x_advs = []
    rs = []
    changed = []
    preds = []
    preds_new = []
    nb_iters = []
    total_feats_used = []
    feat_indices = []
    original_label = []
    counterfactual_label = []
    counter_probs = []

    bar = range(len(test_x))
    bar = tqdm(range(len(test_x)), bar_format=(''))#'Generating Contrastive Sample...{percentage:3.0f}%')
    for i in bar:
        #print(f"i:{i}")
        x = test_x[i:i+1]
        x_var = Variable(torch.from_numpy(x)).type(torch.FloatTensor)
        lb_org, lb_new, x_adv, feats_idx, fin_probs = generate(x_var, model, gen_model, args,
                                                               scaler=scaler, trainer=trainer, **kargs)
        original_label.append(lb_org)
        counterfactual_label.append(lb_new)
        total_feats_used.append(len(feats_idx))
        x_advs.append(x_adv)
        changed.append(lb_new != lb_org)
        feat_indices.append(feats_idx)
        counter_probs.append(fin_probs)

    x_advs = np.array(x_advs)
    avg_feat_changed = np.mean(total_feats_used)
    vals, counts = np.unique(changed, return_counts=True)

    num_correct = counts[np.where(vals == True)[0]]
    fidelity = num_correct/(len(changed))*1.0
    counterfactual_x = x_advs

    results = {'avg_feat_changed': avg_feat_changed,
               'fidelity': fidelity,
               'counterfactual_x': counterfactual_x,
               'counterfactual_label': counterfactual_label,
               'original_label': original_label,
               'counter_probs': counter_probs,
               }

    return results
