from src.benchmarks.GRACE_method.explain import *
from src.benchmarks.GRACE_method.fcn import *
from src.benchmarks.GRACE_method.grace import *
from src.benchmarks.GRACE_method.methods_refactored import NaiveGradient

from src.benchmarks.GRACE_method.selector import *
from src.benchmarks.GRACE_method.trainer import *
from src.benchmarks.GRACE_method.utils import *


class arguments():

    def __init__(self,
                 csv="spam.csv",
                 hiddens=[50, 30],
                 lr=0.001,
                 gen_gamma=0.5,
                 gen_max_features=5,
                 explain_units="points",
                 seed=77,
                 pre_scaler=0,
                 model_scaler=1,
                 lr_reduce_rate=0.1,
                 patience=5,
                 max_epochs=500,
                 batch_size=128,
                 model_temp_path="model_persistance/model_test_1",
                 alpha=1.0,
                 num_normal_feat=3,
                 explain_text=1,
                 gen_overshoot=0.0001,
                 gen_max_iter=50,
                 explain_table=1,
                 verbose_threshold=50):

        self.csv=csv
        self.hiddens=hiddens
        self.lr=lr
        self.gen_gamma=gen_gamma
        self.gen_max_features=gen_max_features
        self.explain_units=explain_units
        self.seed=seed
        self.pre_scaler=pre_scaler
        self.model_scaler=model_scaler
        self.lr_reduce_rate=lr_reduce_rate
        self.patience=patience
        self.max_epochs=max_epochs
        self.batch_size=batch_size
        self.model_temp_path=model_temp_path
        self.alpha=alpha
        self.num_normal_feat=num_normal_feat
        self.explain_text=explain_text
        self.gen_overshoot=gen_overshoot
        self.gen_max_iter=gen_max_iter
        self.explain_table=explain_table
        self.verbose_threshold=verbose_threshold


def load_model(train_data, args):
    num_feat = train_data.getX().shape[1]
    num_class = len(np.unique(train_data.gety()))
    scaler = StandardScaler(with_std=True)
    scaler.fit(train_data.getX())
    stds = np.sqrt(scaler.var_)
    if args.model_scaler:
        model = FCN(num_feat, num_class, args.hiddens, scaler.mean_, stds)
    else:
        model = FCN(num_feat, num_class, args.hiddens)
    return model


def train(args):
    # load data and model
    scaler, le, _, _, features, train_data, val_data, test_data = read_data(
        args.csv, args.seed, scaler=args.pre_scaler)
    model = load_model(train_data, args)

    # train and test the model
    trainer = Trainer(model, lrate=args.lr, lr_reduce_rate=args.lr_reduce_rate)
    trainer.train(train_dataset=train_data,
                  val_dataset=val_data,
                  patience=args.patience,
                  num_epochs=args.max_epochs,
                  batch_size=args.batch_size)

    torch.save(model.state_dict(), args.model_temp_path)

    _, val_acc, val_f1, val_pred = trainer.validate(val_data)
    _, test_acc, test_f1, test_pred = trainer.validate(test_data)
    print_performance(val_acc, val_f1, test_acc, test_f1)

def test(args):
    # load data and model
    scaler, le, _, _, features, train_data, val_data, test_data = read_data(
        args.csv, args.seed, scaler=args.pre_scaler)
    model = load_model(train_data, args)
    model.load_state_dict(torch.load(args.model_temp_path))
    trainer = Trainer(model)

    # configurations for generating explanation
    num_feat = train_data.getX().shape[1]
    bound_min, bound_max, bound_type = get_constraints(train_data.getX())
    alphas = args.alpha * \
        np.ones(num_feat) if args.alpha > 0 else np.std(train_data.getX(), axis=0)
    feature_selector = FeatureSelector(train_data.getX(), args.gen_gamma) if args.gen_gamma > 0.0 else None
    avg_feat_changed, fidelity, counterfactual_x, counterfactual_label, original_label = test_grace(model,
                                            trainer,
                                            test_data.getX(),
                                            args,
                                            method="Naive",
                                            scaler=scaler,
                                            bound_min=bound_min,
                                            bound_max=bound_max,
                                            bound_type=bound_type,
                                            alphas=alphas,
                                            feature_selector=feature_selector)

    print_results(avg_feat_changed, fidelity)


def explain(args):
    # load data and model
    scaler, le, _, _, features, train_data, val_data, test_data = read_data(
        args.csv, args.seed, scaler=args.pre_scaler)
    model = load_model(train_data, args)
    model.load_state_dict(torch.load(args.model_temp_path))
    trainer = Trainer(model)

    # load generation model
    gen_model = NaiveGradient

    # configurations for generating explanation
    num_feat = train_data.getX().shape[1]
    bound_min, bound_max, bound_type = get_constraints(train_data.getX())
    alphas = args.alpha * \
        np.ones(num_feat) if args.alpha > 0 else np.std(train_data.getX(), axis=0)
    feature_selector = FeatureSelector(train_data.getX(), args.gen_gamma) if args.gen_gamma > 0.0 else None

    # generate explanation on a random sample from test set    
    lb_new = lb_org = 0
    while lb_new == lb_org:
        i = np.random.choice(len(test_data.getX())) # select a random sample from test set
        x = test_data.getX()[i:i+1][0]
        x_var = Variable(torch.from_numpy(x.reshape(1,-1))).type(torch.FloatTensor)

        lb_org, lb_new, x_adv, feats_idx = generate(x_var, model, gen_model, args,
                                                    scaler=scaler, 
                                                    trainer=trainer,
                                                    bound_min=bound_min,
                                                    bound_max=bound_max,
                                                    bound_type=bound_type,
                                                    alphas=alphas,
                                                    feature_selector=feature_selector)

    # show explanation
    # print(features[feats_idx])
    if scaler:
        x = scaler.inverse_transform(x.reshape(1, -1))[0]
    if args.explain_table:
        explain_table(x, x_adv, lb_org, lb_new, feats_idx, features, args.num_normal_feat)
    if args.explain_text:
        explain_text(x, x_adv, lb_org, lb_new, feats_idx, features, units=args.explain_units)