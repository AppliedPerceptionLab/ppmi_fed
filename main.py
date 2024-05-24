from sklearn.preprocessing import MinMaxScaler

from downstream_analysis import learn_classification
from framework.cwt_pred import cwt_pred
from framework.ditto_pred import ditto_pred
from framework.fedavg_pred import fedavg_pred
from framework.fedmiss_pred import fedmiss_pred
from framework.fedprox_pred import fedprox_pred
from framework.pw_pred import pw_pred
from framework.simpavg_pred import simpavg_pred
from framework.baseline_pred import baseline_pred
from ppmi_data.sites import get_sites
from plot import save_log, plot_a1_a2, plot_roc_pr
from utils import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

config = {
    "SEED": 43,
    "gpu": 0,

    "NUM_FEATURES": 17,     # [17, 64]
    "IR_SIZE": 7,
    "layer_width": 10,
    "depth": 3,
    "drop_out": 0.1,

    "total_iterations": 150,
    "client_iterations": 50,
    "client_fractions": 0.2,  # [0.2, 0.5]

    "replication": True,

    "demo": "_demo",  # ["_demo", ""]
    "miss_ratio": 0.1,  # [0.1]
    "corr_ratio": 0.6,  # [0.1, 0.3, 0.6]
    "batch_size": 32,  # [16, 32]

    "imputation": True,  # [True, False]
    "fed_name": "fed_avg",  # ['baseline', 'cwt', 'ditto', 'fed_avg', 'fed_miss', 'fed_prox', 'pw', 'simp_avg']
    "na_impute": "mean",  # ['zero', 'mean']

    "lr": 1e-6,
    "alpha": 0.1,  # fed_prox
    "lambda": 0.1,  # ditto

    "downstream_column": 'updrs_totscore'  # ['updrs1_score', 'updrs2_score', 'updrs3_score', 'updrs_totscore']
    # SITE,PATNO,visit_name,age_at_baseline,SEX,HISPLAT,race,COHORT,ess,updrs1PQ_score,updrs1_score,updrs2_score,NHY,updrs3_score,updrs4_score,moca,MSEADLG,rem,upsit,updrstot_score
}


def run(train_datasets, valid_datasets, test_datasets):
    fed_solver = None
    if config['fed_name'] == "fed_avg":
        fed_solver = fedavg_pred
    elif config['fed_name'] == "fed_prox":
        fed_solver = fedprox_pred
    elif config['fed_name'] == "cwt":
        fed_solver = cwt_pred
    elif config['fed_name'] == "fed_miss":
        fed_solver = fedmiss_pred
    elif config['fed_name'] == "ditto":
        fed_solver = ditto_pred
    elif config['fed_name'] == "pw":
        fed_solver = pw_pred
    elif config['fed_name'] == "simp_avg":
        fed_solver = simpavg_pred
    elif config['fed_name'] == "baseline":
        fed_solver = baseline_pred

    model_weights = f"weights/new/weights_b{config['batch_size']}{config['demo']}/corr_ratio_{config['corr_ratio']}/clin_frac_{config['client_fractions']}/{config['fed_name']}.h5"

    if isfile(model_weights):
        print(f"{config['fed_name']} Loaded")
        server = torch.load(model_weights).to(device)
    else:
        loss_s, server = fed_solver(train_datasets, valid_datasets, test_datasets, config).train()
        if len(loss_s) == 1:
            loss_s = [loss_s[0] for _ in range(config['total_iterations'])]
        save_log(f"results/new/results_b{config['batch_size']}{config['demo']}/corr_ratio_{config['corr_ratio']}/clin_frac_{config['client_fractions']}/train_losses/{config['fed_name']}_train_loss.txt", loss_s)
        torch.save(server, model_weights)

    a1, a2 = calculate_a1_a2(server, [DataLoader(test_dataset, config['batch_size']) for test_dataset in test_datasets], device)
    update_test_losses(config, a1, a2)

    return server


def downstream(dataset, dataset_PATNO, server=None):
    if config['imputation']:
        dataset_org = dataset
        sc = MinMaxScaler(feature_range=(0, 1))
        columns = dataset.columns
        dataset = sc.fit_transform(dataset)
        dataset = impute(config, dataset, columns, server, device)
        dataset = sc.inverse_transform(dataset)
        dataset = pd.DataFrame(dataset, columns=columns)
        dataset = impute_nan(dataset, dataset_org)

    else:
        if config['na_impute'] == "zero":
            for c in dataset.columns:
                dataset[c].fillna(0, inplace=True)
        if config['na_impute'] == "mean":
            for c in dataset.columns:
                dataset[c].fillna(dataset[c].mean(skipna=True), inplace=True)

    dataset['PATNO'] = dataset_PATNO
    acc_mean, acc_std, f1_mean, f1_std = learn_classification(dataset, config)

    update_downstream_results(config, acc_mean, acc_std, f1_mean, f1_std)


if __name__ == "__main__":
    device = initialize(config)
    if not config['replication']:
        if config['demo'] == "":
            ppmi = pd.read_csv("ppmi_data/train_curated.txt", sep=',')
        else:
            ppmi = pd.read_csv("ppmi_data/train_curated_demo.txt", sep=',')
    else:
        ppmi = pd.read_csv("new_data/ppmi_summary.csv", sep=',')

    train_datasets, valid_datasets, test_datasets = get_sites(ppmi, config['replication'], config["miss_ratio"], config["corr_ratio"])

    ppmi_patno = ppmi['PATNO']
    ppmi = ppmi.drop(['SITE', 'PATNO', 'COHORT'], axis=1)
    ppmi = ppmi.dropna(subset=[config['downstream_column']])

    for fed_name in ['baseline', 'cwt', 'ditto', 'fed_avg', 'fed_miss', 'fed_prox', 'pw', 'simp_avg']:
        config['fed_name'] = fed_name
        print("Running Algorithm: {0}".format(config['fed_name']))
        server = run(train_datasets, valid_datasets, test_datasets)
        downstream(ppmi, ppmi_patno, server)

    config['imputation'] = False
    for na_impute in ['zero', 'mean']:
        config['na_impute'] = na_impute
        downstream(ppmi, ppmi_patno)

    plot_roc_pr(config)
    plot_a1_a2(config)
