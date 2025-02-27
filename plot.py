import pickle
import matplotlib.pyplot as plt
import os


def save_log(output_file, accs):
    file = open(output_file, 'w')
    for acc in accs:
        file.write(str(acc[0]) + " " + str(acc[1]) + "\n")
    file.close()


def plot_a(axs, dict_val, x_len, err_type):
    for key in dict_val.keys():
        plt_vals = dict_val[key][err_type]
        axs.plot(list(range(1, len(plt_vals) + 1)), plt_vals, label=key)
    if err_type == 1:
        axs.set_xlabel('# of Iterations')
    axs.set_xlim(1, x_len)


def read_file(file_name):
    data_pts_A1 = []
    data_pts_A2 = []

    with open(file_name, 'r') as f:
        for i in f.readlines():
            a1, a2 = i.split(" ")
            data_pts_A1.append(float(a1))
            data_pts_A2.append(float(a2))
    return [data_pts_A1, data_pts_A2]


def plot_a1_a2(config):
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    for i, corr_ratio in enumerate([0.1, 0.3, 0.6]):
        data_dict = {}
        data_path = f"results_b{config['batch_size']}{config['demo']}/corr_ratio_{corr_ratio}/clin_frac_{config['client_fractions']}/train_losses"

        for file in os.listdir(data_path):
            data = read_file(os.path.join(data_path, file))
            if 'fed_avg' in file:
                data_dict['FedAVG'] = data
            elif 'fed_prox' in file:
                data_dict['FedProx'] = data
            elif 'cwt' in file:
                data_dict['cwt'] = data
            elif 'fed_miss' in file:
                data_dict['FedMiss'] = data
            elif 'ditto' in file:
                data_dict['DITTO'] = data
            elif 'pw' in file:
                data_dict['PW'] = data
            elif 'simp_avg' in file:
                data_dict['SimpAvg'] = data
            elif 'baseline' in file:
                A1, A2 = data[0], data[1]
                A1 = [A1[0] for _ in range(config['total_iterations'])]
                A2 = [A2[0] for _ in range(config['total_iterations'])]
                data = [A1, A2]
                data_dict['baseline'] = data

        plot_a(axs[0, i], data_dict, config["total_iterations"], err_type=0)
        axs[0, i].set_title(f'Rounds with \n{corr_ratio * 100}% Corruption Ratio')
        plot_a(axs[1, i], data_dict, config["total_iterations"], err_type=1)

    for i, ax in enumerate(axs.flat):
        if i == 0:
            ax.set(ylabel=f'A1 (MSE)\n10% Missing Values in Validation')
        if i == 3:
            ax.set(ylabel=f'A2 (MSE)\n10% Missing Values in Validation')

    handles, labels = axs.flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels)
    plt.savefig(f"results_b{config['batch_size']}{config['demo']}/clin_frac_{config['client_fractions']}_train_plot.png")
    plt.show()


def plot_roc_pr(config):
    with open(f"results_b{config['batch_size']}{config['demo']}/corr_ratio_{config['corr_ratio']}/clin_frac_{config['client_fractions']}/downstream_clf.pkl", 'rb') as f:
        data = pickle.load(f)
    data = data[config['downstream_column']]

    _, ax_roc = plt.subplots(figsize=(6, 6))
    _, ax_pr = plt.subplots(figsize=(6, 6))
    for k in data.keys():
        ax_roc.plot(data[k]['fpr'], data[k]['tpr'], label=k + r" (AUC = %0.3f $\pm$ %0.3f)" % (data[k]['auc'], data[k]['auc_std']))
        ax_pr.step(data[k]['recall'], data[k]['precision'], label=k + r" (AUC = %.3f $\pm$ %0.3f)" % (data[k]['pr_auc'], data[k]['pr_auc_std']))
    ax_roc.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"ROC curve with variability\n(Positive label '{config['downstream_column']}')",
    )
    ax_pr.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="Recall",
        ylabel="Precision",
        title=f"PR curve with variability\n(Positive label '{config['downstream_column']}')",
    )
    ax_roc.legend(loc=4, fontsize='small')
    ax_roc.figure.savefig(f"results_b{config['batch_size']}{config['demo']}/corr_ratio_{config['corr_ratio']}/clin_frac_{config['client_fractions']}/{config['downstream_column']}_roc.png")

    ax_pr.legend(loc='lower left', fontsize='small')
    ax_pr.figure.savefig(f"results_b{config['batch_size']}{config['demo']}/corr_ratio_{config['corr_ratio']}/clin_frac_{config['client_fractions']}/{config['downstream_column']}_pr.png")
    plt.show()
