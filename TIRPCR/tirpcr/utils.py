import os
import shutil
import zipfile

import urllib.parse
import urllib.request

import torch
import torch.utils.data
from dataset import *
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import Counter, defaultdict

def maybe_download_and_unzip_file(file_url, file_name=None):
    if file_name is None:
        file_name = os.path.basename(file_url)

    if not os.path.exists(file_name):
        print(f'Downloading: {file_name}')

        with urllib.request.urlopen(file_url) as response, open(file_name, 'wb') as target_file:
            shutil.copyfileobj(response, target_file)

        print(f'Downloaded: {file_name}')

        file_extension = os.path.splitext(file_name)[1]
        if file_extension == '.zip':
            print(f'Extracting zip: {file_name}')
            with zipfile.ZipFile(file_name, 'r') as zip_file:
                zip_file.extractall('.')

    else:
        print(f'Exists: {file_name}')



def load_model(model_class, filename):
    def _map_location(storage, loc):
        return storage
    map_location = None
    if not torch.cuda.is_available():
        map_location = _map_location

    state = torch.load(str(filename), map_location=map_location)

    model = model_class(**state['model_params'])
    model.load_state_dict(state['model_state'])

    return model


def save_model(model, filename, model_params=None):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    state = {
        'model_params': model_params or {},
        'model_state': model.state_dict(),
    }

    torch.save(state, str(filename))


def cal_deviation(hidden_val, golds_treatment, logits_treatment, normalized=False):

    ones_idx, zeros_idx = np.where(golds_treatment == 1), np.where(golds_treatment == 0)
    hidden_val = np.asarray(hidden_val)
    hidden_treated, hidden_controlled = hidden_val[ones_idx], hidden_val[zeros_idx]
    if not normalized:
        logits_treatment = 1 / (1 + np.exp(-logits_treatment))
    p_T = len(ones_idx[0])/(len(ones_idx[0])+len(zeros_idx[0]))
    treated_w, controlled_w = p_T/logits_treatment[ones_idx], (1-p_T)/(1.-logits_treatment[zeros_idx])
    treated_w = np.clip(treated_w, a_min=np.quantile(treated_w, 0.01),
                               a_max=np.quantile(treated_w, 0.99))
    controlled_w = np.clip(controlled_w, a_min=np.quantile(controlled_w, 0.01),
                        a_max=np.quantile(controlled_w, 0.99))
    treated_w, controlled_w = np.reshape(treated_w, (len(treated_w),1)), np.reshape(controlled_w, (len(controlled_w),1))
    hidden_treated_w, hidden_controlled_w = np.multiply(hidden_treated, treated_w), np.multiply(hidden_controlled, controlled_w)

    hidden_treated_mu, hidden_treated_var = np.mean(hidden_treated, axis=0), np.var(hidden_treated, axis=0)
    hidden_controlled_mu, hidden_controlled_var = np.mean(hidden_controlled, axis=0), np.var(hidden_controlled, axis=0)
    VAR = np.sqrt((hidden_treated_var + hidden_controlled_var) / 2)
    hidden_deviation = np.abs(hidden_treated_mu - hidden_controlled_mu) / VAR
    hidden_deviation[np.isnan(hidden_deviation)] = 0
    max_unbalanced_original = np.max(hidden_deviation)

    hidden_treated_w_mu, hidden_treated_w_var = np.mean(hidden_treated_w, axis=0), np.var(hidden_treated_w, axis=0)
    hidden_controlled_w_mu, hidden_controlled_w_var = np.mean(hidden_controlled_w, axis=0), np.var(hidden_controlled_w, axis=0)
    VAR = np.sqrt((hidden_treated_w_var + hidden_controlled_w_var) / 2)
    hidden_deviation_w = np.abs(hidden_treated_w_mu - hidden_controlled_w_mu) / VAR
    hidden_deviation_w[np.isnan(hidden_deviation_w)] = 0
    max_unbalanced_weighted = np.max(hidden_deviation_w)

    plot(hidden_treated, hidden_controlled, 'original.png')
    plot(hidden_treated_w, hidden_controlled_w, 'weighted.png')

    return max_unbalanced_original, hidden_deviation, max_unbalanced_weighted, hidden_deviation_w


def plot(hidden_treated, hidden_controlled, save_file):

    tsne = TSNE(n_components=2)

    treated_embedded = tsne.fit_transform(hidden_treated)
    controlled_embedded = tsne.fit_transform(hidden_controlled)

    plt.figure()

    treated_x, treated_y = treated_embedded[:,0], treated_embedded[:,1]
    controlled_x, controlled_y = controlled_embedded[:,0], controlled_embedded[:,1]

    plt.scatter(treated_x, treated_y, alpha=0.8, c='red', edgecolors='none', s=30, label='treated')
    plt.scatter(controlled_x, controlled_y, alpha=0.8, c='blue', edgecolors='none', s=30, label='controlled')

    plt.legend()
    plt.savefig(save_file)


def cal_ATE(golds_treatment, logits_treatment, golds_outcome, normalized=False):

    ones_idx, zeros_idx = np.where(golds_treatment == 1), np.where(golds_treatment == 0)
    if not normalized:
        logits_treatment = 1 / (1 + np.exp(-logits_treatment))
    p_T = len(ones_idx[0]) / (len(ones_idx[0]) + len(zeros_idx[0]))
    treated_w, controlled_w = p_T / logits_treatment[ones_idx], (1 - p_T) / (1. - logits_treatment[zeros_idx])
    treated_w = np.clip(treated_w, a_min=np.quantile(treated_w, 0.01),
                        a_max=np.quantile(treated_w, 0.99))
    controlled_w = np.clip(controlled_w, a_min=np.quantile(controlled_w, 0.01), a_max=np.quantile(controlled_w, 0.99))
    treated_w, controlled_w = np.reshape(treated_w, (len(treated_w), 1)), np.reshape(controlled_w,
                                                                                     (len(controlled_w), 1))
    treated_outcome, controlled_outcome = golds_outcome[ones_idx], golds_outcome[zeros_idx]
    treated_outcome_w, controlled_outcome_w = treated_outcome * treated_w, controlled_outcome * controlled_w

    UncorrectedEstimator_EY1_val, UncorrectedEstimator_EY0_val = np.mean(treated_outcome), np.mean(controlled_outcome)
    ATE = UncorrectedEstimator_EY1_val-UncorrectedEstimator_EY0_val
    IPWEstimator_EY1_val, IPWEstimator_EY0_val=np.mean(treated_outcome_w), np.mean(controlled_outcome_w)
    ATE_w = IPWEstimator_EY1_val-IPWEstimator_EY0_val

    return (UncorrectedEstimator_EY1_val, UncorrectedEstimator_EY0_val, ATE), (
    IPWEstimator_EY1_val, IPWEstimator_EY0_val, ATE_w)


def get_cohort_size(data_dir):
    cohorts = os.listdir(data_dir)
    cohorts_size = dict()
    for cohort in cohorts:
        load_cohort = pickle.load(open(data_dir+cohort, 'rb'))
        cohorts_size[cohort]=len(load_cohort)
    pickle.dump(cohorts_size, open('./pickles/cohorts_size.pkl', 'wb'))


def load_cohort_size():
    return pickle.load(open('mymodel/pickles/cohorts_size.pkl', 'rb'))


def get_medi_CAD_indication():
    outputs = open('', 'w')
    out = set()
    with open('', 'r') as f:
        next(f)
        for row in f:
            row = row.split(',')
            if len(row) < 4:
                continue
            rx_id, drugname, icd = row[0], row[1], row[2]
            if icd[:3] in (''):
                if (rx_id, drugname) not in out:
                    outputs.write('{},{}\n'.format(rx_id, drugname))
                out.add((rx_id, drugname))

    outputs.close()


def plot_deviation(unbalanced_deviation, balanced_deviation, labels, save_plt):
    plt.figure(figsize=(12, 6), dpi=100)
    for i in range(len(unbalanced_deviation)):
        if i == 0:
            plt.scatter(unbalanced_deviation[i], i, color='sandybrown', s=50, label='unweighted')
            plt.scatter(balanced_deviation[i], i, color='skyblue', s=50, label='LR_IPW')
        else:
            plt.scatter(unbalanced_deviation[i], i, color='sandybrown', s=50)
            plt.scatter(balanced_deviation[i], i, color='skyblue', s=50)
        plt.plot([unbalanced_deviation[i], balanced_deviation[i]], [i, i], color='dimgray')

    plt.yticks(range(len(unbalanced_deviation)), labels)
    plt.legend()
    plt.xlabel('Absolute Standard Mean Difference')
    plt.ylabel('Covariates')
    plt.tight_layout()
    plt.savefig(save_plt)