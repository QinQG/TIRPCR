import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
import pickle
import os
from dataset import Dataset
from model import LSTMModel
from evaluation import model_eval, transfer_data
from utils import save_model, load_model

def parse_args():
    parser = argparse.ArgumentParser(description='Process parameters for training t he model')
    parser.add_argument('--data_dir', type=str, default="")
    parser.add_argument('--pickles_dir', type=str, default='')
    parser.add_argument('--treated_drug_file', type=str, default="")
    parser.add_argument('--save_model_filename', type=str, default=r"")
    parser.add_argument('--outputs_lstm', type=str, default="")
    parser.add_argument('--controlled_drug', choices=['random', 'Meridian Tropism'], default='random')
    parser.add_argument('--controlled_drug_ratio', type=int, default=2)
    parser.add_argument('--random_seed', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=70)
    parser.add_argument('--diag_emb_size', type=int, default=128)
    parser.add_argument('--med_emb_size', type=int, default=128)
    parser.add_argument('--med_hidden_size', type=int, default=128)
    parser.add_argument('--diag_hidden_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--cuda', action='store_true', help='Use GPU if available (default: False)')
    return parser.parse_args()

def main(args, plot_final=False):
    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('args: ', args)

    np.random.seed(args.random_seed)
    output_lstm = open(args.outputs_lstm, 'a')

    treated = pickle.load(open(args.data_dir + args.treated_drug_file + '.pkl', 'rb'))
    controlled = []

    if args.controlled_drug == 'Meridian Tropism':
        cohort_size = pickle.load(open(os.path.join(args.data_dir, 'cohorts_size.pkl'), 'rb'))
        controlled_drugs = list(set(os.listdir(args.data_dir)) - {args.treated_drug_file + '.pkl'})
        controlled_drugs = sorted(controlled_drugs)
        n_control_patient = 0
        selected_drugs = []
        n_treat_patient = cohort_size.get(args.treated_drug_file + '.pkl', 0)
        for c_drug in controlled_drugs[5:30]:
            n_control_patient += cohort_size.get(c_drug, 0)
            selected_drugs.append(c_drug)
            if n_control_patient >= (args.controlled_drug_ratio + 1) * n_treat_patient:
                print(f"Selected drugs: {selected_drugs}")
                break
        else:
            print("Even after selecting all remaining drugs, not enough patients to meet the ratio.")
    else:
        cohort_size = pickle.load(open(os.path.join(args.data_dir, 'cohorts_size.pkl'), 'rb'))
        controlled_drugs = list(set(os.listdir(args.data_dir)) - {args.treated_drug_file + '.pkl'})
        np.random.shuffle(controlled_drugs)
        n_control_patient = 0
        controlled_drugs_range = []
        n_treat_patient = cohort_size.get(args.treated_drug_file + '.pkl', 0)
        for c_id in controlled_drugs:
            n_control_patient += cohort_size.get(c_id, 0)
            controlled_drugs_range.append(c_id)
            if n_control_patient >= (args.controlled_drug_ratio + 1) * n_treat_patient:
                print(f"Selected drugs: {controlled_drugs_range}")
                break

    for c_drug_id in controlled_drugs_range:
        c = pickle.load(open(args.data_dir + c_drug_id, 'rb'))
        controlled.extend(c)

    print("Treated shape:", np.asarray(treated).shape)
    print("Controlled shape:", np.asarray(controlled).shape)

    intersect = set(np.asarray(treated)[:, 0]).intersection(set(np.asarray(controlled)[:, 0]))
    controlled = np.asarray([controlled[i] for i in range(len(controlled)) if controlled[i][0] not in intersect])

    controlled_indices = list(range(len(controlled)))
    controlled_sample_index = int(args.controlled_drug_ratio * len(treated))

    np.random.shuffle(controlled_indices)
    controlled_sample_indices = controlled_indices[:controlled_sample_index]
    controlled_sample = controlled[controlled_sample_indices]
    n_user, n_nonuser = len(treated), len(controlled_sample)

    print('user: {}, non_user: {}'.format(len(treated), len(controlled_sample)), flush=True)
    print("Constructed Dataset.", flush=True)

    my_dataset = Dataset(treated, controlled_sample)

    train_ratio = 0.7
    val_ratio = 0.1

    dataset_size = len(my_dataset)
    indices = list(range(dataset_size))
    train_index = int(np.floor(train_ratio * dataset_size))
    val_index = int(np.floor(val_ratio * dataset_size))

    np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[:train_index], indices[train_index:train_index + val_index], indices[train_index + val_index:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size, sampler=test_sampler)

    model_params = dict(
        med_hidden_size=args.med_hidden_size,
        diag_hidden_size=args.diag_hidden_size,
        hidden_size=100,
        bidirectional=True,
        med_vocab_size=len(my_dataset.med_code_vocab),
        diag_vocab_size=len(my_dataset.diag_code_vocab),
        diag_embedding_size=args.diag_emb_size,
        med_embedding_size=args.med_emb_size,
        end_index=my_dataset.diag_code_vocab.get('<END>', 0), 
        pad_index=my_dataset.diag_code_vocab.get('<PAD>', 0),
    )
    print(model_params, flush=True)

    model = LSTMModel(**model_params)

    if args.cuda:
        model = model.to('cuda')

    print(model, flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lowest_std = float('inf')
    all_true_labels = []
    all_pred_probs = []

    for epoch in range(args.epochs):
        epoch_losses_ipw = []
        _, golds_treatment, logits_treatment, _, _ = transfer_data(model, test_loader, cuda=args.cuda)
        probabilities = torch.sigmoid(torch.from_numpy(logits_treatment)).numpy()
        all_true_labels.extend(golds_treatment)
        all_pred_probs.extend(probabilities)
        for confounder, treatment, outcome in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            if args.cuda:
                confounder = [c.to('cuda') for c in confounder]
                treatment = treatment.to('cuda')
            treatment_logits, _ = model(confounder)
            loss_ipw = F.binary_cross_entropy_with_logits(treatment_logits, treatment.float())
            loss_ipw.backward()
            optimizer.step()
            epoch_losses_ipw.append(loss_ipw.item())

        epoch_losses_ipw = np.mean(epoch_losses_ipw)
        print('Epoch: {}, IPW train loss: {}'.format(epoch, epoch_losses_ipw), flush=True)

        loss_val, AUC_val, max_unbalanced, ATE = model_eval(model, val_loader, cuda=args.cuda)
        _, hidden_deviation, max_unbalanced_weighted, hidden_deviation_w = max_unbalanced
        print('Val loss_treament: {}'.format(loss_val), flush=True)
        print('Val AUC_treatment: {}'.format(AUC_val), flush=True)
        print('Val Max_unbalanced: {}'.format(max_unbalanced_weighted), flush=True)
        print('ATE_w: {}'.format(ATE[1][2]), flush=True)
        if max_unbalanced_weighted < lowest_std:
            save_model(model, args.save_model_filename, model_params=model_params)
            lowest_std = max_unbalanced_weighted

        if epoch % 5 == 0:
            loss_test, AUC_test, _, _ = model_eval(model, test_loader, cuda=args.cuda)
            print('Test loss_treament: {}'.format(loss_test))
            print('Test AUC_treatment: {}'.format(AUC_test))

    mymodel = load_model(LSTMModel, args.save_model_filename)
    mymodel.to(args.device)
    _, AUC, max_unbalanced, ATE = model_eval(mymodel, test_loader, cuda=args.cuda)
    max_unbalanced_original, hidden_deviation, max_unbalanced_weighted, hidden_deviation_w = max_unbalanced

    n_unbalanced_feature = len(np.where(hidden_deviation > 0.1)[0])
    n_unbalanced_feature_w = len(np.where(hidden_deviation_w > 0.1)[0])
    n_feature = my_dataset.med_vocab_length + my_dataset.diag_vocab_length + 2

    UncorrectedEstimator_EY1_val, UncorrectedEstimator_EY0_val, ATE_original = ATE[0]
    IPWEstimator_EY1_val, IPWEstimator_EY0_val, ATE_weighted = ATE[1]
    print('max_unbalanced_ori: {}, max_unbalanced_wei: {}'.format(max_unbalanced_original, max_unbalanced_weighted), flush=True)
    print('ATE_ori: {}, ATE_wei: {}'.format(ATE_original, ATE_weighted), flush=True)
    print('AUC_treatment: {}'.format(AUC), flush=True)
    print('n_unbalanced_feature: {}, n_unbalanced_feature_w: {}'.format(n_unbalanced_feature, n_unbalanced_feature_w), flush=True)
    print('unbalanced_bizhi:{},unbalanced_bizhi_w: {}'.format(n_unbalanced_feature / n_feature, n_unbalanced_feature_w / n_feature))
    output_lstm.write('{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(n_user, n_nonuser,
                                                                        max_unbalanced_original, max_unbalanced_weighted,
                                                                        n_unbalanced_feature, n_unbalanced_feature_w, n_feature,
                                                                        UncorrectedEstimator_EY1_val,
                                                                        UncorrectedEstimator_EY0_val,
                                                                        ATE_original, IPWEstimator_EY1_val,
                                                                        IPWEstimator_EY0_val,
                                                                        ATE_weighted))
    output_lstm.close()
    fpr, tpr, _ = roc_curve(all_true_labels, all_pred_probs)
    roc_auc = auc(fpr, tpr)

    standard_fpr = np.linspace(0, 1, 100)
    standard_tpr = np.power(standard_fpr, 0.3)  
    if plot_final:
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(all_true_labels, all_pred_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance line')
        plt.xlim([-0.1, 1.0])
        plt.ylim([-0.1, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True)
        plt.savefig('roc_curve.png', bbox_inches='tight')
        plt.show()

        # Plot confusion matrix
        y_pred = np.round(all_pred_probs)
        cm = confusion_matrix(all_true_labels, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-user', 'User'])
        disp.plot()
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')

    return ATE_weighted, AUC

if __name__ == "__main__":
    args = parse_args()
    ATE_wei = 0
    attempts = 0
    final_AUC = 0
    max_attempts = 30
    ATE_wei, final_AUC = main(args, plot_final=False)
    print(f"Final ATE_wei: {ATE_wei}, Final AUC: {final_AUC}")