import torch
from utils import *
from dataset import *
import numpy as np

import faulthandler
from torch_geometric.loader import DataLoader
from torch import nn
import torch.nn.functional as F
from general_transformer import GenearalTransformer, Highway
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from transformer_pytorch import FeedForward
from model.gat import GATNet as gat
from torchsummary import summary
faulthandler.enable()
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
#6666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666
class CNNSEBlock(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding=1):
        super(CNNSEBlock, self).__init__()
        # the first
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)

        # the first SE layers
        self.fc1 = nn.Conv1d(planes, planes // 4, kernel_size=1)  # Use nn.Conv1d instead of nn.Linear
        self.fc2 = nn.Conv1d(planes // 4, planes, kernel_size=1)
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        #         # Squeeze
        #         w = F.avg_pool1d(out, kernel_size=out.size(2))
        #         w = F.relu(self.fc1(w))
        #         w = torch.sigmoid(self.fc2(w))
        #         # Excitation
        #         out = out * w
        return out

class TransformerDTA(nn.Module):
    def __init__(self,
                  model_drug_atom, 
                  model_prot_atom, model_prot_token, d_model=128, reduction='mean'):
        super(TransformerDTA, self).__init__()
        self.model_drug_atom = model_drug_atom
        self.model_prot_atom = model_prot_atom
        self.model_prot_token = model_prot_token

        #         self.classifier = nn.Linear(4 * d_model, 1)
        assert reduction in ['mean', 'cls', 'max'], 'Invalid reduction mode'
        self.reduction = reduction

        self.layernorms = nn.LayerNorm(3 * d_model * 3, eps=1e-6)

        self.linear_predict = nn.Sequential(
            nn.Linear(3* d_model * 3, 1024),
            #             nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            #             nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            #             nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

        # activation and regularization
        dropout = 0.2
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.highway = Highway(3, 3* d_model * 3)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.prot_cnn_atom1 = CNNSEBlock(d_model, d_model * 2)
        self.smi_cnn_atom1 = CNNSEBlock(d_model, d_model * 2)

        self.prot_cnn_atom2 = CNNSEBlock(d_model * 2, d_model * 2)
        self.smi_cnn_atom2 = CNNSEBlock(d_model * 2, d_model * 2)

        self.prot_cnn_atom3 = CNNSEBlock(d_model * 2, d_model * 3)
        self.smi_cnn_atom3 = CNNSEBlock(d_model * 2, d_model * 3)

        self.prot_cnn_token1 = CNNSEBlock(d_model, d_model * 2)
        self.smi_cnn_token1 = CNNSEBlock(d_model, d_model * 2)

        self.prot_cnn_token2 = CNNSEBlock(d_model * 2, d_model * 2)
        self.smi_cnn_token2 = CNNSEBlock(d_model * 2, d_model * 2)

        self.prot_cnn_token3 = CNNSEBlock(d_model * 2, d_model * 3)
        self.smi_cnn_token3 = CNNSEBlock(d_model * 2, d_model * 3)

    def _embed(self, x, mask):
        if self.reduction == 'mean':
            x = x.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1)[:, None]
        elif self.reduction == 'max':
            x = x.masked_fill(~mask[..., None], 0).max(1)[0]
        else:
            x = x[:, 0, :]
        return x

    def forward(self, 
                data,target,
                prot_atom, prot_atom_mask,
                prot_token, prot_token_mask):
        prot_atom_x = self.model_prot_atom(prot_atom, prot_atom_mask)
        prot_atom_x = prot_atom_x.transpose(1, 2)
        prot_atom_x = self.prot_cnn_atom1(prot_atom_x)
        prot_atom_x = self.prot_cnn_atom2(prot_atom_x)
        prot_atom_x = self.prot_cnn_atom3(prot_atom_x)
        prot_atom_x = torch.max(prot_atom_x, dim=2)[0]
        prot_token_x = self.model_prot_token(prot_token, prot_token_mask)
        print(prot_token_x.shape)
        prot_token_x = prot_token_x.transpose(1, 2)
        prot_token_x = self.prot_cnn_token1(prot_token_x)
        prot_token_x = self.prot_cnn_token2(prot_token_x)
        prot_token_x = self.prot_cnn_token3(prot_token_x)
        prot_token_x = torch.max(prot_token_x, dim=2)[0]
        # drug_atom_x = self.model_drug_atom(x, edge_index)
        # drug_atom_x = drug_atom_x.transpose(1, 2)
        # drug_atom_x = self.smi_cnn_atom1(drug_atom_x)
        # drug_atom_x = self.smi_cnn_atom2(drug_atom_x)
        # drug_atom_x = self.smi_cnn_atom3(drug_atom_x)
        # drug_atom_x = torch.max(drug_atom_x, dim=2)[0]
        # print(prot_atom_x.shape)
        # print(prot_token_x.shape)
        drug_atom_x=self.model_drug_atom(data,target)
        embedding = torch.cat([
            drug_atom_x, 
            prot_atom_x, prot_token_x], axis=-1)
        # torch.Size([16, 768])
        # print(embedding.shape)
        #         embedding = torch.cat([drug_atom_x, drug_token_x, prot_atom_x, prot_token_x], axis=-1)
        embedding = self.layernorms(embedding)
        embedding = self.highway(embedding)

        out = self.linear_predict(embedding)

        return out
from sklearn.metrics import average_precision_score, auc, precision_recall_curve
def get_aupr(Y, P, threshold=7.0):
    # print(Y.shape,P.shape)
    Y = np.where(Y >= threshold, 1, 0)
    P = np.where(P >= threshold, 1, 0)
    aupr = average_precision_score(Y, P)
    precision, recall, thresholds = precision_recall_curve(Y, P)
    # Use AUC function to calculate the area under the curve of precision recall curve
    auc_precision_recall = auc(recall, precision)

    return auc_precision_recall
MODEL_TYPE = 'transformer'
EPOCHS = 350
BATCH_SIZE = 16
VOCAB_SIZE_drug_atom = 67+1
VOCAB_SIZE_prot_atom = 25 + 1
# VOCAB_SIZE_drug_token = 23532+ 1
# VOCAB_SIZE_prot_atom = 16693+ 1
VOCAB_SIZE_drug_token = 1000+ 1
n_layer = 1
n_head  = 4
d_model = 128
N_HEADS = n_head
N_LAYERS = n_layer
D_MODEL = d_model
prot_len_atom = 1024
prot_len_token = 512
VOCAB_SIZE_prot_token = 10000+ 1
reduction = 'max'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
drug_graphDTA=gat().to(device)
prot_atom_transformer = GenearalTransformer(MODEL_TYPE, VOCAB_SIZE_prot_atom, prot_len_atom, D_MODEL, N_HEADS,
                                            N_LAYERS, device)
prot_token_transformer = GenearalTransformer(MODEL_TYPE, VOCAB_SIZE_prot_token, prot_len_token, D_MODEL,
                                                N_HEADS, N_LAYERS, device)
prot_atom_transformer = prot_atom_transformer.to(device)
prot_token_transformer = prot_token_transformer.to(device)
model = TransformerDTA(
    drug_graphDTA, 
    prot_atom_transformer,
    prot_token_transformer,
    D_MODEL, 
    reduction=reduction).to(device)
all_drug = []
all_protein = []
all_Y = []
data_file = '/home/b519/lqc/MGDTA-main/datasets/davis-uniq-data.csv'
threshold = 7.0
def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))
def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))
def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))
def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)
with open(data_file, 'r') as f:
    all_lines = f.readlines()

    for line in all_lines:
        row = line.rstrip().split(',')
        all_drug.append(row[0])
        all_protein.append(row[1])
        all_Y.append(row[2])
kf = KFold(n_splits=5, shuffle=True)
lr = 1e-4
def get_mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse
def get_pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp
def get_cindex(Y, P):
    summ = 0
    pair = 0

    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair != 0:
        return summ / pair
    else:
        return 0
dataset = 'davis'
model_file_name = 'models/spm-nhead-' + str(N_HEADS) + '-nlayer-' + str(N_LAYERS) + '-d-model-' + str(D_MODEL) + dataset + '-' + MODEL_TYPE + '-lr-' + str(lr) + '.model'
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8)
early_stopping = EarlyStopping(patience=100, verbose=True, path=model_file_name)
for split, (train_index, test_index) in enumerate(kf.split(all_Y)):
        train_valid_protein_cv = np.array(all_protein)[train_index]
        train_valid_drug_cv = np.array(all_drug)[train_index]

        train_valid_Y_cv = np.array(all_Y)[train_index]

        test_protein_cv = np.array(all_protein)[test_index]

        test_drug_cv = np.array(all_drug)[test_index]

        test_Y_cv = np.array(all_Y)[test_index]

        train_size = train_valid_protein_cv.shape[0]

        valid_size = int(train_size / 5.0)  # ?

        train_protein_cv = train_valid_protein_cv[:train_size - valid_size]
        train_drug_cv = train_valid_drug_cv[:train_size - valid_size]
        train_Y_cv = train_valid_Y_cv[:train_size - valid_size]
        valid_protein_cv = train_valid_protein_cv[train_size - valid_size:]
        valid_drug_cv = train_valid_drug_cv[train_size - valid_size:]
        valid_Y_cv = train_valid_Y_cv[train_size - valid_size:]
        train_ds = pack_dataset2(train_drug_cv, train_protein_cv, train_Y_cv)
        valid_ds = pack_dataset2(valid_drug_cv, valid_protein_cv, valid_Y_cv)
        test_ds = pack_dataset2(test_drug_cv, test_protein_cv, test_Y_cv)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=True)
        for epoch in range(1, EPOCHS):
            model.train()
            losses = []
            for i, batch in enumerate(train_loader):
                


                data,target,target_atom,target_token,affinity=batch
                target_atom_mask=target_atom != 0
                target_token_mask=target_token != 0

                data,target , target_atom, target_token, target_atom_mask,target_token_mask,affinity = (x.to(device) for x in (data, target,target_atom, target_token,target_atom_mask,target_token_mask, affinity))
                logits = model(data,target,target_atom, target_atom_mask,target_token, target_token_mask)
                loss = criterion(logits.float(), affinity.float())
                # optimizer.zero_grad()
                losses.append(loss.item())
                loss.backward()
            #             torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        
            print('Average loss', np.mean(losses))
            model.eval()
            with torch.no_grad():
                preds = torch.Tensor()
                trues = torch.Tensor()

                for batch in tqdm(valid_loader):
                    data,target,target_atom,target_token,affinity=batch
                    data,target,target_atom,target_token,affinity = (x.to(device) for x in (
                    data,target,target_atom,target_token,affinity))
                    target_atom_mask, target_token_mask = target_atom != 0, target_token != 0
                    logits = model(data,target,target_atom, target_atom_mask,target_token, target_token_mask)
                    preds = torch.cat((preds, logits.cpu()), 0)
                    trues = torch.cat((trues, affinity.view(-1, 1).cpu()), 0)
                preds, trues = preds.numpy().flatten(), trues.numpy().flatten()
                rm2 = get_rm2(trues, preds)
                mse = get_mse(trues, preds)
                pearson = get_pearson(trues, preds)
                aupr = get_aupr(trues, preds, threshold=threshold)
                # ci_val = get_cindex(trues, preds)
                ci_val = get_cindex(np.float64(trues), np.float64(preds))
                early_stopping(ci_val, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                print('val mse:', mse)
                print('val pearson', pearson)

                print('val AUPR', aupr)
                print('val ci:', ci_val)
                print(f'Epoch: {epoch + 1}')