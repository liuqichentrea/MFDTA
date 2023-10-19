# 11111111111111111111111111111111111111111111111111111111111111111111111111111111111
from dataset import *
from torch import nn
from general_transformer import GenearalTransformer, Highway
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from model.gat import GATNet as gat
from model.ginconv import GINConvNet
from utils import *
import matplotlib.pyplot as plt
import torch
from my_metrics import *
from network import TransformerDTA
from sklearn.model_selection import KFold
from model.rnn_attention import DeepLSTM

dataset = 'davis'

if dataset == 'KIBA_S1':
    data_file = '/home/junjie/MultiViewDTA/data/kiba-subset1-data.csv'
    threshold = 12.1
elif dataset == 'davis':
    data_file = './datasets/davis-uniq-data.csv'
    threshold = 7.0



def evaluation(val_loader, model):
    model.eval()

    with torch.no_grad():
        preds = torch.Tensor()
        trues = torch.Tensor()

        for batch in tqdm(val_loader):
            data, target, target_atom, target_token, feat_clip, affinity = batch
            data, target, target_atom, target_token, feat_clip, affinity = (x.to(device) for x in (
                data, target, target_atom, target_token, feat_clip, affinity))

            target_atom_mask, target_token_mask = target_atom != 0, target_token != 0
            logits = model(data, target, target_atom, target_atom_mask, target_token, target_token_mask, feat_clip)
            preds = torch.cat((preds, logits.cpu()), 0)
            trues = torch.cat((trues, affinity.view(-1, 1).cpu()), 0)

        preds, trues = preds.numpy().flatten(), trues.numpy().flatten()

        rm2 = get_rm2(trues, preds)
        mse = get_mse(trues, preds)
        pearson = get_pearson(trues, preds)
        aupr = get_aupr(trues, preds, threshold=threshold)
        ci = get_cindex(np.float64(trues), np.float64(preds))

    return rm2, mse, pearson, aupr, ci

def train_valid(model, train_loader, val_loader, EPOCHS, criterion, optimizer, scheduler, model_file_name):
    mses = []
    pearsons = []
    auprs = []
    cis = []
    rm2s = []

    for epoch in range(1, EPOCHS):
        model.train()
        losses = []
        for i, batch in enumerate(train_loader):
            data, target, target_atom, target_token, feat_clip, affinity = batch

            target_atom_mask = target_atom != 0
            target_token_mask = target_token != 0

            data, target, target_atom, target_token, target_atom_mask, target_token_mask, feat_clip, affinity = \
                (x.to(device) for x in (data, target, target_atom, target_token, target_atom_mask, target_token_mask, feat_clip, affinity))


            logits = model(data, target, target_atom, target_atom_mask, target_token, target_token_mask, feat_clip)
            loss = criterion(logits.float(), affinity.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print('Average loss', np.mean(losses))

        train_rm2, train_mse, train_pearson, train_aupr, train_ci = evaluation(train_loader, model)
        rm2, mse, pearson, aupr, ci = evaluation(val_loader, model)

        mses.append(mse)
        pearsons.append(pearson)
        auprs.append(aupr)
        cis.append(ci)
        rm2s.append(rm2)

        print(f'train--epoch : {epoch + 1}, rm2: {train_rm2:.4f}, mse: {train_mse:.4f}, pearson: {train_pearson:.4f}, aupr: {train_aupr:.4f}, ci: {train_ci:.4f}')
        print(f'val----epoch : {epoch + 1}, rm2: {rm2:.4f}, mse: {mse:.4f}, pearson: {pearson:.4f}, aupr: {aupr:.4f}, ci: {ci:.4f}')
        print(f'best val     : {epoch + 1}, rm2: {np.max(rm2s):.4f}, mse: {np.min(mses):.4f}, pearson: {np.max(pearsons):.4f}, aupr: {np.max(auprs):.4f}, ci: {np.max(cis):.4f}')

        plt.cla()
        plt.plot(pearsons, label='pearson')
        plt.plot(auprs, label='aupr')
        plt.plot(cis, label='ci')
        plt.plot(rm2s, label='rm2')
        plt.legend()
        plt.savefig('figures/metrics')

        plt.cla()
        plt.plot(mses, label='mse')
        plt.legend()
        plt.savefig('figures/mse')

    return np.max(rm2s), np.min(mses), np.max(pearson), np.max(auprs), np.max(cis)


def run_grid_cv_4_model(lr, n_layer, n_head, d_model):
    N_HEADS = n_head
    N_LAYERS = n_layer
    D_MODEL = d_model

    all_ci2 = []
    all_mse = []
    all_r = []
    all_aupr = []
    all_rm2 = []


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

        #paer1 模型定义
        prot_atom_transformer = GenearalTransformer(MODEL_TYPE, VOCAB_SIZE_prot_atom, prot_len_atom, D_MODEL, N_HEADS,
                                                    N_LAYERS, device)
        prot_token_transformer = GenearalTransformer(MODEL_TYPE, VOCAB_SIZE_prot_token, prot_len_token, D_MODEL,
                                                     N_HEADS, N_LAYERS, device)

        #cuda
        prot_atom_transformer = prot_atom_transformer.to(device)
        prot_token_transformer = prot_token_transformer.to(device)

        #part2模型定义
        drug_graphDTA = GINConvNet().to(device)

        #part3模型定义
        rnn_model = DeepLSTM(input_size=78, hidden_size=128, output_size=384, num_layers=1)


        reduction = 'max'

        model_file_name = 'models/spm-nhead-' + str(N_HEADS) + '-nlayer-' + str(N_LAYERS) + '-d-model-' + str(
             D_MODEL) + dataset + '-' + MODEL_TYPE + '-lr-' + str(lr) + '.model'

        model = TransformerDTA(drug_graphDTA,

                               prot_atom_transformer,
                               prot_token_transformer,

                               rnn_model,

                               D_MODEL, reduction=reduction)

        model.to(device)

        #     optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
        #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8)

        criterion = nn.MSELoss()

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=EPOCHS
        )

        rm2, mse, pearson, aupr, ci = train_valid(model, train_loader, valid_loader, EPOCHS, criterion, optimizer, scheduler, model_file_name)


        all_mse.append(mse)
        all_r.append(pearson)
        all_aupr.append(aupr)
        all_rm2.append(rm2)
        all_ci2.append(ci)

    return all_mse, all_r, all_aupr, all_rm2, all_ci2



lr = 1e-4
n_layer = 1
n_head  = 4
d_model = 128


all_drug = []
all_protein = []
all_Y = []
with open(data_file, 'r') as f:
    all_lines = f.readlines()

    for line in tqdm(all_lines):
        row = line.rstrip().split(',')
        all_drug.append(row[0])
        all_protein.append(row[1])
        all_Y.append(row[2])


Y_s = all_Y.copy()
Y_s = np.array([float(y) for y in Y_s])

pos_num = np.sum(Y_s>threshold)
neg_num = np.sum(Y_s<=threshold)


miny = Y_s.min()
maxy = Y_s.max()
print(f'min_y : {miny}, max_y: {maxy}')
print(f'负样本数量: {neg_num}, 正样本数量: {pos_num}')

kf = KFold(n_splits=10, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = 'transformer'

EPOCHS = 400
BATCH_SIZE = 128

VOCAB_SIZE_drug_atom = 67+1
VOCAB_SIZE_prot_atom = 25 + 1
VOCAB_SIZE_drug_token = 1000+ 1
VOCAB_SIZE_prot_token = 10000+ 1


drug_len_atom = 256
prot_len_atom = 1024
drug_len_token = 128
prot_len_token = 512

all_mse2, all_r, all_aupr, all_rm2, all_ci2 = run_grid_cv_4_model(lr, n_layer, n_head, d_model)

print(all_mse2)

print('*='*20)
print('cindex:  {0:6f}({1:6f})'.format(np.mean(all_ci2),  np.std(all_ci2)))
print('mse:     {0:6f}({1:6f})'.format(np.mean(all_mse2), np.std(all_mse2)))
print('rm2:     {0:6f}({1:6f})'.format(np.mean(all_rm2),  np.std(all_rm2)))
print('pearson: {0:6f}({1:6f})'.format(np.mean(all_r),    np.std(all_r)))
print('AUPR:    {0:6f}({1:6f})'.format(np.mean(all_aupr), np.std(all_aupr)))
