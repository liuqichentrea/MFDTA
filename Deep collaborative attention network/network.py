import torch
from torch import nn
import torch.nn.functional as F
from general_transformer import GenearalTransformer, Highway
import numpy as np
from model.rnn_attention import DeepLSTM


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.01, emb_name='encoder.embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='encoder.embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


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

        return out


class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_size):
        super(CrossAttentionFusion, self).__init__()
        self.feature_size = feature_size

        # 定义权重参数
        self.weight1 = nn.Linear(feature_size, 1)
        self.weight2 = nn.Linear(feature_size, 1)
        self.weight3 = nn.Linear(feature_size, 1)
        self.weight4 = nn.Linear(feature_size, 1)


    def forward(self, feature1, feature2, feature3, feature4):
        # 计算注意力权重
        attention1 = self.weight1(feature1)
        attention2 = self.weight2(feature2)
        attention3 = self.weight3(feature3)
        attention4 = self.weight4(feature4)

        # 特征加权融合
        fused_feature = torch.cat([attention1 * feature1,
                                   attention2 * feature2,
                                   attention3 * feature3,
                                   attention4 * feature4], dim=1)

        with open('atten_weights.txt', 'a') as f:
            f.write(f'{attention1[0][0].item()}, {attention2[0][0].item()}, {attention3[0][0].item()}, {attention4[0][0].item()}\n')

        return fused_feature


class TransformerDTA(nn.Module):
    def __init__(self,
                model_drug_atom,
                model_prot_atom, model_prot_token,
                rnn_model,
                d_model=128, reduction='mean'):
        super(TransformerDTA, self).__init__()

        self.model_drug_atom = model_drug_atom

        self.model_prot_atom = model_prot_atom
        self.model_prot_token = model_prot_token

        self.rnn_model = rnn_model

        assert reduction in ['mean', 'cls', 'max'], 'Invalid reduction mode'
        self.reduction = reduction

        self.chs = 4

        self.layernorms = nn.LayerNorm(self.chs * d_model * 3, eps=1e-6)

        self.linear_predict = nn.Sequential(
            nn.Linear(self.chs * d_model * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

        # activation and regularization
        dropout = 0.2
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.highway = Highway(self.chs, self.chs * d_model * 3)
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


        self.atten = CrossAttentionFusion(feature_size=384)

    def _embed(self, x, mask):
        if self.reduction == 'mean':
            x = x.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1)[:, None]
        elif self.reduction == 'max':
            x = x.masked_fill(~mask[..., None], 0).max(1)[0]
        else:
            x = x[:, 0, :]
        return x

    def forward(self,
                data, target,
                prot_atom, prot_atom_mask,
                prot_token, prot_token_mask,
                feat_clip
                ):
        # part1
        prot_atom_x = self.model_prot_atom(prot_atom, prot_atom_mask)
        prot_atom_x = prot_atom_x.transpose(1, 2)
        prot_atom_x = self.prot_cnn_atom1(prot_atom_x)
        prot_atom_x = self.prot_cnn_atom2(prot_atom_x)
        prot_atom_x = self.prot_cnn_atom3(prot_atom_x)
        prot_atom_x = torch.max(prot_atom_x, dim=2)[0]

        prot_token_x = self.model_prot_token(prot_token, prot_token_mask)
        prot_token_x = prot_token_x.transpose(1, 2)
        prot_token_x = self.prot_cnn_token1(prot_token_x)
        prot_token_x = self.prot_cnn_token2(prot_token_x)
        prot_token_x = self.prot_cnn_token3(prot_token_x)
        prot_token_x = torch.max(prot_token_x, dim=2)[0]

        #part2
        drug_atom_x = self.model_drug_atom(data, target)

        #part3
        clip_x = self.rnn_model(feat_clip)

        #融合p1+p2+p3
        # embedding = torch.cat([
        #     drug_atom_x,
        #     prot_atom_x,
        #     prot_token_x,
        #     clip_x
        # ], axis=-1)

        #基于注意力机制融合p1+p2+p3
        embedding = self.atten(
            drug_atom_x,
            prot_atom_x,
            prot_token_x,
            clip_x
        )

        embedding = self.layernorms(embedding)
        embedding = self.highway(embedding)

        out = self.linear_predict(embedding)

        return out


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