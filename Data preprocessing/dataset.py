import sentencepiece as spm
import torch
from torch_geometric.data import Dataset, Data
from rdkit import Chem
import networkx as nx
from utils import *
from tqdm import tqdm
import warnings
import torch.nn.functional as F

# 忽略警告
warnings.filterwarnings("ignore")


sp_prot = spm.SentencePieceProcessor()
sp_prot.Load("protein_sent_token.model")
 
    
sp_smi = spm.SentencePieceProcessor()
sp_smi.Load("smi_sent_token.model")




class DTADataset(Dataset):
    def __init__(self, datas, prot_features_atom, 
                            prot_features_token,affinities):
        self.datas = datas
        self.prot_features_atom = prot_features_atom
        self.prot_features_token = prot_features_token
        self.affinities = affinities
    def __len__(self):
        return len(self.affinities)

    def __getitem__(self, index):
        data = self.datas[index]
        prot_feature_atom = self.prot_features_atom[index]
        prot_feature_token = self.prot_features_token[index]
        affinitie = self.affinities[index]
        return data,torch.FloatTensor(prot_feature_atom.astype(float)),torch.LongTensor(prot_feature_token.astype(float)),torch.LongTensor(affinitie.astype(float))



    def protein2emb_encoder(x):
        max_p = prot_len_token
        t1 = pbpe.process_line(x).split()  # split
        try:
            i1 = np.asarray([words2idx_p[i] for i in t1])  # index
        except:
            i1 = np.array([0])
            #print(x)

        l = len(i1)

        if l < max_p:
            i = np.pad(i1, (0, max_p - l), 'constant', constant_values = 0)
            input_mask = ([1] * l) + ([0] * (max_p - l))
        else:
            i = i1[:max_p]
            input_mask = [1] * max_p

        return i #, np.asarray(input_mask)

def drug2emb_encoder(x):
    max_d = drug_len_token
    #max_d = 100
    t1 = dbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        #print(x)
    
    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i #, np.asarray(input_mask)

CHARISOSMISET = {
"=":1,
"-":2,
"/":3,
".":4,
"(":5,
")":6,
"[":7,
"]":8,
"@":9,
"*":10,
"\\":11,
"#":12,
"%":13,
"+":14,
"0":15,
"1":16,
"2":17,
"3":18,
"4":19,
"5":20,
"6":21,
"7":22,
"8":23,
"9":24,
"a":25,
"A":26,
"b":27,
"B":28,
"c":29,
"C":30,
"d":31,
"D":32,
"e":33,
"E":34,
"f":35,
"F":36,
"g":37,
"G":38,
"h":39,
"H":40,
"i":41,
"I":42,
"K":43,
"l":44,
"L":45,
"m":46,
"M":47,
"n":48,
"N":49,
"o":50,
"O":51,
"p":52,
"P":53,
"r":54,
"R":55,
"s":56,
"S":57,
"t":58,
"T":59,
"u":60,
"U":61,
"V":62,
"W":63,
"X":64,
"y":65,
"Y":66,
"Z":67
}


CHARISOSMILEN = len(CHARISOSMISET)



def label_smiles(line, MAX_SMI_LEN):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]): #	x, smi_ch_ind, y
        X[i] = CHARISOSMISET[ch]

    return X #.tolist()


CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }

CHARPROTLEN = 25

def label_sequence(line, MAX_SEQ_LEN):
    
    X = np.zeros(MAX_SEQ_LEN)

    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = CHARPROTSET[ch]

    return X #.tolist()


# for reformer
drug_len_atom = 256
prot_len_atom = 1024
drug_len_token = 128
prot_len_token = 512

drug_len_token_spm = 128
prot_len_token_spm = 512


def spm_token_smiles(line, MAX_SMI_LEN):
    X = np.zeros(MAX_SMI_LEN)
    ids = sp_smi.EncodeAsIds(line)
    if len(ids) > MAX_SMI_LEN:   
        X = ids[:MAX_SMI_LEN]
    else:
        X[:len(ids)]= ids
    return X #.tolist()

def spm_token_prot(line, MAX_SMI_LEN):
    X = np.zeros(MAX_SMI_LEN)
    ids = sp_prot.EncodeAsIds(line)
    if len(ids) > MAX_SMI_LEN:   
        X = ids[:MAX_SMI_LEN]
    else:
        X[:len(ids)]= ids
    return X #.tolist()


def pack_dataset(drug, target, Y):
    drug_features_atom = np.zeros((len(drug), drug_len_atom))
    prot_features_atom = np.zeros((len(target), prot_len_atom))
    drug_features_token = np.zeros((len(drug), drug_len_token))
    prot_features_token = np.zeros((len(target), prot_len_token))

    for i, d in enumerate(drug) :
        drug_features_atom[i] = label_smiles( d,drug_len_atom)
        drug_features_token[i] = drug2emb_encoder( d)
        
    for i, p in enumerate(target):
        prot_features_atom[i] = label_sequence(p,prot_len_atom)
        prot_features_token[i] = protein2emb_encoder(p)

    affinities = np.array(Y).reshape(-1, 1)
    dta_dataset = DTADataset(drug_features_atom, prot_features_atom, 
                            drug_features_token,prot_features_token,affinities)   
    return dta_dataset


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return features, edge_index, c_size


seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
max_seq_len = 1000

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x



def pack_dataset2(drug, target, Y):
    datas = []
    affinities = np.array(Y).reshape(-1, 1)

    for i, p in tqdm(enumerate(drug), total=len(drug), leave=True):
        feature, edge_index, c_size = smile_to_graph(p)

        feature = torch.as_tensor(np.array(feature))

        edge_index = torch.LongTensor(edge_index).transpose(1, 0)
        targ = seq_cat(target[i])

        max_len = 200

        padding_x = max_len-feature.size(0)
        feat_clip = F.pad(feature, pad=(0,0, 0,padding_x))

        prot_features_atom = label_sequence(target[i], prot_len_atom)
        prot_features_token = spm_token_prot(target[i], prot_len_token)


        data = Data(
            x=feature, \
            edge_index=edge_index, \
            prot_features_atom=torch.FloatTensor(np.array(prot_features_atom).astype(float)),\
            prot_features_token=torch.FloatTensor(np.array(prot_features_token).astype(float)),\
            y=torch.FloatTensor(affinities[i].astype(float)), \
            target=torch.FloatTensor(targ.astype(float)),
            feat_clip = feat_clip
        )

        datas.append(data)

    return CustomGraphDataset(datas)




class CustomGraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        super(CustomGraphDataset, self).__init__(None, transform=None, pre_transform=None)
        
    def len(self):
        return len(self.data_list)

    def get(self, idx):
        # 根据索引 idx，返回一个 `torch_geometric.data.Data` 对象，表示一个图形
        return Data(x=torch.tensor(self.data_list[idx].x, dtype=torch.float),\
                    edge_index=torch.tensor(self.data_list[idx].edge_index,dtype=torch.long)),\
                    torch.tensor(self.data_list[idx].target,dtype=torch.long),\
                    torch.tensor(self.data_list[idx].prot_features_atom,dtype=torch.long),\
                    torch.tensor(self.data_list[idx].prot_features_token,dtype=torch.long), \
               torch.tensor(self.data_list[idx].feat_clip, dtype=torch.float), \
        torch.tensor(self.data_list[idx].y, dtype=torch.float)


def pack_dataset3(drug, target, seq2go, Y):
    drug_features_atom = np.zeros((len(drug), drug_len_atom))
    prot_features_atom = np.zeros((len(target), prot_len_atom))
    drug_features_token = np.zeros((len(drug), drug_len_token))
    prot_features_token = np.zeros((len(target), prot_len_token))
#     go_emb = np.zeros(( len(target), 128, 768 ))
    
    for i, d in enumerate(drug) :
        drug_features_atom[i] = label_smiles( d,drug_len_atom)
        drug_features_token[i] = spm_token_smiles( d, drug_len_token)
        
    for i, p in enumerate(target):
        prot_features_atom[i] = label_sequence(p,prot_len_atom)
        prot_features_token[i] = spm_token_prot(p, prot_len_token)
#         go_emb[i] = seq2go[p]
        
    affinities = np.array(Y).reshape(-1, 1)
    dta_dataset = OntoDTADataset(drug_features_atom, prot_features_atom, 
                            drug_features_token,prot_features_token, seq2go,target, affinities)   
    return dta_dataset   

class OntoDTADataset(Dataset):
    def __init__(self, drugs_atom, targets_atom,
                       drugs_token, targets_token, seq2go, target, affinities):
        self.drugs_atom = drugs_atom
        self.targets_atom = targets_atom 
        self.drugs_token = drugs_token 
        self.targets_token = targets_token
        self.affinities = affinities
        self.seq2go = seq2go
        self.target = target
    
    def __len__(self):
        return len(self.affinities)

    def __getitem__(self, index):
        drug_atom = self.drugs_atom[index]
        target_atom = self.targets_atom[index]
        drug_token = self.drugs_token[index]
        target_token = self.targets_token[index]
        affinity = self.affinities[index]
        go_emb = self.seq2go[self.target[index]]
        return torch.LongTensor(drug_atom), torch.LongTensor(target_atom), torch.LongTensor(drug_token), torch.LongTensor(target_token),  torch.FloatTensor(go_emb), torch.FloatTensor(affinity)
    


 
