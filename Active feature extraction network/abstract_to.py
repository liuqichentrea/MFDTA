import numpy as np
import bisect
import rdkit.Chem as Chem
import os

from rdkit import Chem
# Simple implementaion of a binary tree.

class Tree(object):

    def __init__(self, left=None, right=None, value=None):
        self.left = left
        self.right = right
        self.value = value


    def __str__(self):
        return self.to_string(0)


    __repr__ = __str__


    def to_string(self, n):
        s = ""
        space = "".join(["    "] * n)
        if not self.value is None:
            s = str(self.value)
        else:
            if not self.left is None:
                s +=  "----"  + self.left.to_string(n+1)
            if not self.right is None:
                s += "\n" + space + "`---" + self.right.to_string(n+1)

        return s



    def get_depth(self):
        d1, d2 = 0, 0
        if not self.left is None:
            d1 = 1 + self.left.get_depth()
        if not self.right is None:
            d2 = 1 + self.right.get_depth()

        return max(d1, d2)



    def encode_leafs(self):
        return self.encode_r([])


    def encode_r(self, encoding):
        if not self.value is None:
            return [(self.value, "".join(encoding))]

        else:
            if not self.left is None:
                encoding.append("0")
                l1 = self.left.encode_r(encoding)
                encoding.pop()
            else:
                l1 = []
            if not self.right is None:
                encoding.append("1")
                l2 = self.right.encode_r(encoding)
                encoding.pop()
            else:
                l2 = []

            return l1 + l2


def build_tree_from_list(l, lookup=None):

    return btl(l, len(l)-1, 0, lookup)



def btl(l, n, use, lookup):

    if n >= 0:
        pair = l[n][use]
        t1 = btl(l, n-1, pair[0], lookup)
        t2 = None if len(pair) == 1 else btl(l, n-1, pair[1], lookup)
        return Tree(t1, t2)

    else:
        if not lookup is None:
            use = lookup[use]
        return Tree(value=use)



# Read a file containing SMILES
# The file should be a .smi or a .csv where the first column should contain a SMILES string
def read_file(file_name, drop_first=True):
    
    molObjects = []

    with open(file_name) as f:
        for l in f:
            if drop_first:
                drop_first = False
                continue

            l = l.strip().split(",")[0]  #strip()表示删除掉数据中的换行符，split（‘，’）则是数据中遇到‘,’ 就隔开。把每行的每个字符一个个分开，变成一个list
            smi = drop_salt(l.strip())
            molObjects.append(Chem.MolFromSmiles(smi))

    return molObjects
# Drop salt from SMILES string
def drop_salt(s):
    s = s.split(".")
    return [x for _, x in sorted(zip(map(len,s), s), reverse=True)][0]




# Get a martix containing the similarity of different fragments
def get_dist_matrix(fragments):

    id_dict = {}

    ms = []

    i = 0
    for smi, (m, _) in fragments.items():
        ms.append(m)
        id_dict[i] = smi
        i += 1


    distance_matrix = np.zeros([len(ms)] * 2)

    for i in range(len(ms)):
        for j in range(i+1,len(ms)):
            distance_matrix[i,j] = similarity(id_dict[i], id_dict[j], ms[i], ms[j])
            distance_matrix[j,i] = distance_matrix[i,j]

    return distance_matrix, id_dict




# Create pairs of fragments in a greedy way based on a similarity matrix
def find_pairs(distance_matrix):

    left = np.ones(distance_matrix.shape[0])
    pairs = []

    candidates = sorted(zip(distance_matrix.max(1),zip(range(distance_matrix.shape[0]),
                                                       distance_matrix.argmax(1))))
    use_next = []

    while len(candidates) > 0:
        v, (c1,c2) = candidates.pop()

        if left[c1] + left[c2] == 2:
            left[c1] = 0
            left[c2] = 0
            pairs.append([c1,c2])

        elif np.sum(left) == 1: # Just one sample left
            sampl = np.argmax(left)
            pairs.append([sampl])
            left[sampl] = 0


        elif left[c1] == 1:
            row = distance_matrix[c1,:] * left
            c2_new = row.argmax()
            v_new = row[c2_new]
            new =  (v_new, (c1, c2_new))
            bisect.insort(candidates, new)

    return pairs



# Create a new similarity matrix from a given set of pairs
# The new similarity is the maximal similarity of any fragment in the sets that are combined.
def build_matrix(pairs, old_matrix):

    new_mat = np.zeros([len(pairs)] * 2) - 0.1

    for i in range(len(pairs)):
        for j in range(i+1, len(pairs)):
            new_mat[i,j] = np.max((old_matrix[pairs[i]])[:,[pairs[j]]])
            new_mat[j,i] = new_mat[i,j]
    return new_mat


# Get a containing pairs of nested lists where the similarity between fragments in a list is higher than between
#   fragments which are not in the same list.
def get_hierarchy(fragments):

    distance_matrix,  id_dict = get_dist_matrix(fragments)
    working_mat = (distance_matrix + 0.001) * (1- np.eye(distance_matrix.shape[0]))


    pairings = []

    while working_mat.shape[0] > 1:
        pairings.append(find_pairs(working_mat))
        working_mat = build_matrix(pairings[-1], working_mat)

    return pairings, id_dict



# Build a binary tree from a list of fragments where the most similar fragments are neighbouring in the tree.
# This paths from the root in the tree to the fragments in the leafs is then used to build encode fragments.
def get_encodings(fragments):

    pairings, id_dict = get_hierarchy(fragments)

    assert id_dict

    t = build_tree_from_list(pairings, lookup=id_dict)
    encodings = dict(t.encode_leafs())
    decodings = dict([(v, fragments[k][0]) for k,v in encodings.items()])

    return encodings, decodings



# Encode a fragment.
def encode_molecule(m, encodings):
    fs = [Chem.MolToSmiles(f) for f in split_molecule(m)]
    encoded = "-".join([encodings[f] for f in fs])
    return encoded


# Decode a string representation into a fragment.
def decode_molecule(enc, decodings):
    fs = [Chem.Mol(decodings[x]) for x in enc.split("-")]
    return join_fragments(fs)


# Decode an array representation into a fragment.
def decode(x, translation):
    enc = ["".join([str(int(y)) for y in e[1:]]) for e in x if e[0] == 1]
    fs = [Chem.Mol(translation[e]) for e in enc]
    if not fs:
        return Chem.Mol()
    return join_fragments(fs)


# Encode a list of molecules into their corresponding encodings
def encode_list(mols, encodings):
  
    enc_size = None
    for v in encodings.values():
        enc_size = len(v)
        break
    assert enc_size


    def get_len(x):
        return (len(x) + 1) / enc_size

    encoded_mols = [encode_molecule(m, encodings) for m in mols]
    X_mat = np.zeros((len(encoded_mols), MAX_FRAGMENTS, enc_size + 1))


    for i in range(X_mat.shape[0]):
        es = encoded_mols[i].split("-")

        for j in range(X_mat.shape[1]):
            if j < len(es):
                e = np.asarray([int(c) for c in es[j]])
                if not len(e): continue
                
                X_mat[i,j,0] = 1
                X_mat[i,j,1:] = e

    return X_mat

import Levenshtein
from rdkit.Chem import rdFMCS
MAX_FRAGMENTS = 12



# Calculate similartity between two molecules (or fragments) based on their edit distance
def calculateDistance(smi1,smi2): 
    return 1 - ETA * Levenshtein.distance(smi1, smi2)


# Calculate the MCS Tanimoto similarity between two molecules
def calculateMCStanimoto(ref_mol, target_mol):

    numAtomsRefCpd = float(ref_mol.GetNumAtoms())
    numAtomsTargetCpd = float(target_mol.GetNumAtoms())

    if numAtomsRefCpd < numAtomsTargetCpd:
        leastNumAtms = int(numAtomsRefCpd)
    else:
        leastNumAtms = int(numAtomsTargetCpd)

    pair_of_molecules = [ref_mol, target_mol]
    numCommonAtoms = rdFMCS.FindMCS(pair_of_molecules, 
                                    atomCompare=rdFMCS.AtomCompare.CompareElements,
                                    bondCompare=rdFMCS.BondCompare.CompareOrderExact, matchValences=True).numAtoms
    mcsTanimoto = numCommonAtoms/((numAtomsTargetCpd+numAtomsRefCpd)-numCommonAtoms)

    return mcsTanimoto, leastNumAtms



# Calculate the similarity of two molecules (with SMILE representations smi1 and smi2) 
#  This is the maximum of the two functions above
def similarity(smi1, smi2, mol1, mol2):
    global s1,s2
    d1 = calculateDistance(smi1, smi2)
    d2 = calculateMCStanimoto(mol1, mol2)[0]
    
    return max(d1, d2)





# Save all decodings as a file (fragments are stored as SMILES)
def save_decodings(decodings):
    decodings_smi = dict([(x,Chem.MolToSmiles(m)) for x,m in decodings.items()])

    if not os.path.exists("History/"):
        os.makedirs("History")

    with open("History/decodings.txt","w+") as f:
        f.write(str(decodings_smi))

# Read encoding list from file
def read_decodings():
    with open("History/decodings.txt","r") as f:
        d = eval(f.read())
        return dict([(x,Chem.MolFromSmiles(m)) for x,m in d.items()])
from rdkit import Chem
import numpy as np

MOL_SPLIT_START = 70
MAX_ATOMS = 20
MAX_FREE = 3
MAX_FRAGMENTS = 12
ETA = 0.1

# Main module for handleing the interactions with molecules





# Atom numbers of noble gases (should not be used as dummy atoms)
NOBLE_GASES = set([2, 10, 18, 36, 54, 86])
ng_correction = set()


# Drop salt from SMILES string
def drop_salt(s):
    s = s.split(".")
    return [x for _, x in sorted(zip(map(len,s), s), reverse=True)][0]




# Check if it is ok to break a bond.
# It is ok to break a bond if:
#    1. It is a single bond
#    2. Either the start or the end atom is in a ring, but not both of them.
def okToBreak(bond):

    if bond.IsInRing():
        return False

    if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
        return False


    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()

    if not(begin_atom.IsInRing() or end_atom.IsInRing()):
        return False
    elif begin_atom.GetAtomicNum() >= MOL_SPLIT_START or \
            end_atom.GetAtomicNum() >= MOL_SPLIT_START:
        return False
    else:
        return True



# Divide a molecule into fragments
def split_molecule(mol):

    split_id = MOL_SPLIT_START

    res = []
    to_check = [mol]
    while len(to_check) > 0:
        ms = spf(to_check.pop(), split_id)
        if len(ms) == 1:
            res += ms
        else:
            to_check += ms
            split_id += 1

    return create_chain(res)


# Function for doing all the nitty gritty splitting work.
def spf(mol, split_id):

    bonds = mol.GetBonds()
    for i in range(len(bonds)):
        if okToBreak(bonds[i]):
            mol = Chem.FragmentOnBonds(mol, [i], addDummies=True, dummyLabels=[(0, 0)])
            # Dummy atoms are always added last
            n_at = mol.GetNumAtoms()
            mol.GetAtomWithIdx(n_at-1).SetAtomicNum(split_id)
            mol.GetAtomWithIdx(n_at-2).SetAtomicNum(split_id)
            return Chem.rdmolops.GetMolFrags(mol, asMols=True)

    # If the molecule could not been split, return original molecule
    return [mol]



# Build up a chain of fragments from a molecule.
# This is required so that a given list of fragments can be rebuilt into the same
#   molecule as was given when splitting the molecule
def create_chain(splits):
    splits_ids = np.asarray(
        [sorted([a.GetAtomicNum() for a in m.GetAtoms()
              if a.GetAtomicNum() >= MOL_SPLIT_START]) for m in splits])

    splits_ids = \
        [sorted([a.GetAtomicNum() for a in m.GetAtoms()
              if a.GetAtomicNum() >= MOL_SPLIT_START]) for m in splits]

    splits2 = []
    mv = np.max(splits_ids)
    look_for = [mv if isinstance(mv, np.int64) else mv[0]]
    join_order = []

    mols = []

    for i in range(len(splits_ids)):
        l = splits_ids[i]
        if l[0] == look_for[0] and len(l) == 1:
            mols.append(splits[i])
            splits2.append(splits_ids[i])
            splits_ids[i] = []


    while len(look_for) > 0:
        sid = look_for.pop()
        join_order.append(sid)
        next_mol = [i for i in range(len(splits_ids))
                      if sid in splits_ids[i]]

        if len(next_mol) == 0:
            break
        next_mol = next_mol[0]

        for n in splits_ids[next_mol]:
            if n != sid:
                look_for.append(n)
        mols.append(splits[next_mol])
        splits2.append(splits_ids[next_mol])
        splits_ids[next_mol] = []

    return [simplify_splits(mols[i], splits2[i], join_order) for i in range(len(mols))]



# Split and keep track of the order on how to rebuild the molecule
def simplify_splits(mol, splits, join_order):

    td = {}
    n = 0
    for i in splits:
        for j in join_order:
            if i == j:
                td[i] = MOL_SPLIT_START + n
                n += 1
                if n in NOBLE_GASES:
                    n += 1


    for a in mol.GetAtoms():
        k = a.GetAtomicNum()
        if k in td:
            a.SetAtomicNum(td[k])

    return mol


# Go through a molecule and find attachment points and define in which order they should be re-joined.
def get_join_list(mol):

    join = []
    rem = []
    bonds = []

    for a in mol.GetAtoms():
        an = a.GetAtomicNum()
        if an >= MOL_SPLIT_START:
            while len(join) <= (an - MOL_SPLIT_START):
                rem.append(None)
                bonds.append(None)
                join.append(None)

            b = a.GetBonds()[0]
            ja = b.GetBeginAtom() if b.GetBeginAtom().GetAtomicNum() < MOL_SPLIT_START else \
                 b.GetEndAtom()
            join[an - MOL_SPLIT_START] = ja.GetIdx()
            rem[an - MOL_SPLIT_START] = a.GetIdx()
            bonds[an - MOL_SPLIT_START] = b.GetBondType()
            a.SetAtomicNum(0)

    return [x for x in join if x is not None],\
           [x for x in bonds if x is not None],\
           [x for x in rem if x is not None]


# Join a list of fragments toghether into a molecule
#   Throws an exception if it is not possible to join all fragments.
def join_fragments(fragments):

    to_join = []
    bonds = []
    pairs = []
    del_atoms = []
    new_mol = fragments[0]

    j,b,r = get_join_list(fragments[0])
    to_join += j
    del_atoms += r
    bonds += b
    offset = fragments[0].GetNumAtoms()

    for f in fragments[1:]:

        j,b,r = get_join_list(f)
        p = to_join.pop()
        pb = bonds.pop()

        # Check bond types if b[:-1] == pb
        if b[:-1] != pb:
            assert("Can't connect bonds")



        pairs.append((p, j[-1] + offset,pb))

        for x in j[:-1]:
            to_join.append(x + offset)
        for x in r:
            del_atoms.append(x + offset)
        bonds += b[:-1]

        offset += f.GetNumAtoms()
        new_mol = Chem.CombineMols(new_mol, f)


    new_mol =  Chem.EditableMol(new_mol)

    for a1,a2,b in pairs:
        new_mol.AddBond(a1,a2, order=b)

    # Remove atom with greatest number first:
    for s in sorted(del_atoms, reverse=True):
        new_mol.RemoveAtom(s)
    return new_mol.GetMol()





# Decide the class of a fragment
#   Either R-group, Linker or Scaffold
def get_class(fragment):

    is_ring = False
    n = 0

    for a in fragment.GetAtoms():
        if a.IsInRing():
            is_ring = True

        if a.GetAtomicNum() >= MOL_SPLIT_START:
            n += 1

    smi = Chem.MolToSmiles(fragment)

    if n == 1:
        cl = "R-group"
    elif is_ring:
        cl = "Scaffold-" + str(n)
    else:
        cl = "Linker-" + str(n)

    return cl




# Enforce conditions on fragments
def should_use(fragment):

    n = 0
    m = 0
    for a in fragment.GetAtoms():
        m += 1
        if a.GetAtomicNum() >= MOL_SPLIT_START:
            n += 1
        if n > MAX_FREE or m > MAX_ATOMS:
            return False

    return True




# Split a list of molecules into fragments.
def get_fragments(mols):

    used_mols = np.zeros(len(mols)) != 0

    fragments = dict()

    # Get all non-ring single bonds (including to H) and store in list (listofsinglebonds)
    i = -1
    for mol in mols:
        i += 1
        try:
            fs = split_molecule(mol)
        except:
            continue

        if len(fs) <= MAX_FRAGMENTS and all(map(should_use, fs)):
            used_mols[i] = True
        else:
            continue

        for f in fs:
            cl = get_class(f)
            fragments[Chem.MolToSmiles(f)] = (f, cl)

    return fragments, used_mols
import torch
import torch.nn as nn

class DeepLSTM(nn.Module):
   import torch
import torch.nn as nn

class DeepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers,mole_len):
        super(DeepLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

        self.embedding = nn.Embedding(mole_len, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.fc_out(out)
        return out
if __name__ == "__main__":
    # fragment_file = "Data/molecules500.smi"
    # fragment_mols = read_file(fragment_file)
    # fragments, used_mols = get_fragments(fragment_mols)     
    # encodings, decodings = get_encodings(fragments)
    # print(encodings,decodings)
    # # Hyperparameters
    # input_size = 9
    # hidden_size = 256
    # output_size = 128
    # num_layers = 5



    # import torch
    # active_structure_list=[]
    # active_structure_list_new=[]
    # def binary_str_to_tensor(binary_str):
    #     int_list = [int(c) for c in binary_str]
    #     print(torch.tensor(int_list, dtype=torch.long))
    #     return torch.tensor(int_list, dtype=torch.long)
    # for v in encodings.values():
    #     active_structure_list.append(v)
    # print(active_structure_list)
    # print(len(active_structure_list))
    # # Convert a binary string to a tensor




    # for i in active_structure_list:
    #     active_structure_list_new.append(binary_str_to_tensor(i).tolist())
    # active_structure_list=torch.tensor(active_structure_list_new)

    # model = DeepLSTM(input_size, hidden_size, output_size, num_layers,mole_len=len(active_structure_list))

    # # input_tensor1 = binary_str_to_tensor(input_binary_str1)
    # # input_tensor2 = binary_str_to_tensor(input_binary_str2)
    # # input_tensor3 = binary_str_to_tensor(input_binary_str3)

    # # input_tensor=torch.tensor([input_tensor1.tolist(),input_tensor2.tolist(),input_tensor3.tolist()])
    # print(active_structure_list)
    # print(active_structure_list.shape)

    # print(model(active_structure_list))

    # torch.save(model(active_structure_list), 'file.pt')


    pt_file = torch.load("./file.pt")
    print(pt_file)