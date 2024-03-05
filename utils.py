from rdkit import Chem


def get_valid_mols(path_sub):
    total = 0
    valid_mols = []
    with open(path_sub, "r") as f:
        for line in f:
            smile = line[:-1]
            m = Chem.MolFromSmiles(smile, sanitize=True)
            if m is not None:
                valid_mols.append(m)
            total += 1

    return valid_mols, total


def evaluate(path_sub, path_train):
    valid_mols, total = get_valid_mols(path_sub)
    valid_smiles = []
    for m in valid_mols:
        s = Chem.MolToSmiles(m)
        valid_smiles.append(s)

    smiles_unique = set(valid_smiles)

    with open(path_train, "r") as f:
        smiles_train = {s for s in f.read().split() if s}

    smiles_novel = smiles_unique - smiles_train

    validity = len(valid_smiles) / total
    uniqueness = len(smiles_unique) / total
    novelty = len(smiles_novel) / total

    return validity, uniqueness, novelty
