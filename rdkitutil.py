from rdkit import Chem


def disable_rdkit_log():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')


def canonicalize_smiles(smi):
    # assumes the input smiles contains no blank space chars.
    smi = smi.replace(' ', '')
    disable_rdkit_log()
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ''


def match_smiles(sm1, sm2, canonicalize=True):
    """
    check whether two smiles are identical or not
    """
    if canonicalize:
        sm1 = canonicalize_smiles(sm1)
        sm2 = canonicalize_smiles(sm2)

    if '' in [sm1, sm2]:
        return False

    sm1_set = set(sm1.split('.'))
    sm2_set = set(sm2.split('.'))

    sm1 = '.'.join(sorted(list(sm1_set)))
    sm2 = '.'.join(sorted(list(sm2_set)))

    return sm1 == sm2