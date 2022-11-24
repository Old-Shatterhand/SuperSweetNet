import os.path

from rdkit import Chem
from rdkit.Chem import AllChem


with open("data/glycans.tsv") as data:
    for line in data.readlines()[1:]:
        idx, iupac, smiles = line.split("\t")
        filename = f"data/pdb/{idx}.pdb"
        if os.path.exists(filename):
            continue
        print(idx)
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, maxAttempts=5_000)
                AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
                Chem.MolToPDBFile(mol, filename)
        except Exception as e:
            pass
