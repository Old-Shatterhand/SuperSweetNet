import ast
import os
import random
import numpy as np

import pandas as pd
from glyles import converter


IUPAC, SMILES = 0, 1


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def get_dummies(data):
    items = dict((x, i) for i, x in enumerate(sorted(set([x for sublist in data for x in sublist]))))
    output = np.zeros((len(data), len(items)), dtype=int)
    for i, entry in enumerate(data):
        for x in entry:
            output[i, items[x]] = 1
    return pd.DataFrame(output, columns=items.keys())


def split(data, mode):
    if mode == "random":
        splits = []
        for _ in range(len(data)):
            x = random.random()
            if x < 0.7:
                splits.append("train")
            elif x < 0.9:
                splits.append("val")
            else:
                splits.append("test")
        return splits
    raise ValueError("Unknown splitting technique!")


def convert(iupac, sd):
    if iupac in sd:
        return sd[iupac]
    if iupac[0] == "[" \
            or "(z" in iupac \
            or " z" in iupac \
            or "-z" in iupac \
            or '-ulosaric' in iupac \
            or '-ulosonic' in iupac \
            or '-uronic' in iupac \
            or '-aric' in iupac \
            or '0dHex' in iupac \
            or 'Anhydro' in iupac \
            or 'en' in iupac \
            or 'Coum' in iupac \
            or 'Ins' in iupac:
        return None
    smiles = converter.convert(iupac, returning=True, silent=True)[0][1]
    if smiles == "" or any(x in smiles for x in ["[Se", "[Te", "[Po", "[At", "[Ga", "[Ge", "[In", "[Sn", "[Sn", "[Sb", "[Tl", "[Pb", "[Bi"]):
        return None
    return smiles


def clean(level):
    level = ast.literal_eval(level)
    if len(level) == 0:
        return None
    return [x for x in level]


def main(filepath, datapath, split_mode="random", class_level="Species", smiles_help=None):
    datapath = datapath.replace("?", class_level.lower() + "_2")
    """class_level from 'Species', 'Genus', 'Order', 'Class', 'Phylum', 'Kingdom', 'Domain'"""

    if smiles_help is not None:
        data = pd.read_csv(smiles_help, sep='\t')
        sd = dict(zip(data["glycan"], data["SMILES"]))
        del data
    else:
        sd = {}

    data = pd.read_csv(filepath, sep='\t')
    data = data[['glycan', class_level]]
    data[class_level] = data[class_level].apply(clean)
    data.dropna(subset=[class_level], axis='rows', inplace=True)
    data["SMILES"] = data["glycan"].apply(lambda x: convert(x, sd))
    data["split"] = split(data, split_mode)
    labels = get_dummies(data[class_level])
    data = pd.concat([data.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
    data.dropna(subset=["SMILES"], axis='rows', inplace=True)
    data[:50].to_csv(datapath, index=False, sep="\t")
    print(data.shape)


if __name__ == '__main__':
    main("./data/glycowork_v05.tsv", "./data/pred_?.tsv", "random", "Domain", "./data/pred_domain.tsv")
