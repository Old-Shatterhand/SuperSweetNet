import os
import random

import pandas as pd
from glyles import converter


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


def split(data, mode):
    if mode == "random":
        splits = []
        for _ in range(len(data["IUPAC"])):
            x = random.random()
            if x < 0.7:
                splits.append("train")
            elif x < 0.9:
                splits.append("val")
            else:
                splits.append("test")
        return splits
    raise ValueError("Unknown splitting technique!")


def main(filepath, datapath, split_mode="random"):
    with open(filepath, "r") as datastream:
        data = {"IUPAC": [], "SMILES": [], "Species": []}
        lines = datastream.readlines()
        for i, line in enumerate(lines):
            print(f"\r{i}/{len(lines)}", end="")
            if i == 0:
                continue
            if i == 1000:
                break
            glycan, species = line.split("\t")[:2]
            if species == "[]":
                continue
            with suppress_stdout_stderr():
                iupac, smiles = converter.convert(glycan, returning=True)[0]
            if smiles == "":
                continue
            species = species.replace("'", "").replace("[", "").replace("]", ", ")[:-2]
            data["IUPAC"].append(iupac)
            data["SMILES"].append(smiles)
            data["Species"].append(species)

        print("\tFinished")
        splits = split(data, split_mode)
        df = pd.DataFrame(list(zip(*(data["IUPAC"], data["SMILES"], splits))), columns=["IUPAC", "SMILES", "Split"])
        species = pd.get_dummies(data["Species"])
        df = df.join(species)
        df.to_csv(datapath, index=False, sep="\t")


if __name__ == '__main__':
    main("./data/glycowork_v05.tsv", "./data/filtered_data.tsv", "random")
