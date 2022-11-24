from glyles import convert


print("Read data")
glycans = []
with open("data/pred_class.tsv", "r") as data:
    for line in data.readlines()[1:]:
        glycans.append(line.split("\t")[0].strip())

print("Convert")
smiles = convert(glycan_list=glycans, cpu_count=14, verbose=None)

print("Saving")
with open("data/smiles.txt", "w") as output, open("data/glycans.tsv", "w") as glyput:
    print("ID\tIUPAC\tSMILES", file=glyput)
    for i, (iupac, smile) in enumerate(smiles):
        print(smile, file=output)
        print(f"Gly{i+1:05}\t{iupac}\t{smile}", file=glyput)
