import json
import sys
import os

outfile = "obama_amazon_metadata.csv"
if len(sys.argv) != 2:
    print("Usage: script.py <obama_dataset_base_path>")
    sys.exit()

datadir = sys.argv[1]

jsondir = datadir + os.sep + "gentle_json"
jsonfiles = sorted([jsondir + os.sep + f for f in os.listdir(jsondir) if f[-len(".json"):] == ".json"])

metadata_lines = []
for n, jsonf in enumerate(jsonfiles):
    print("Processed file {} of {}".format(n + 1, len(jsonfiles)))
    with open(jsonf, "r") as f:
        this_json = json.load(f)
    text = this_json["transcript"].strip()
    fname = jsonf.split(os.sep)[-1][:-len(".json")]
    metadata_lines.append("{}|{}\n".format(fname, text))

with open(outfile, "w") as f:
    f.writelines(metadata_lines)

print("Wrote out {}".format(outfile))
