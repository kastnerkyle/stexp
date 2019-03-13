from __future__ import print_function
import json
import os
from scipy.io import wavfile
import subprocess
import argparse
import re

def pwrap(args, shell=False):
    p = subprocess.Popen(args, shell=shell, stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    return p

# Print output
# http://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
def execute(cmd, shell=False):
    popen = pwrap(cmd, shell=shell)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line

    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def pe(cmd, shell=True, verbose=True):
    """
    Print and execute command on system
    """
    all_lines = []
    for line in execute(cmd, shell=shell):
        if verbose:
            print(line, end="")
        all_lines.append(line.strip())
    return all_lines

all_wav_path = "raw_wav"
all_txt_path = "raw_txt"

align_wav_path = "aligned_wav"
align_json_path = "aligned_json"
if not os.path.exists(align_wav_path):
    os.makedirs(align_wav_path)

if not os.path.exists(align_json_path):
    os.makedirs(align_json_path)

raw_wav_files = os.listdir(all_wav_path)
raw_txt_files = os.listdir(all_txt_path)

wav_basenames = [rwf[:-len(".wav")] for rwf in raw_wav_files]
txt_basenames = [rjf[:-len(".txt")] for rjf in raw_txt_files]
intersected_basenames = list(set(wav_basenames) | set(txt_basenames))

if len(intersected_basenames) != len(txt_basenames) or len(intersected_basenames) != len(wav_basenames):
    print("Number of matched json:wav pairs is {}, but doesn't match number of raw files! Double check name alignment between raw_wav and raw_json".format(len(intersected_basenames)))

# All of this is for running gentle from the current directory
# now there are two directories
here = os.getcwd()
there = here + os.sep + "raw_voice_cleanup" + os.sep + "alignment"
if not os.path.exists(there + os.sep + "txts"):
    cmd = "ln -s {}/raw_txt {}/txts".format(here, there)
    res = pe(cmd, verbose=False)
if not os.path.exists(there + os.sep + "wavs"):
    cmd = "ln -s {}/raw_wav {}/wavs".format(here, there)
    res = pe(cmd, verbose=False)

# This part sets up and runs gentle over the data
# in wavdir and txtdir respectively
# the output from gentle is *slightly* non-deterministic
# not sure how to do anything about this honestly
# for now, just running once but in theory you could iterate
# several times until it "settles" on a minimum error
wavdir = here + os.sep + all_wav_path
txtdir = here + os.sep + all_txt_path
outjsondir = here + os.sep + align_json_path
wavfile_list = sorted([wavdir + os.sep + wvf for wvf in os.listdir(wavdir)])
txtfile_list = sorted([txtdir + os.sep + txf for txf in os.listdir(txtdir)])
if not os.path.exists(outjsondir):
    os.mkdir(outjsondir)

# try to match every txt file and every wav file by name
# TODO: clean up this logic
wv_bases = sorted([str(os.sep).join(wv.split(os.sep)[:-1]) for wv in wavfile_list])
tx_bases = sorted([str(os.sep).join(tx.split(os.sep)[:-1]) for tx in txtfile_list])
wv_names = sorted([wv.split(os.sep)[-1] for wv in wavfile_list])
tx_names = sorted([tx.split(os.sep)[-1] for tx in txtfile_list])
wv_i = 0
tx_i = 0
wv_match = []
tx_match = []
while True:
    if tx_i >= len(tx_names) or wv_i >= len(wv_names):
        break
    if "." in tx_names[tx_i]:
        tx_part = ".".join(tx_names[tx_i].split(".")[:1])
    else:
        # support txt files with no ext
        tx_part = tx_names[tx_i]
    wv_part = ".".join(wv_names[wv_i].split(".")[:1])
    if wv_part == tx_part:
        wv_match.append(wv_bases[wv_i] + os.sep + wv_names[wv_i])
        tx_match.append(tx_bases[tx_i] + os.sep + tx_names[tx_i])
        wv_i += 1
        tx_i += 1
    else:
        print("WAV AND TXT NAMES DIDN'T MATCH AT STEP, ADD LOGIC")
        from IPython import embed; embed(); raise ValueError()

# run gentle alignment over the paired files
# add a timeout because sometimes Kaldi breaks the pipe and things hang forever...
gentle_timeout = 30
assert len(wv_match) == len(tx_match)
for n, (wvf, txf) in enumerate(zip(wv_match, tx_match)):
    base = txf.split(os.sep)[-1][:-len(".txt")] # remove .txt, and preceding path
    ojf = outjsondir + os.sep + base + ".json"
    if os.path.exists(ojf):
        print("gentle json file already found at {}, skipping".format(ojf))
    else:
        print("Aligning {}/{}, {}:{}".format(n + 1, len(wv_match), wvf, txf))
        cmd = "timeout {} python {}/gentle/align.py --disfluency {} {}".format(gentle_timeout, there, wvf, txf)
        try:
            res = pe(cmd, verbose=False)
            rj = json.loads("".join(res))
            err_count_new = sum([w["case"] == "not-found-in-audio" for w in rj["words"]])
            print("Writing out {}".format(ojf))
            with open(ojf, 'w') as f:
                 json.dump(rj, f, sort_keys=False, indent=4,
                           ensure_ascii=True)
        except:
            print("WARNING: Unknown exception in '{}', continuing...".format(cmd))
