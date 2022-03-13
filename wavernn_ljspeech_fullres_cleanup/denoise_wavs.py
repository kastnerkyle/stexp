"""Denoise wavs dataset

usage: denoise_wavs.py --wav-dir [--output-dir]

options:
  --wav-dir=<wav-dir>  Directory where processed outputs are saved.
  --output-dir=<output-dir>  Directory where processed outputs are saved. [default: denoised_dir].
  -h, --help  Show help message.
"""
import os
from docopt import docopt
from tqdm import tqdm
from scipy.io import wavfile
import noisereduce as nr
import shutil

if __name__ == "__main__":
    args = docopt(__doc__)
    # no wavdir argument for now we hardcode it
    wav_dir = args["--wav-dir"]
    output_dir = args["--output-dir"]

    noise_rate, noise_data = wavfile.read("noisy_example.wav")
    noise_part = noise_data[-18000:-5000]

    wav_files = [wav_dir + os.sep + k for k in os.listdir(wav_dir)]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i, wav_file in enumerate(tqdm(wav_files)):
        #load data 
        rate, data = wavfile.read(wav_file)
        # perform noise reduction 
        cleaned_data = nr.reduce_noise(y=data, y_noise=noise_part, sr=rate)
        fname = wav_file.split(os.sep)[-1]
        wavfile.write(output_dir + os.sep + fname, rate, cleaned_data)
