"""Create downsampled spectrograms from a (possibly sampled) wav file

usage: reconstruct_npy.py <wav-file>

options:
    -h, --help                  Show this help message and exit
"""
from kkpthlib.datasets.speech.audio_processing.audio_tools import herz_to_mel, mel_to_herz
from kkpthlib.datasets.speech.audio_processing.audio_tools import stft, istft
from kkpthlib.utils import split
from kkpthlib.utils import split_np
from kkpthlib.utils import interleave_np
from kkpthlib.utils import interleave
from docopt import docopt
from scipy.io import wavfile
from scipy import signal
from shutil import copyfile
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def melspectrogram_preprocess(data, sample_rate):
    # takes in a raw sequence scaled between -1 and 1 (such as loaded from a wav file)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    x = data
    sr = sample_rate

    # hardcode these values...
    n_mels = 256

    mel_freq_min = 125
    mel_freq_max = 7600

    stft_size = 6 * 256
    stft_step = 256

    n_fft = stft_size
    n_step = stft_step
    fmin = mel_freq_min
    fmax = mel_freq_max

    # preemphasis filter
    preemphasis_coef = 0.97
    ref_level_db = 20
    min_level_db = -90

    # preemphasis filter
    coef = preemphasis_coef
    b = np.array([1.0, -coef], x.dtype)
    a = np.array([1.0], x.dtype)
    preemphasis_filtered = signal.lfilter(b, a, x)

    # mel weights
    # nfft - 1 because onesided=False cuts off last bin
    weights = np.zeros((n_mels, n_fft - 1), dtype="float32")

    fftfreqs = np.linspace(0, float(sr) / 2., n_fft - 1, endpoint=True)

    min_mel = herz_to_mel(fmin)
    max_mel = herz_to_mel(fmax)
    mels = np.linspace(min_mel, max_mel, n_mels + 2)
    mel_f = mel_to_herz(mels)[:, 0]

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / float(fdiff[i])
        upper = ramps[i + 2] / float(fdiff[i + 1])

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0., np.minimum(lower, upper))
    # slaney style norm
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    mel_weights = weights

    # do stft
    ref_level_db = ref_level_db
    min_level_db = min_level_db
    def _amp_to_db(a):
        min_level = np.exp(min_level_db / 20. * np.log(10))
        return 20 * np.log10(np.maximum(min_level, a))

    # ONE SIDED MUST BE FALSE!!!!!!!!
    abs_stft = np.abs(stft(preemphasis_filtered, fftsize=n_fft, step=n_step, real=True, compute_onesided=False))
    melspec_ref = _amp_to_db(np.dot(mel_weights, abs_stft.T)) - ref_level_db
    melspec_clip = np.clip((melspec_ref - min_level_db) / -min_level_db, 0, 1)
    return melspec_clip.T

if __name__=="__main__":
    args = docopt(__doc__)
    wav_file = args["<wav-file>"]

    fs, d = wavfile.read(wav_file)
    # put it between -1 and 1
    d = d.astype("float32") / (2 ** 15)
    mel = melspectrogram_preprocess(d, fs)
    max_frame_count = mel.shape[0]
    # pad to even multiple of 2, 4, 8
    divisors = [2, 4, 8]
    for di in divisors:
        # nearest divisble number above, works because largest divisor divides by smaller
        # we need something that has a length in time (frames) divisible by 2 4 and 8 due to the nature of melnet
        # same for frequency but frequency is a power of 2 so no need to check it
        q = int(max_frame_count / di)
        if float(max_frame_count / di) == int(max_frame_count / di):
            max_frame_count = di * q
        else:
            max_frame_count = di * (q + 1)
    new_mel = np.zeros((max_frame_count, mel.shape[1])).astype(mel.dtype)
    new_mel[:len(mel)] = mel
    mel = new_mel

    input_axis_split_list = [2, 1, 2, 1, 2]

    all_x_splits = []
    x_t = mel[None, ..., None]
    all_x_splits.append((x_t, x_t))
    for aa in input_axis_split_list:
        all_x_splits.append(split_np(x_t, axis=aa))
        x_t = all_x_splits[-1][0]
    # out, split 1 time, split 2 times, etc... down to the smallest split
    base_folder = "deconstructed/"
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)
    copyfile(wav_file, base_folder + wav_file.split("/")[-1])
    for _n in range(len(all_x_splits)):
        if _n == (len(all_x_splits) - 1):
            fname_base = "output_unnormalized_samples"
        else:
            fname_base = "tier{}_0_unnormalized_samples".format(_n)
        this_mel_split = all_x_splits[::-1][_n][0]
        np.save(base_folder + fname_base + ".npy", this_mel_split)
        plt.imshow(this_mel_split[0, ..., 0])
        plt.title(wav_file.split("/")[-1])
        plt.savefig(base_folder + fname_base + ".png")
        plt.close()
