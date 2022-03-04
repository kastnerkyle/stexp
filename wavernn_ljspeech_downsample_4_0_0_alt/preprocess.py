"""
Preprocess dataset

usage: preproess.py [options]

options:
     --output-dir=<dir>      Directory where processed outputs are saved. [default: data_dir].
   -h, --help              Show help message.
"""
import os
from docopt import docopt
import numpy as np
import math, pickle, os
from audio import *
#from alt_audio import *
from hparams import hparams as hp
from utils import *
from tqdm import tqdm
from scipy.io import wavfile


from kkpthlib.datasets import EnglishSpeechCorpus


def get_wav_mel(speech, path):
    """Given path to .wav file, get the quantized wav and mel spectrogram as numpy vectors

    """
    #wav = load_wav(path)
    fs, wav = wavfile.read(path)
    wav = wav.astype(np.float32) / (2. ** 15)
    mel = speech._melspectrogram_preprocess(wav, fs)
    # existing code wants file to be (mel, time) with mel freqs starting from 0
    mel = mel.T
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.matshow(mel)
    plt.savefig("new_tmp.png")
    from IPython import embed; embed(); raise ValueError()
    """
    if hp.input_type == 'raw':
        return wav.astype(np.float32), mel
    elif hp.input_type == 'mulaw':
        quant = mulaw_quantize(wav, hp.mulaw_quantize_channels)
        return quant.astype(np.int), mel
    elif hp.input_type == 'bits':
        quant = quantize(wav)
        return quant.astype(np.int), mel
    else:
        raise ValueError("hp.input_type {} not recognized".format(hp.input_type))


def process_data(speech, output_path):
    """
    given wav directory and output directory, process wav files and save quantized wav and mel
    spectrogram to output directory
    """


    train_dataset_ids = []
    test_dataset_ids = []
    # get list of wav files
    #wav_files = os.listdir(wav_dir)
    wav_dir = speech.wav_folder
    train_wav_files = [wav_dir + k + ".wav" for k in speech.train_keep_keys]
    test_wav_files = [wav_dir + k + ".wav" for k in speech.valid_keep_keys]
    train_file_id = [k for k in speech.train_keep_keys]
    test_file_id = [k for k in speech.valid_keep_keys]

    # process testing_wavs
    test_path = os.path.join(output_path,'test')

    train_mel_path = os.path.join(output_path, "mel")
    train_wav_path = os.path.join(output_path, "wav")
    test_mel_path = os.path.join(test_path, "mel")
    test_wav_path = os.path.join(test_path, "wav")
    os.makedirs(train_mel_path, exist_ok=True)
    os.makedirs(train_wav_path, exist_ok=True)
    os.makedirs(test_mel_path, exist_ok=True)
    os.makedirs(test_wav_path, exist_ok=True)
    # check wav_file
    assert len(train_wav_files) != 0 or train_wav_files[0][-4:] == '.wav', "no wav files found!"
    assert len(test_wav_files) != 0 or test_wav_files[0][-4:] == '.wav', "no wav files found!"
    # create training and testing splits
    for i, wav_file in enumerate(tqdm(train_wav_files)):
        # get the file id
        #file_id = '{:d}'.format(i).zfill(5)
        file_id = train_file_id[i]
        wav, mel = get_wav_mel(speech, wav_file)
        # save
        np.save(os.path.join(train_mel_path,file_id+".npy"), mel)
        np.save(os.path.join(train_wav_path,file_id+".npy"), wav)
        # add to dataset_ids
        train_dataset_ids.append(file_id)

    # save dataset_ids
    with open(os.path.join(output_path,'train_dataset_ids.pkl'), 'wb') as f:
        pickle.dump(train_dataset_ids, f)

    for i, wav_file in enumerate(test_wav_files):
        file_id = test_file_id[i]
        wav, mel = get_wav_mel(speech, wav_file)
        # save test_wavs
        np.save(os.path.join(test_mel_path,file_id+".npy"),mel)
        np.save(os.path.join(test_wav_path,file_id+".npy"),wav)
        test_dataset_ids.append(file_id)

    with open(os.path.join(test_path,'test_dataset_ids.pkl'), 'wb') as f:
        pickle.dump(test_dataset_ids, f)

    print("\npreprocessing done, total processed wav files:{}.\nProcessed files are located in:{}".format(len(train_wav_files) + len(test_wav_files), os.path.abspath(output_path)))


if __name__=="__main__":
    args = docopt(__doc__)
    # no wavdir argument for now we hardcode it
    #wav_dir = args["<wav-dir>"]
    output_dir = args["--output-dir"]

    # dimensions of mel are (mel_filts, time)
    random_seed=2122
    data_random_state = np.random.RandomState(random_seed)
    folder_base = "/usr/local/data/kkastner/ljspeech_cleaned"
    fixed_minibatch_time_secs = 4
    fraction_train_split = .9
    speech = EnglishSpeechCorpus(metadata_csv=folder_base + "/metadata.csv",
                                 wav_folder=folder_base + "/wavs/",
                                 alignment_folder=folder_base + "/alignment_json/",
                                 fixed_minibatch_time_secs=fixed_minibatch_time_secs,
                                 train_split=fraction_train_split,
                                 random_state=data_random_state)

    # create paths
    output_path = os.path.join(output_dir,"")
    test_path = os.path.join(output_dir, "test")

    # create dirs
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # process data
    process_data(speech, output_path)



def test_get_wav_mel():
    wav, mel = get_wav_mel('sample.wav')
    print(wav.shape, mel.shape)
    print(wav)
