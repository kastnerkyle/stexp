"""Sample from WaveRNN Model.

usage: reconstruct_npy.py [options] <wav-file>

options:
    --checkpoint=<path>         Restore model from checkpoint path
    -h, --help                  Show this help message and exit
"""

import torch
import random
import numpy as np
import os

from kkpthlib.datasets import EnglishSpeechCorpus
from scipy.io import wavfile
from audio import quantize

#default_seed = 2112
default_seed = 40000
#default_seed = 4142 
print("Setting all possible default seeds based on {}".format(default_seed))
# try to get deterministic runs
def seed_everything(seed=1234):
    random.seed(seed)
    tseed = random.randint(1, 1E6)
    tcseed = random.randint(1, 1E6)
    npseed = random.randint(1, 1E6)
    ospyseed = random.randint(1, 1E6)
    torch.manual_seed(tseed)
    torch.cuda.manual_seed_all(tcseed)
    np.random.seed(npseed)
    os.environ['PYTHONHASHSEED'] = str(ospyseed)
    #torch.backends.cudnn.deterministic = True

seed_everything(default_seed)

from docopt import docopt

import os
from os.path import dirname, join, expanduser
from tqdm import tqdm
import sys

import matplotlib.pyplot as plt
import librosa

from model import build_model

from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from model import build_model
from distributions import *
from loss_function import nll_loss
from dataset import raw_collate, discrete_collate, AudiobookDataset
from hparams import hparams as hp
from lrschedule import noam_learning_rate_decay, step_learning_rate_decay

from scipy.io import wavfile
def soundsc(X, gain_scale=.8, copy=True):
    # copied from kkpthlib
    """
    Approximate implementation of soundsc from MATLAB without the audio playing.

    Parameters
    ----------
    X : ndarray
        Signal to be rescaled

    gain_scale : float
        Gain multipler, default .9 (90% of maximum representation)

    copy : bool, optional (default=True)
        Whether to make a copy of input signal or operate in place.

    Returns
    -------
    X_sc : ndarray
        (-32767, 32767) scaled version of X as int16, suitable for writing
        with scipy.io.wavfile
    """
    X = np.array(X.astype("float64"), copy=copy)
    X = (X - X.min()) / (X.max() - X.min())
    X = 2 * X - 1
    X = gain_scale * X
    X = X * 2 ** 15
    return X.astype('int16')

global_step = 0
global_epoch = 0
global_test_step = 0
use_cuda = torch.cuda.is_available()

def save_checkpoint(device, model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(step))
    optimizer_state = optimizer.state_dict()
    global global_test_step
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "global_test_step": global_test_step,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path, DEVICE="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    """
    if use_cuda:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    """
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer, DEVICE="cuda"):
    global global_step
    global global_epoch
    global global_test_step

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path, DEVICE=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]
    global_test_step = checkpoint.get("global_test_step", 0)

    return model


def test_save_checkpoint():
    checkpoint_path = "checkpoints/"
    device = torch.device("cuda" if use_cuda else "cpu")
    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    global global_step, global_epoch, global_test_step
    save_checkpoint(device, model, optimizer, global_step, checkpoint_path, global_epoch)

    model = load_checkpoint(checkpoint_path+"checkpoint_step000000000.pth", model, optimizer, False)


def evaluate_model(model, data_loader, checkpoint_dir, limit_eval_to=5):
    """evaluate model and save generated wav and plot

    """
    test_path = data_loader.dataset.test_path
    test_files = os.listdir(test_path)
    counter = 0
    output_dir = os.path.join(checkpoint_dir,'eval')
    for f in test_files:
        if f[-7:] == "mel.npy":
            mel = np.load(os.path.join(test_path,f))
            wav = model.generate(mel)
            # save wav
            wav_path = os.path.join(output_dir,"checkpoint_step{:09d}_wav_{}.wav".format(global_step,counter))
            librosa.output.write_wav(wav_path, wav, sr=hp.sample_rate)
            # save wav plot
            fig_path = os.path.join(output_dir,"checkpoint_step{:09d}_wav_{}.png".format(global_step,counter))
            fig = plt.plot(wav.reshape(-1))
            plt.savefig(fig_path)
            # clear fig to drawing to the same plot
            plt.clf()
            counter += 1
        # stop evaluation early via limit_eval_to
        if counter >= limit_eval_to:
            break


def train_loop(device, model, data_loader, optimizer, checkpoint_dir):
    """Main training loop.

    """
    # create loss and put on device
    if hp.input_type == 'raw':
        if hp.distribution == 'beta':
            criterion = beta_mle_loss
        elif hp.distribution == 'gaussian':
            criterion = gaussian_loss
    elif hp.input_type == 'mixture':
        criterion = discretized_mix_logistic_loss
    elif hp.input_type in ["bits", "mulaw"]:
        criterion = nll_loss
    else:
        raise ValueError("input_type:{} not supported".format(hp.input_type))

    

    global global_step, global_epoch, global_test_step
    while global_epoch < hp.nepochs:
        running_loss = 0
        for i, (x, m, y) in enumerate(tqdm(data_loader)):
            x, m, y = x.to(device), m.to(device), y.to(device)
            y_hat = model(x, m)
            y = y.unsqueeze(-1)
            loss = criterion(y_hat, y)
            # calculate learning rate and update learning rate
            if hp.fix_learning_rate:
                current_lr = hp.fix_learning_rate
            elif hp.lr_schedule_type == 'step':
                current_lr = step_learning_rate_decay(hp.initial_learning_rate, global_step, hp.step_gamma, hp.lr_step_interval)
            else:
                current_lr = noam_learning_rate_decay(hp.initial_learning_rate, global_step, hp.noam_warm_up_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            optimizer.zero_grad()
            loss.backward()
            # clip gradient norm
            nn.utils.clip_grad_norm_(model.parameters(), hp.grad_norm)
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (i+1)
            # saving checkpoint if needed
            if global_step != 0 and global_step % hp.save_every_step == 0:
                save_checkpoint(device, model, optimizer, global_step, checkpoint_dir, global_epoch)
            # evaluate model if needed
            if global_step != 0 and global_test_step !=True and global_step % hp.evaluate_every_step == 0:
                print("step {}, evaluating model: generating wav from mel...".format(global_step))
                evaluate_model(model, data_loader, checkpoint_dir)
                print("evaluation finished, resuming training...")

            # reset global_test_step status after evaluation
            if global_test_step is True:
                global_test_step = False
            global_step += 1
        
        print("epoch:{}, running loss:{}, average loss:{}, current lr:{}".format(global_epoch, running_loss, avg_loss, current_lr))
        global_epoch += 1


def kk_evaluate_model(model, data_loader, limit_eval_to=5):
    """evaluate model and save generated wav and plot

    """
    test_path = data_loader.dataset.test_path
    test_files = os.listdir(test_path)
    counter = 0
    output_dir = 'eval'
    os.makedirs(output_dir)
    for f in test_files:
        if f[-7:] == "mel.npy":
            mel = np.load(os.path.join(test_path,f))
            print("mel")
            from IPython import embed; embed(); raise ValueError()
            wav = model.generate(mel, DEVICE="cpu")
            # save wav
            wav_path = os.path.join(output_dir,"eval_checkpoint_step{:09d}_wav_{}.wav".format(global_step,counter))
            librosa.output.write_wav(wav_path, wav, sr=hp.sample_rate)
            # save wav plot
            fig_path = os.path.join(output_dir,"eval_checkpoint_step{:09d}_wav_{}.png".format(global_step,counter))
            fig = plt.plot(wav.reshape(-1))
            plt.savefig(fig_path)
            # clear fig to drawing to the same plot
            plt.clf()
            counter += 1
        # stop evaluation early via limit_eval_to
        if counter >= limit_eval_to:
            break


if __name__=="__main__":
    args = docopt(__doc__)
    checkpoint_path = args["--checkpoint"]

    wav_file = args["<wav-file>"]
    use_device= 'cuda' if torch.cuda.is_available() else 'cpu'
    #use_device="cpu"
    print("using device:{}".format(use_device))
    model = build_model().to(use_device)

    optimizer = optim.Adam(model.parameters(),
                           lr=hp.initial_learning_rate, betas=(
        hp.adam_beta1, hp.adam_beta2),
        eps=hp.adam_eps, weight_decay=hp.weight_decay,
        amsgrad=hp.amsgrad)

    if hp.fix_learning_rate:
        print("using fixed learning rate of :{}".format(hp.fix_learning_rate))
    elif hp.lr_schedule_type == 'step':
        print("using exponential learning rate decay")
    elif hp.lr_schedule_type == 'noam':
        print("using noam learning rate decay")

    # load checkpoint
    if checkpoint_path is None:
        print("no checkpoint specified as --checkpoint argument, exiting...")
        sys.exit()

    # reset optimizer
    model = load_checkpoint(checkpoint_path, model, optimizer, False, DEVICE=use_device)
    model = model.to(use_device)

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
                                 random_state=data_random_state,
                                 force_mini_sample=True,
                                 bypass_checks=True,
                                 build_skiplist=False)

    fs, d = wavfile.read(wav_file)
    # between -1 and 1 now
    d = d.astype("float32") / (2 ** 15)
    mels = speech._melspectrogram_preprocess(d, fs)
    m_in = mels.T
    output_dir = 'eval'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    np.save(output_dir + os.sep + "mel.npy", m_in)

    pth_mels = torch.Tensor(m_in).to(use_device)
    wav = model.generate(pth_mels, DEVICE=use_device)

    wav_path = os.path.join(output_dir,"eval_checkpoint_step{:09d}_wav_{}_full.wav".format(global_step,0))
    scaled_wav = soundsc(wav - np.mean(wav))
    wavfile.write(wav_path, hp.sample_rate, scaled_wav)
