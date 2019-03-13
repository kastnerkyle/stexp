import math, pickle, os
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import *
import sys
import models.nocond as nc
import models.vqvae as vqvae
import models.wavernn1 as wr
import utils.env as env
import argparse
import platform
import re
import utils.logger as logger
import time
import subprocess
import librosa

import config

parser = argparse.ArgumentParser(description='Train or run some neural net')
parser.add_argument('--generate', '-g', action='store_true')
parser.add_argument('--float', action='store_true')
parser.add_argument('--half', action='store_true')
parser.add_argument('--load', '-l')
parser.add_argument('--scratch', action='store_true')
parser.add_argument('--model', '-m')
parser.add_argument('--force', action='store_true', help='skip the version check')
parser.add_argument('--count', '-c', type=int, default=3, help='size of the test set')
parser.add_argument('--partial', action='append', default=[], help='model to partially load')
args = parser.parse_args()

if args.float and args.half:
    sys.exit('--float and --half cannot be specified together')

if args.float:
    use_half = False
elif args.half:
    use_half = True
else:
    use_half = False

DEVICE = "cpu"

model_type = args.model or 'vqvae'

#model_name = f'{model_type}.43.upconv'
model_name = f'{model_type}.43.upconv'

if model_type == 'vqvae':
    model_fn = lambda dataset: vqvae.Model(rnn_dims=896, fc_dims=896, global_decoder_cond_dims=dataset.num_speakers(),
                  upsample_factors=(4, 4, 4), normalize_vq=True, noise_x=True, noise_y=True, DEVICE=DEVICE).to(DEVICE)
    dataset_type = 'multi'
elif model_type == 'wavernn':
    raise ValueError("NYI wavernn")
    model_fn = lambda dataset: wr.Model(rnn_dims=896, fc_dims=896, pad=2,
                  upsample_factors=(4, 4, 4), feat_dims=80, DEVICE=DEVICE).to(DEVICE)
    dataset_type = 'single'
elif model_type == 'nc':
    raise ValueError("NYI nc")
    model_fn = lambda dataset: nc.Model(rnn_dims=896, fc_dims=896, DEVICE=DEVICE).to(DEVICE)
    dataset_type = 'single'
else:
    sys.exit(f'Unknown model: {model_type}')

if dataset_type == 'multi':
    """
    data_path = config.multi_speaker_data_path
    data_path = "gt_data_dir"
    with open(f'{data_path}/index.pkl', 'rb') as f:
        index = pickle.load(f)
    test_index = [x[-1:] if i < 2 * args.count else [] for i, x in enumerate(index)]
    train_index = [x[:-1] if i < args.count else x for i, x in enumerate(index)]
    dataset = env.MultispeakerDataset(train_index, data_path)
    """
elif dataset_type == 'single':
    raise ValueError("NYI single")
    data_path = config.single_speaker_data_path
    with open(f'{data_path}/dataset_ids.pkl', 'rb') as f:
        index = pickle.load(f)
    test_index = index[-args.count:] + index[:args.count]
    train_index = index[:-args.count]
    dataset = env.AudiobookDataset(train_index, data_path)
else:
    raise RuntimeError('bad dataset type')

#print(f'dataset size: {len(dataset)}')
#model = model_fn(dataset)
data_path = "sample_data_dir"
model = vqvae.Model(rnn_dims=896, fc_dims=896, global_decoder_cond_dims=109, #dataset.num_speakers(),
                    upsample_factors=(4, 4, 4), normalize_vq=True, noise_x=True, noise_y=True, DEVICE=DEVICE).to(DEVICE)

if use_half:
    model = model.half()

for partial_path in args.partial:
    model.load_state_dict(torch.load(partial_path), strict=False, map_location=DEVICE)
model = model.to(DEVICE)

paths = env.Paths(model_name, data_path)

if args.scratch or args.load == None and not os.path.exists(paths.model_path()):
    # Start from scratch
    step = 0
else:
    if args.load:
        prev_model_name = re.sub(r'_[0-9]+$', '', re.sub(r'\.pyt$', '', os.path.basename(args.load)))
        prev_model_basename = prev_model_name.split('_')[0]
        model_basename = model_name.split('_')[0]
        if prev_model_basename != model_basename and not args.force:
            sys.exit(f'refusing to load {args.load} because its basename ({prev_model_basename}) is not {model_basename}')
        if args.generate:
            paths = env.Paths(prev_model_name, data_path)
        prev_path = args.load
    else:
        prev_path = paths.model_path()

    #step = env.restore(prev_path, model, DEVICE=DEVICE)
    def kk_restore(path, model, DEVICE="cuda"):
        partial_path = str(os.sep).join(path.split(os.sep)[:-1])
        all_saved_models = [m for m in os.listdir(partial_path) if "_" in m and "step" not in m]
        # get from most recent to least
        all_saved_models = sorted(all_saved_models, key=lambda x: int(x.split("_")[-1][:-len(".pyt")]))[::-1]
        path = partial_path + os.sep + all_saved_models[0]
        print("LOADING MODEL FROM {}".format(path))
        model.load_state_dict(torch.load(path, map_location=DEVICE))

        match = re.search(r'_([0-9]+)\.pyt', path)
        if match:
            return int(match.group(1))
        else:
            raise ValueError("Match fail in kk_restore... fixme")

    step = kk_restore(prev_path, model, DEVICE=DEVICE)

model = model.to(DEVICE)

#model.freeze_encoder()

optimiser = optim.Adam(model.parameters())

if args.generate:
    #model.do_generate(paths, step, data_path, test_index, use_half=use_half, verbose=True, deterministic=True)

    def kk_forward_generate(model, global_decoder_cond, samples, deterministic=False, use_half=False, verbose=False):
        if use_half:
            samples = samples.half()
        # samples: (L)
        #logger.log(f'samples: {samples.size()}')
        model.eval()
        with torch.no_grad() :
            continuous = model.encoder(samples)
            discrete, vq_pen, encoder_pen, entropy = model.vq(continuous.unsqueeze(2))
            logger.log(f'entropy: {entropy}')
            # cond: (1, L1, 64)
            #logger.log(f'cond: {cond.size()}')
            print("Beginning decoder sampling...")
            output = model.overtone.generate(discrete.squeeze(2), global_decoder_cond, use_half=use_half, verbose=verbose)
        model.train()
        return output

    def kk_do_generate(model, paths, step, data_path, deterministic=False, use_half=False, verbose=False):
        k = step // 1000
        num_speakers = 109
        npy_files = []
        for root, dirs, files in os.walk(data_path):
            for fi in files:
                if fi.endswith(".npy"):
                     npy_files.append(os.path.join(root, fi))
        data = [torch.FloatTensor(np.load(npy)).to(DEVICE) for npy in npy_files]
        n_points = len(data)
        """
        dataset = kkMultispeakerDataset(test_index, data_path)
        loader = DataLoader(dataset, shuffle=False)
        data = [x for x in loader]
        n_points = len(data)
        """
        #gt = [(x[0].float() + 0.5) / (2**15 - 0.5) for x in data]
        gt = [(x.float() + 0.5) / (2**15 - 0.5) for x in data]
        extended = [np.concatenate([np.zeros(model.pad_left_encoder(), dtype=np.float32), x, np.zeros(model.pad_right(), dtype=np.float32)]) for x in gt]
        # make 10 speaker labels at random
        def oh(num):
            t = np.zeros((1, num_speakers))
            t[0, num] = 1.
            return t[0]

        sample_rate = 22050
        speakers = [torch.FloatTensor(oh(s)) for s in list(range(n_points))]
        maxlen = max([len(x) for x in extended])
        aligned = [torch.cat([torch.FloatTensor(x), torch.zeros(maxlen-len(x))]) for x in extended]
        os.makedirs(paths.gen_path(), exist_ok=True)
        out = kk_forward_generate(model, torch.stack(speakers + list(reversed(speakers)), dim=0).to(model.DEVICE), torch.stack(aligned + aligned, dim=0).to(model.DEVICE), verbose=verbose, use_half=use_half)
        logger.log(f'out: {out.size()}')
        for i, x in enumerate(gt) :
            librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_target.wav', x.cpu().numpy(), sr=sample_rate)
            audio = out[i][:len(x)].cpu().numpy()
            librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_generated.wav', audio, sr=sample_rate)
            audio_tr = out[n_points+i][:len(x)].cpu().numpy()
            librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_transferred.wav', audio_tr, sr=sample_rate)

    kk_do_generate(model, paths, step, data_path, use_half=use_half, verbose=True, deterministic=True)

else:
    raise ValueError("No training! Pass -g flag")
