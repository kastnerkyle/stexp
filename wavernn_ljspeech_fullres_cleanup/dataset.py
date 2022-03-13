import numpy as np

import os

import torch
from torch.utils.data import DataLoader, Dataset
from hparams import hparams as hp
from utils import mulaw_quantize, inv_mulaw_quantize
import pickle

from functools import reduce
from operator import mul

def split_np(x, axis):
    # copied from kkpthlib/utils.py
    assert x.shape[axis] % 2 == 0
    if axis == 0:
        return x[0::2], x[1::2]
    elif axis == 1:
        return x[:, 0::2], x[:, 1::2]
    elif axis == 2:
        return x[:, :, 0::2], x[:, :, 1::2]
    elif axis == 3:
        return x[:, :, :, 0::2], x[:, :, :, 1::2]
    else:
        raise ValueError("Only currently support axis 0 1 2 or 3")

def interleave_np(x_1, x_2, axis):
    # copied from kkpthlib/utils.py
    if axis == 1:
        c = np.empty((x_1.shape[0], 2 * x_1.shape[1], x_1.shape[2], x_1.shape[3])).astype(x_1.dtype)
        c[:, ::2] = x_1
        c[:, 1::2] = x_2
        return c
    if axis == 2:
        c = np.empty((x_1.shape[0], x_1.shape[1], 2 * x_1.shape[2], x_1.shape[3])).astype(x_1.dtype)
        c[:, :, ::2, :] = x_1
        c[:, :, 1::2, :] = x_2
        return c
    if axis == 3:
        c = np.empty((x_1.shape[0], x_1.shape[1], x_1.shape[2], 2 * x_1.shape[3])).astype(x_1.dtype)
        c[:, :, :, ::2] = x_1
        c[:, :, :, 1::2] = x_2
        return c

def patchify(img, patch_shape):
    img = np.ascontiguousarray(img)  # won't make a copy if not needed
    X, Y = img.shape
    x, y = patch_shape
    shape = ((X-x+1), (Y-y+1), x, y) # number of patches, patch_shape
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
    #    i,j,k,l are incremented by one
    strides = img.itemsize*np.array([Y, 1, Y, 1])
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)


class AudiobookDataset(Dataset):
    def __init__(self, data_path, split_n_mels=None, axis_splits=None, axis_splits_offset=None):
        # axis splits offset goes from first to last
        # eg 4_0, 3_0, 2_0, 1_0, 0_0
        self.axis_splits = axis_splits
        self.axis_splits_offset = axis_splits_offset
        self.split_n_mels = split_n_mels
        self._num_to_try_cache = None
        if split_n_mels is None:
            raise ValueError("Need to pass split_n_mels!")
        self.path = os.path.join(data_path, "")
        with open(os.path.join(self.path,'train_dataset_ids.pkl'), 'rb') as f:
            self.train_metadata = pickle.load(f)

        self.train_metadata_index = {}
        for i, el in enumerate(sorted(self.train_metadata)):
            self.train_metadata_index[i] = el
        self.train_mel_path = os.path.join(data_path, "mel")
        self.train_wav_path = os.path.join(data_path, "wav")

        self.test_path = os.path.join(data_path, "test")
        self.test_mel_path = os.path.join(data_path, "test", "mel")
        self.test_wav_path = os.path.join(data_path, "test", "wav")

    def __getitem__(self, index):
        file = self.train_metadata_index[index]
        m = np.load(os.path.join(self.train_mel_path,'{}.npy'.format(file)))
        x = np.load(os.path.join(self.train_wav_path,'{}.npy'.format(file)))
        m_in = m
        x_t = x
        if self.axis_splits is not None and self.axis_splits_offset is not None:
            m_in, x_t = self.split(m_in, x_t)
            # need to generalize this to 0-2-4?
            # make it an hparam and call it a day
            rs = hp.time_resample_factor
            if rs != 1:
                t = [np.interp(np.arange(rs * m_in.shape[1]) / rs, np.arange(m_in.shape[1]), m_in[i, :]) for i in range(m_in.shape[0])]
                m_in = np.vstack(t)
        return m_in, x_t

    def split(self, mel, wav=None):
        m = mel
        x = wav

        m_in = m
        x_t = x
        if self.axis_splits is not None and self.axis_splits_offset is not None:
            axis_split_list = [int(str(self.axis_splits)[i]) for i in range(len(str(self.axis_splits)))]

            all_m_splits = []
            # get it to batch time freq 1 format
            m_t = m.T[None, ..., None]
            x_t = x

            divisors = [2, 4, 8]
            max_frame_count = m_t.shape[1]
            for di in divisors:
                # nearest divisible number above, works because largest divisor divides by smaller
                # we need something that has a length in time (frames) divisible by 2 4 and 8 due to the nature of melnet
                # same for frequency but frequency is a power of 2 so no need to check it
                q = int(max_frame_count / di)
                if float(max_frame_count / di) == int(max_frame_count / di):
                    max_frame_count = di * q
                else:
                    max_frame_count = di * (q + 1)
            assert max_frame_count == int(max_frame_count)
            sz = m_t.shape[1]
            diff = max_frame_count - sz
            if diff != 0:
                m_t = np.concatenate((m_t, 0. * m_t[:, :diff]), axis=1)
                if wav is not None:
                    x_t = np.concatenate((x, int(x.mean()) + 0 * x[:m_t.shape[1] * diff]))

            axis_splits = self.axis_splits
            axis1_m = [2 for a in str(axis_splits)[self.axis_splits_offset:] if a == "1"]
            axis2_m = [2 for a in str(axis_splits)[self.axis_splits_offset:] if a == "2"]
            axis1_m = reduce(mul, axis1_m)
            axis2_m = reduce(mul, axis2_m)

            m_in = m_t[0, ::axis1_m, ::axis2_m, 0]
            '''

            orig_shape = m_t.shape
            if axis2_m != 1:
                m_t = np.concatenate((m_t, 0. * m_t[:, :, :(axis2_m - 1)]), axis=2)

            # time splits by axis 1
            # freq splits by axis 2
            # BUT we need to swap them to get the right patches...
            # need patches of axis1_m time steps and axis2_m freq steps
            subsampled = patchify(m_t[0, :, :, 0], (axis1_m, axis2_m))
            # log spaced points from 0 to subsampled.shape[1] - 1 - (axis2_m - 1)
            # call it upper_lim = subsampled.shape[1] - 1 - (axis2_m -1)
            # "reverse" geom space with total range (in our case 0 to upper_lim)
            # https://stackoverflow.com/questions/61235157/backwards-np-geomspace-so-a-higher-density-occurs-as-log-function-gets-higher
            # take subset from non-padded patches
            # add 1 because we sub 1 later
            upper_lim = subsampled.shape[1] - (axis2_m - 1)
            # after first run, don't need to search for num_to_try each time
            if self._num_to_try_cache == None:
                num_to_try = self.split_n_mels // 2
            else:
                num_to_try = self._num_to_try_cache

            while True:
                # gradually increase num_to_try until we end up with self.split_n_mels unique points after rounding
                # blend of linear and logarithmic curves for points to take, should help high freqs
                log_grid_points = upper_lim - np.logspace(np.log2(1) / np.log2(1.5), np.log2(upper_lim) / np.log2(1.5), num=num_to_try, base=1.5)
                int_log_grid_points = np.unique(np.round(log_grid_points).astype("int32"))
                lin_grid_points = np.linspace(0, upper_lim, num_to_try)
                int_lin_grid_points = np.unique(np.round(lin_grid_points).astype("int32"))

                grid_points = list(int_lin_grid_points[int_lin_grid_points < .8 * upper_lim]) + list(int_log_grid_points[int_log_grid_points >= .8 * upper_lim])
                grid_points = [u for u in np.unique(np.array(grid_points))]
                # for upper_lim ~256
                # and n_mels 32
                # slight gap at the blend point but should be fine
                # np.array(grid_points[1:]) - np.array(grid_points[:-1])
                # array([12, 11, 11, 12, 12, 11, 11, 12, 12, 11, 11, 12, 12, 11, 11, 12, 12, 13, 10,  8,  6,  4,  4,  2,  3,  1,  1,  1,  1,  1,  1])
                if len(grid_points) >= self.split_n_mels:
                    break
                else:
                    num_to_try += 1
            if self._num_to_try_cache == None:
                self._num_to_try_cache = num_to_try

            if len(grid_points) != self.split_n_mels:
                if len(grid_points) - self.split_n_mels > 1:
                    raise ValueError("More than 1 grid point needs to be dropped, check data preproc in dataset.py and hparams.py")
                # drop the highest freq to make the dims correct
                grid_points = grid_points[:-1]
            assert len(grid_points) == self.split_n_mels
            r_ind_sub = np.sort(grid_points)
            # take every subsampled patch, 0th time index in the patch itself (arbitrary)
            partial_sub = subsampled[::axis1_m, r_ind_sub, 0]
            # take middle frequency in patch (arbitrary but should minimize error at high and low end due to edge effects)
            m_in = partial_sub[:, :, partial_sub.shape[-1] // 2]
            # final result is time, freq
            '''

            """
            r_ind = np.arange(subsampled.shape[1])
            r_ind = [p for p in r_ind if p not in grid_points]
            random_state = np.random.RandomState(1213)
            random_state.shuffle(r_ind)
            r_ind_sub = np.sort(grid_points + r_ind[:(self.split_n_mels - len(grid_points))])

            '''
            for aa in axis_split_list:
                all_m_splits.append(split_np(m_t, axis=aa))
                m_t = all_m_splits[-1][0]
            m_in = all_m_splits[::-1][self.axis_splits_offset][0]
            '''

            # take 0th time element of each patch, we sampled semi-randomly the (overlapped!) patches in frequency, and subsampled in time
            partial_sub = subsampled[::axis1_m, r_ind_sub, 0]
            # back to freq, time - interpolate from the bottom to the top of the patch range over the frequencies
            # some form of meshgrid / mgrid should do this better, TODO
            final_sub = []
            for _i in range(partial_sub.shape[0]):
                pairs_0 = np.arange(partial_sub.shape[1])
                # want to get indices up to but not including 8
                pairs_1 = np.linspace(0, partial_sub.shape[2], num=partial_sub.shape[1], endpoint=False).astype(int)
                pairs = list(zip(pairs_0, pairs_1))
                frame_sub = []
                for _j in range(len(pairs)):
                    frame_sub.append(partial_sub[_i, pairs[_j][0], pairs[_j][1]])
                final_sub.append(np.array(frame_sub)[None])
            m_in = np.concatenate(final_sub)
            """
            # return as freq, time
            m_in = m_in.T
        return m_in, x_t

    def __len__(self):
        return len(self.train_metadata)


def raw_collate(batch) :
    """collate function used for raw wav forms, such as using beta/guassian/mixture of logistic
    """
    
    pad = 2
    mel_win = hp.seq_len // hp.hop_size + 2 * pad
    max_offsets = [x[0].shape[-1] - (mel_win + 2 * pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + pad) * hp.hop_size for offset in mel_offsets]
    
    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] \
            for i, x in enumerate(batch)]
    
    coarse = [x[1][sig_offsets[i]:sig_offsets[i] + hp.seq_len + 1] \
              for i, x in enumerate(batch)]
    
    mels = np.stack(mels).astype(np.float32)
    coarse = np.stack(coarse).astype(np.float32)
    
    mels = torch.FloatTensor(mels)
    coarse = torch.FloatTensor(coarse)
    
    x_input = coarse[:,:hp.seq_len]
    
    y_coarse = coarse[:, 1:]
    
    return x_input, mels, y_coarse



def discrete_collate(batch) :
    """collate function used for discrete wav output, such as 9-bit, mulaw-discrete, etc.
    """
    
    pad = 2
    mel_win = hp.seq_len // hp.hop_size + 2 * pad
    max_offsets = [x[0].shape[-1] - (mel_win + 2 * pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + pad) * hp.hop_size for offset in mel_offsets]
    
    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] \
            for i, x in enumerate(batch)]
    
    coarse = [x[1][sig_offsets[i]:sig_offsets[i] + hp.wav_seq_factor * hp.seq_len + 1] \
              for i, x in enumerate(batch)]
    
    mels = np.stack(mels).astype(np.float32)
    try:
        coarse = np.stack(coarse).astype(np.int64)
    except:
        sz = np.max([len(c) for c in coarse])
        c_errs = [n for n, c in enumerate(coarse) if len(c) != sz]
        #print("error in stacking, possible empty file???")
        # this is weird, bad, and random? some wav file truncated
        # copy a neighbor instead...
        for c_err in c_errs:
            # will wrap around due to negative indexing
            while True:
                idx = c_err - 1
                if idx not in c_errs:
                    break
            coarse[c_err] = coarse[idx].copy()
            mels[c_err] = mels[idx].copy()
        coarse = np.stack(coarse).astype(np.int64)
    
    mels = torch.FloatTensor(mels)
    coarse = torch.LongTensor(coarse)
    if hp.input_type == 'bits':
        x_input = 2 * coarse[:, :hp.wav_seq_factor * hp.seq_len].float() / (2**hp.bits - 1.) - 1.
    elif hp.input_type == 'mulaw':
        x_input = inv_mulaw_quantize(coarse[:, :hp.wav_seq_factor * hp.seq_len], hp.mulaw_quantize_channels)
    
    y_coarse = coarse[:, 1:]
    
    return x_input, mels, y_coarse


def no_test_raw_collate():
    import matplotlib.pyplot as plt
    from test_utils import plot, plot_spec
    data_id_path = "data_dir/"
    data_path = "data_dir/"
    print(hp.seq_len)
    
    with open('{}dataset_ids.pkl'.format(data_id_path), 'rb') as f:
        dataset_ids = pickle.load(f)
    dataset = AudiobookDataset(data_path)
    print(len(dataset))

    data_loader = DataLoader(dataset, collate_fn=raw_collate, batch_size=32, 
                         num_workers=0, shuffle=True)

    x, m, y = next(iter(data_loader))
    print(x.shape, m.shape, y.shape)
    plot(x.numpy()[0]) 
    plot(y.numpy()[0])
    plot_spec(m.numpy()[0])


def test_discrete_collate():
    import matplotlib.pyplot as plt
    from test_utils import plot, plot_spec
    data_id_path = "data_dir/"
    data_path = "data_dir/"
    print(hp.seq_len)
    
    with open('{}dataset_ids.pkl'.format(data_id_path), 'rb') as f:
        dataset_ids = pickle.load(f)
    dataset = AudiobookDataset(data_path)
    print(len(dataset))

    data_loader = DataLoader(dataset, collate_fn=discrete_collate, batch_size=32, 
                         num_workers=0, shuffle=True)

    x, m, y = next(iter(data_loader))
    print(x.shape, m.shape, y.shape)
    plot(x.numpy()[0]) 
    plot(y.numpy()[0])
    plot_spec(m.numpy()[0])



def no_test_dataset():
    data_id_path = "data_dir/"
    data_path = "data_dir/"
    print(hp.seq_len)
    dataset = AudiobookDataset(data_path)
