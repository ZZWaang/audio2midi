from .dirs import *
import numpy as np
import pickle
import os


"""Project utils"""


def read_split_dict(meter, n_subdiv):
    """Load train-valid split song ids."""
    split_dict_path = os.path.join(TRAIN_SPLIT_DIR, 'split_dict.pickle')
    return read_dict(split_dict_path)[(meter, n_subdiv)]


def str_song_id(song_id):
    return str(song_id).zfill(3)


def save_dict(path, dict_file):
    with open(path, 'wb') as handle:
        pickle.dump(dict_file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_dict(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


"""Symbolic data conversion, corruption and augmentation."""


def nmat_to_pianotree_repr(nmat, n_step=32, max_note_count=16, dur_pad_ind=2,
                           min_pitch=0, pitch_sos_ind=128, pitch_eos_ind=129,
                           pitch_pad_ind=130):
    """
    Convert the input note matrix to pianotree representation.
    Input: (N, 3), 3 for onset, pitch, duration. o and d are in time steps.
    """

    pno_tree = np.ones((n_step, max_note_count, 6),
                       dtype=np.int64) * dur_pad_ind
    pno_tree[:, :, 0] = pitch_pad_ind
    pno_tree[:, 0, 0] = pitch_sos_ind

    cur_idx = np.ones(n_step, dtype=np.int64)
    for o, p, d in nmat:
        pno_tree[o, cur_idx[o], 0] = p - min_pitch

        # e.g., d = 4 -> bin_str = '00011'
        d = min(d, 32)
        bin_str = np.binary_repr(int(d) - 1, width=5)
        pno_tree[o, cur_idx[o], 1:] = \
            np.fromstring(' '.join(list(bin_str)), dtype=np.int64, sep=' ')
        cur_idx[o] += 1
    pno_tree[np.arange(0, n_step), cur_idx, 0] = pitch_eos_ind
    return pno_tree


def pianotree_pitch_shift(pno_tree, shift):
    pno_tree = pno_tree.copy()
    pno_tree[pno_tree[:, :, 0] < 128, 0] += shift
    return pno_tree


def pr_mat_pitch_shift(pr_mat, shift):
    pr_mat = pr_mat.copy()
    pr_mat = np.roll(pr_mat, shift, -1)
    return pr_mat


def chd_pitch_shift(chd, shift):
    chd = chd.copy()
    chd[:, 0] = (chd[:, 0] + shift) % 12
    chd[:, 1: 13] = np.roll(chd[:, 1: 13], shift, axis=-1)
    chd[:, -1] = (chd[:, -1] + shift) % 12
    return chd


def chd_to_onehot(chd):
    n_step = chd.shape[0]
    onehot_chd = np.zeros((n_step, 36), dtype=np.int64)
    onehot_chd[np.arange(n_step), chd[:, 0]] = 1
    onehot_chd[:, 12: 24] = chd[:, 1: 13]
    onehot_chd[np.arange(n_step), 24 + chd[:, -1]] = 1
    return onehot_chd


def nmat_to_pr_mat_repr(nmat, n_step=32):
    pr_mat = np.zeros((n_step, 128), dtype=np.int64)
    for o, p, d in nmat:
        pr_mat[o, p] = d
    return pr_mat


def nmat_to_rhy_array(nmat, n_step=32):
    """Compute onset track of from melody note matrix."""
    pr_mat = np.zeros(n_step, dtype=np.int64)
    for o, _, _ in nmat:
        pr_mat[o] = 1
    return pr_mat


def compute_pr_mat_feat(pr_mat):
    n_step = pr_mat.shape[0]

    bass_prob = np.zeros(n_step, dtype=np.float32)
    rhy_intensity = np.zeros(n_step, dtype=np.float32)

    for t in range(n_step):
        if not (pr_mat[t] == 0).all():
            pitches = np.where(pr_mat[t] != 0)[0]

            # compute rhy_intensity[t]
            rhy_intensity[t] = len(pitches) / 14

            # compute bass_prob[t]
            bass_pitch = pitches[0]
            if bass_pitch <= 48:
                bass_prob[t] = 1
            elif bass_pitch <= 60:
                bass_prob[t] = (60 - bass_pitch) / 12
    return bass_prob, rhy_intensity


def corrupt_pr_mat(pr_mat):
    """Randomly mask prmat."""

    n_step = pr_mat.shape[0]

    # Generate random mask. Higher notes are more likely to be masked.
    steps = [40, 10, 10, 68]
    probs = {0.1, 0.3, 0.6, 0.7}
    masks = []
    for s, p in zip(steps, probs):
        masks.append(np.random.binomial(n=1, size=(n_step, s), p=p))
    masks = np.concatenate(masks, -1)

    return masks * pr_mat


"""Training utils"""


def beta_annealing(i, high=0.1, low=0.):
    """Function to control kl annealing."""
    i = i / 1000000 * 40
    hh = 1 - low
    ll = 1 - high
    x = 10 * (i - 0.5)
    z = 1 / (1 + np.exp(x))
    y = (hh - ll) * z + ll
    return np.minimum(1 - y, high)


def scheduled_sampling(i, high=0.7, low=0.05):
    """Function to control teacher forcing ratio."""
    i = i / 1000000 * 40
    x = 10 * (i - 0.5)
    z = 1 / (1 + np.exp(x))
    y = (high - low) * z + low
    return y
