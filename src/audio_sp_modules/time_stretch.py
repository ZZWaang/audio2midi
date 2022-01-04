from librosa import core
import numpy as np


"""
Modified from the librosa implementation. 

Time-stretched can be applied so that a varing-bpm can be stretched to even 
bpm.
"""


def phase_vocoder(D, time_steps, hop_length=None):

    n_fft = 2 * (D.shape[0] - 1)

    if hop_length is None:
        hop_length = int(n_fft // 4)


    # Create an empty output array
    d_stretch = np.zeros((D.shape[0], len(time_steps)), D.dtype, order="F")

    # Expected phase advance in each bin
    phi_advance = np.linspace(0, np.pi * hop_length, D.shape[0])

    # Phase accumulator; initialize to the first sample
    phase_acc = np.angle(D[:, 0])

    # Pad 0 columns to simplify boundary logic
    D = np.pad(D, [(0, 0), (0, 10)], mode="constant")

    for (t, step) in enumerate(time_steps):

        columns = D[:, int(step) : int(step + 2)]

        # Weighting for linear magnitude interpolation
        alpha = np.mod(step, 1.0)

        mag = (1.0 - alpha) * np.abs(columns[:, 0]) + \
              alpha * np.abs(columns[:, 1])

        # Store to output array
        d_stretch[:, t] = mag * np.exp(1.0j * phase_acc)

        # Compute phase advance
        dphase = np.angle(columns[:, 1]) - \
                 np.angle(columns[:, 0]) - phi_advance

        # Wrap to -pi:pi range
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))

        # Accumulate phase
        phase_acc += phi_advance + dphase

    return d_stretch


def time_stretch(y, time_steps, len_stretch, **kwargs):
    stft = core.stft(y, **kwargs)

    # Stretch by phase vocoding
    stft_stretch = phase_vocoder(stft, time_steps)

    y_stretch = core.istft(stft_stretch, dtype=y.dtype, length=len_stretch,
                           **kwargs)

    return y_stretch
