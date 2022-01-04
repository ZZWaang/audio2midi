import torch
from typing import Optional, Tuple
from torch import Tensor
from torchaudio.functional import phase_vocoder, resample
from torchaudio.transforms import MelScale
import math

"""
This python file is a fast and approximate implementation of pitch-shift.
The input is a wave, the output is the mel-spectrogram.
"""


SPECIAL_SR = {-6: 15591 - 9,
              -5: 16518 + 2,
              -4: 17501 - 1,
              -3: 18541 + 9,
              -2: 19644 + 6,
              -1: 20812 + 13,
              0: 22050,
              1: 23361 + 12,
              2: 24750,
              3: 26222 - 7,
              4: 27781 + 2,
              5: 29433 - 3
              }


def pitch_shift_to_spec(
        waveform: Tensor,
        sample_rate: int,
        n_steps: int,
        mel_scale: MelScale,
        n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        window: Optional[Tensor] = None):

    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(window_length=win_length,
                                   device=waveform.device)

    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    ori_len = shape[-1]

    if n_steps != 0:
        new_sr = SPECIAL_SR[n_steps]

        # compute resampling rate from new sample rate
        rate = sample_rate / new_sr

        waveform = resample(waveform, float(new_sr), float(sample_rate))
        waveform = resample(waveform, float(sample_rate), float(16000))
        new_l = waveform.size(-1)
    else:
        rate = 1.0
        waveform = resample(waveform, float(sample_rate), float(16000))


    spec_f = torch.stft(input=waveform,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length,
                        window=window,
                        center=True,
                        pad_mode='reflect',
                        normalized=True,
                        onesided=True,
                        return_complex=True)
    phase_advance = torch.linspace(0, math.pi * hop_length, spec_f.shape[-2],
                                   device=spec_f.device)[..., None]
    spec_stretch = phase_vocoder(spec_f, rate, phase_advance)

    spec_stretch = spec_stretch.abs()
    mel_spec_stretch = mel_scale(spec_stretch)
    mel_spec_stretch = torch.log(torch.clamp(mel_spec_stretch, min=1e-5))

    return mel_spec_stretch
