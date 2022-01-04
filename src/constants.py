import numpy as np


N_BIN = 4  # the quantization bins of a beat
SEG_LGTH = 8  # the total beats of a segment

SRC_SR = 44100  # the original sample rate of mp3 and split results
TGT_SR = 22050  # the sample rate to save the data for training, i.e., npy sr
MODEL_SR = 16000

STRETCH_BPM = 100
STRETCH_BEAT_FRAME = (60 / STRETCH_BPM) * TGT_SR
STRETCH_HOP_LGTH = 512
STRETCH_FPB = np.ceil((STRETCH_BEAT_FRAME / STRETCH_HOP_LGTH)) * \
              STRETCH_HOP_LGTH

N_FFT = 2048
HOP_LGTH = 512
N_MELS = 229
INPUT_SR = 16000
F_MIN = 30
F_MAX = None  # torchaudio default
NORM = None  # torchaudio default


MAX_KEPT_RATIO = 0.7
MIN_KEPT_RATIO = 0.25
MAX_ABS_PITCH_SHIFT = 2

AUG_P = np.array([2, 2, 5, 5, 3, 7, 7, 5, 7, 3, 5, 1])

