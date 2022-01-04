from src.constants import *
from src.audio_sp_modules.time_stretch import time_stretch
from math import ceil
import numpy as np


def pad_audio_npy(audio, beat_secs, exceed_frames=1000):
    """
    This operation generates a copy of the wav and ensures
    len(copy) >= frame of beat_secs[-1] * TGT_SR + exceed_frames
    """
    last_beat_frame = beat_secs[-1] * TGT_SR
    last_audio_frame = len(audio) - 1
    if last_audio_frame < last_beat_frame + exceed_frames:
        pad_data = np.zeros(ceil(last_beat_frame + exceed_frames),
                            dtype=np.float32)
        pad_data[0: len(audio)] = audio
    else:
        pad_data = audio.copy()
    return pad_data


def stretch_a_song(beat_secs, audio, tgt_bpm=100, exceed_frames=1000):
    """Stretch the audio to constant bpm=tgt_bpm."""
    data = pad_audio_npy(audio, beat_secs, exceed_frames=exceed_frames)
    if beat_secs[0] > HOP_LGTH / TGT_SR:
        critical_beats = np.insert(beat_secs, 0, 0)
        beat_dict = dict(zip(beat_secs,
                             np.arange(0, len(beat_secs)) + 1))
    else:
        critical_beats = beat_secs
        beat_dict = dict(zip(beat_secs,
                             np.arange(0, len(beat_secs))))

    critical_frames = critical_beats * TGT_SR
    critical_frames = np.append(critical_frames, len(data))

    frame_intervals = np.diff(critical_frames)
    tgt_interval = (60 / tgt_bpm) * TGT_SR
    rates = frame_intervals / tgt_interval

    steps = [np.arange(critical_frames[i] / HOP_LGTH,
                       critical_frames[i + 1] / HOP_LGTH,
                       rates[i])
             for i in range(len(frame_intervals))]

    time_steps = np.concatenate(steps, dtype=float)

    fpb = np.ceil((tgt_interval / HOP_LGTH)) * HOP_LGTH
    len_stretch = int(fpb * len(steps))

    stretched_song = time_stretch(data, time_steps, len_stretch,
                                  center=False)
    return stretched_song, int(fpb), rates



