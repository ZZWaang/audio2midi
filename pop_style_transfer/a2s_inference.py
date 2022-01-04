from .time_stretch_song import stretch_a_song
import torch
from src.utils import chd_to_onehot
from src.audio_sp_modules.pitch_shift import pitch_shift_to_spec
from src.constants import *
from torchaudio.transforms import MelScale


def segment_a_song(analysis, stretched_song, spb):
    """
    Segment chord and audio into 2-bar segments.
    Note (a possible acceleration): the segmentation selects bar [0 - 2],
    [1 - 3], [2 - 4] etc. However, in the current implementation, we only need
    [0 - 2], [2 - 4], etc.
    We use such implementation in case [1 - 3] will become useful under more
    finer implementation of conversion.
    """

    pre_step = 1000
    db_ids = np.where(analysis[:, 1])[0]  # downbeat positions

    # initialize output audios and chords
    audios = np.zeros((len(db_ids), spb * 8 + pre_step), dtype=np.float64)
    chords = np.zeros((len(db_ids), 8, 14), dtype=np.int64)

    for i, db_id in enumerate(db_ids):
        # segment an audio (prepending previous pre_step frames)
        audio_start = db_id * spb
        audio_end = audio_start + 8 * spb

        if audio_start >= pre_step:
            audio_start = audio_start - pre_step
            seg_audio = stretched_song[audio_start: audio_end]
        else:
            zero_pads = np.zeros(pre_step - audio_start,
                                 dtype=stretched_song.dtype)
            seg_audio = stretched_song[0: audio_end]
            seg_audio = np.concatenate([zero_pads, seg_audio], 0)

        # segment chords
        chord_start = db_id
        chord_end = db_id + 8
        seg_chord = analysis[chord_start: chord_end, 3:]

        # pad the last sample if necessary
        if audio_end > len(stretched_song):
            pad_audio = np.zeros(audio_end - len(stretched_song),
                                 dtype=stretched_song.dtype)
            pad_chord = np.zeros((chord_end - analysis.shape[0], 14),
                                 dtype=seg_chord.dtype)

            seg_audio = np.concatenate([seg_audio, pad_audio], 0)
            seg_chord = np.concatenate([seg_chord, pad_chord], 0)

        audios[i] = seg_audio
        chords[i] = seg_chord
    return audios, chords


def model_compute(model, audios, chords, batch_size, device, require_chord):
    """Batching the input and call model.inference()."""

    batch_starts = np.arange(0, len(audios), batch_size)
    predictions = []

    for start in batch_starts:  # batching
        audio = audios[start: start + batch_size]
        chord = chords[start: start + batch_size]

        # convert chord to 36-d representation
        chord = np.stack([chd_to_onehot(c) for c in chord])
        chord = torch.from_numpy(chord).float().to(device)

        # convert audio to log mel-spectrogram.
        audio = torch.from_numpy(audio).float().to(device)

        mel_scale = MelScale(n_mels=N_MELS, sample_rate=INPUT_SR, f_min=F_MIN,
                             f_max=F_MAX, n_stft=N_FFT // 2 + 1).to(device)
        audio = pitch_shift_to_spec(audio, TGT_SR, 0,
                                    n_fft=N_FFT, hop_length=HOP_LGTH,
                                    mel_scale=mel_scale)
        if require_chord:
            predictions.append(model.inference(audio, chord))
        else:
            predictions.append(model.inference(audio))

    predictions = np.concatenate(predictions, 0)
    return predictions


def model_compute_autoregressive(model, model0, audios, chords, device,
                                 analysis, require_chord):
    """Feed data in batch_size=1 in an autoregressive fashion."""
    db_ids = np.where(analysis[:, 1])[0]
    predictions = np.zeros((len(db_ids), 32, 15, 6), dtype=np.int64)

    for i, (audio, chord) in enumerate(zip(audios, chords)):
        audio = torch.from_numpy(audio).unsqueeze(0).float().to(device)

        mel_scale = MelScale(n_mels=N_MELS, sample_rate=INPUT_SR, f_min=F_MIN,
                             f_max=F_MAX, n_stft=N_FFT // 2 + 1).to(device)
        audio = pitch_shift_to_spec(audio, TGT_SR, 0,
                                    n_fft=N_FFT, hop_length=HOP_LGTH,
                                    mel_scale=mel_scale)

        chord = chd_to_onehot(chord)
        chord = torch.from_numpy(chord).unsqueeze(0).float().to(device)

        if i == 0 or analysis[db_ids[i - 1], 2] != 4:
            pred = model0.inference(audio, chord, None) if require_chord \
                else model0.inference(audio, None)
        else:
            prev_pnotree = predictions[i - 1]

            prev_prmat = model.pianotree_dec. \
                grid_to_pr_and_notes(prev_pnotree)[0]

            prev_prmat = torch.from_numpy(prev_prmat).unsqueeze(0).\
                float().to(device)

            pred = model.inference(audio, chord, prev_prmat) if require_chord \
                else model.inference(audio, prev_prmat)

        predictions[i] = pred

    return predictions


def audio_to_symbolic_prediction(model, model0, analysis,
                                 audio, batch_size):

    device = model.device
    autoregressive = model0 is not None
    require_chords = model.__class__.__name__ == 'Audio2Symb'

    to_notes_func = lambda x: model.pianotree_dec. \
        grid_to_pr_and_notes(x, 60., 0., False)[1]

    # time-stretch a song to equal tempo
    stretched_song, spb, rates = stretch_a_song(analysis[:, 0], audio)

    # segment a song into 2-bar segments (batches)
    audios, chords = segment_a_song(analysis, stretched_song, spb)

    # calling the model on data batches
    if autoregressive:
        predictions = \
            model_compute_autoregressive(model, model0, audios, chords, device,
                                         analysis, require_chords)
    else:
        predictions = \
            model_compute(model, audios, chords, batch_size,
                          device, require_chords)

    return predictions, to_notes_func
