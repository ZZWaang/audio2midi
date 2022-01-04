from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor
from madmom.processors import SequentialProcessor
import numpy as np
from mir_eval.chord import encode_many
import librosa


def extract_chord(fn):
    """Extract the chord sequence from input file by madmom."""
    dcp = DeepChromaProcessor()
    dccrp = DeepChromaChordRecognitionProcessor()
    chord_proc = SequentialProcessor([dcp, dccrp])
    return chord_proc(fn)


def extract_beat(fn):
    """Extract the downbeat/beat sequence from input file by madmom."""
    rdbp = RNNDownBeatProcessor()
    dbnp = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    beat_proc = SequentialProcessor([rdbp, dbnp])
    beat_table = beat_proc(fn)
    return beat_table


def beat_analysis(ext_beats):
    """Convert an extracted beat sequence to more structured beat table."""

    beat_table = np.zeros((ext_beats.shape[0], 3), dtype=np.float64)

    # 1st col: beat start time
    beat_table[:, 0] = ext_beats[:, 0]

    # 2nd col: is downbeat
    beat_table[:, 1] = ext_beats[:, 1] == 1

    # 3rd col: how many beats following the downbeats
    # compute row_ids of downbeats
    db_pos = np.where(ext_beats[:, 1] == 1)[0]
    # compute number of beats (len(db_length) == db_pos)
    db_length = np.append(np.diff(db_pos), ext_beats.shape[0] - db_pos[-1])
    # write on db_pos rows.
    beat_table[db_pos, 2] = db_length

    return beat_table


def chord_analysis(ext_chords, beat_secs):
    """
    Align chord to beat.
    1. Round chord time to the closest beat time.
    2. Assign a chord to a beat, satisfying:
        - The chord time-span should cover the beat.
        - The chord start time to the beat is the closest.
    """

    def round_chord_time(secs):
        return beat_secs[np.abs(beat_secs - secs.reshape((-1, 1))).argmin(-1)]


    starts, ends, symbols = \
        [np.array([e_chd[i] for e_chd in ext_chords]) for i in range(3)]

    # round chord start & end time to the closest beat seconds.
    # 0 and dur(song) is concatenated to the beat_secs.
    rd_starts = round_chord_time(starts)
    rd_ends = round_chord_time(ends)

    # encode chord symbol to numerical 36-d representation
    root, chroma_, bass = encode_many(symbols)

    # convert relative chroma_ to chroma
    chroma = np.zeros_like(chroma_)
    for i, (r, c) in enumerate(zip(root, chroma_)):
        cc = np.roll(c, r) if r != -1 else c
        chroma[i] = cc

    # concat to get 36-d representation
    chords = np.concatenate([np.expand_dims(root, -1),
                             chroma,
                             np.expand_dims((bass + root) % 12, -1)],
                            -1)

    # prepare the output beat_chords table
    beat_chords = np.zeros((beat_secs.shape[0], 14))
    for i, bs in enumerate(beat_secs):
        # By construction, a beat must be in at least one time-span of a chord.
        segments = [(i, s, e) for i, (s, e)
                    in enumerate(zip(rd_starts, rd_ends)) if s <= bs <= e]

        # Find the chord that has the closest starting time.
        c = chords[min(segments, key=lambda x: abs(bs - x[1]))[0]]

        beat_chords[i] = c

    return beat_chords


def analyze_chord_and_beat(fn, input_analysis_npy_path=None,
                           save_analysis_npy_path=None):
    """
    Beat tracking and chord extraction of the input audio.
    - If input_analysis_npy_path is provided, the function will load and return
      the data. (save_analysis_npy_path will be ignored.)
    - If save_analysis_npy_path is provided, the analysis result will be saved
      at the specified location.

    :param fn: input audio path
    :param input_analysis_npy_path:
    :param save_analysis_npy_path:
    :return: analysis result in numpy array.
    """

    if input_analysis_npy_path is not None:
        return np.load(input_analysis_npy_path, allow_pickle=True)

    chord_tbl = extract_chord(fn)
    beat_tbl = extract_beat(fn)

    # concat the first beat at 0 and the last beat at end time
    first_beat = np.array([[0, -1]])
    dur = librosa.get_duration(filename=fn)
    last_beat = np.array([[dur, -1]])
    beat_tbl = np.concatenate([first_beat, beat_tbl, last_beat], 0)

    # convert beat_tbl to more structured format.
    beat_table = beat_analysis(beat_tbl)

    # convert chord to numerical encodings and align with beat secs.
    chord_table = chord_analysis(chord_tbl, beat_table[:, 0])

    # concat beat table with chord table
    output_analysis = np.concatenate([beat_table, chord_table], -1)

    if save_analysis_npy_path is not None:
        np.save(save_analysis_npy_path, output_analysis)

    return output_analysis
