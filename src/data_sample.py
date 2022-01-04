from .dirs import *
from .constants import *
from .utils import read_dict, nmat_to_pianotree_repr, nmat_to_pr_mat_repr, \
    nmat_to_rhy_array, compute_pr_mat_feat, corrupt_pr_mat
from .utils import str_song_id
import torch
import glob
import torchaudio
import numpy as np


class SongNpz:

    """
    A file loader to read from `data/quantized_pop909_4_bin`, containing
    symbolic data (tracks, chords, beats) stroed in .npz files.

    E.g.,
    >>> song_id = 1  # song_id is the POP909 id from 1 - 909.
    >>> song = SongNpz(song_id)
    >>> song.load()
    """

    def __init__(self, song_id):
        song_id = str_song_id(song_id)
        self.song_id = song_id
        self.npz_fn = os.path.join(QUANTIZED_DATA_DIR, song_id + '.npz')

        # the attributes in the npz file. Initialized to None.
        # melody, bridge, and piano tracks
        self.melody = None
        self.bridge = None
        self.piano = None

        # a table recording beat, and downbeat pos.
        self.beat_lookup = None
        # a table recording chord at each beat.
        self.chord_lookup = None

        # an array recording the time (sec) of each beat.
        self.beat_secs = None

        # an array of downbeat beat_ids.
        self.db_pos = None

        # a bool array indicating whether the downbeats are "complete".
        # "complete" means at least one 4/4 measures after it.
        # E.g.,
        # [1, 2, 3, 4, 1, 2, 3, 4] is complete.
        # [1, 2, 3, 1, 2, 3, 4], [1, 2, 3, 4, 1, 2, 3] are not.
        self.db_pos_filter = None

        # i-th row indicates the starting row of the melody, bridge, and piano
        # tracks at i-th beat.
        self.start_table = None

        # an array of complete downbeats.
        self.filtered_db_pos = None

    def load(self, melody=False, bridge=False, piano=False, beat_lookup=False,
             chord_lookup=False, beat_secs=False, db_pos=False,
             db_pos_filter=False, filtered_db_pos=False, start_table=False):
        """
        A function to load data. Parameter is True means to load, and not to
        otherwise.
        """

        data = np.load(self.npz_fn)

        self.melody = data['melody'] if melody else None
        self.bridge = data['bridge'] if bridge else None
        self.piano = data['piano'] if piano else None

        self.beat_lookup = data['beat_lookup'] if beat_lookup else None
        self.chord_lookup = data['chord_lookup'] if chord_lookup else None

        self.beat_secs = data['beat_positions'] if beat_secs else None

        db_pos_val = data['db_pos']
        db_pos_filter_val = data['db_pos_filter']
        self.db_pos = db_pos_val if db_pos else None
        self.db_pos_filter = db_pos_filter_val if db_pos_filter else None
        self.filtered_db_pos = db_pos_val[db_pos_filter_val] \
            if filtered_db_pos else None

        self.start_table = data['start_table'] if start_table else None

    @staticmethod
    def cat_note_mats(note_mats):
        return np.concatenate(note_mats, 0)

    @staticmethod
    def reset_db_to_zeros(note_mat, db):
        note_mat[:, 0] -= db

    def note_mats_at_db(self, db, note_mats, trk_nos):
        """
        Select rows (notes) of the three tracks which lie between beats
        [db: db + 8].
        """

        s_ind = self.start_table[db][trk_nos]
        e_ind = self.start_table[db + SEG_LGTH][trk_nos]
        seg_mats = [mat[s: e] for s, e, mat in zip(s_ind, e_ind, note_mats)]
        return seg_mats

    def brg_pno_at_db(self, db):
        brg_mat, pno_mat = \
            self.note_mats_at_db(db, [self.bridge, self.piano], [1, 2])
        return brg_mat.copy(), pno_mat.copy()

    def mel_at_db(self, db):
        mel_mat = self.note_mats_at_db(db, [self.melody], [0])[0]
        return mel_mat.copy()

    @staticmethod
    def format_reset_seg_mat(seg_mat):
        """
        The input seg_mat is (N, 5)
        The output seg_mat is (N, 3). Columns for onset, pitch, duration.
        Onset ranges between range(0, 32).
        """

        output_mat = np.zeros((len(seg_mat), 3), dtype=np.int64)
        output_mat[:, 0] = seg_mat[:, 0] * N_BIN + seg_mat[:, 1]
        output_mat[:, 1] = seg_mat[:, 3]
        output_mat[:, 2] = seg_mat[:, 2]
        return output_mat


class SongFolder:

    """
    A file loader to read original POP909 data from `data/audio-midi-combined`.
    * original midi file
    * original audio
    * split audio in the original tempo

    In the current implementation, only split accompaniment is accessible.
    """

    def __init__(self, song_id):
        song_id = str_song_id(song_id)
        self.song_id = song_id
        self.folder = self.match_song_folder()
        self.acc_path = os.path.join(self.folder, 'split_audio',
                                     'accompaniment.wav')

    def match_song_folder(self):
        return glob.glob(os.path.join(COMBINED_DATA_DIR,
                                      f'{self.song_id}*'))[0]


class SongStretched:

    """
    A file loader to read from `data/audio_stretched` containing time-stretched
    and re-sampled audio.
    * The audio is resampled to TGT_SR=22050. Time-stretched so that each beat
      contains STRETCH_FPB=13312 frames (i.e., BPM \approx 99.384).
    * The audio is represented in two files: the wav file and a pickle dict.
    * The wav file contains the time-stretched audio.
    * The pickle dict is a mapping from beats sec's (dict key) to the starting
    frames in the wav file (dict value).
    """

    def __init__(self, song_id):
        song_id = str_song_id(song_id)
        self.audio_stretched_path = STRETCHED_AUDIO_NPY_PATH
        self.song_id = song_id
        self.stretched_wav_path = \
            os.path.join(self.audio_stretched_path, self.song_id + '.wav')
        self.stretched_dict_path = \
            os.path.join(self.audio_stretched_path, self.song_id + '.pickle')
        self.stretched_wav = None
        self.stretched_dict = None

    def load(self):
        wav_data, _ = torchaudio.load(self.stretched_wav_path)
        self.stretched_wav = wav_data[0]
        self.stretched_dict = read_dict(self.stretched_dict_path)


class AudioMidiSampleTemplate:

    """
    A template file loader to read complete data of a song including SongNpz,
    SongFolder, and SongStretched.

    The following methods are to be implemented:
    *  __len__: number of complete 8-beat segments in a song.
    * _get_item_by_db: returns the data contained in a segment indexed by
      downbeat. The function is called by __getitem__.
    * __getitem__: returns the i-th music segment.
    """

    def __init__(self, song_id):
        self.song_npz = SongNpz(song_id)
        self.song_folder = SongFolder(song_id)
        self.song_stretched = SongStretched(song_id)
        self.load()

    def load(self, *params):
        raise NotImplementedError

    """Attributes in self.song_npz"""
    @property
    def song_id(self):
        return self.song_npz.song_id

    @property
    def melody(self):
        return self.song_npz.melody

    @property
    def bridge(self):
        return self.song_npz.bridge

    @property
    def piano(self):
        return self.song_npz.piano

    @property
    def chord_lookup(self):
        return self.song_npz.chord_lookup

    @property
    def beat_secs(self):
        return self.song_npz.beat_secs

    @property
    def filtered_db_pos(self):
        return self.song_npz.filtered_db_pos

    """Attributes in self.song_folder"""
    @property
    def acc_path(self):
        return self.song_folder.acc_path

    """Attributes in self.song_resampled"""
    @property
    def stretched_wav_path(self):
        return self.song_stretched.stretched_wav_path

    @property
    def stretched_dict_path(self):
        return self.song_stretched.stretched_dict_path

    @property
    def stretched_wav(self):
        return self.song_stretched.stretched_wav

    @property
    def stretched_dict(self):
        return self.song_stretched.stretched_dict

    def __len__(self):
        """Return number of complete 8-beat segments in a song."""
        return len(self.filtered_db_pos)

    def _get_item_by_db(self, db):
        """Return the segment indexed by downbeat."""
        raise NotImplementedError

    def __getitem__(self, item):
        """Return a specific music segment by calling _get_item_by_db."""
        db = self.filtered_db_pos[item]
        return self._get_item_by_db(db)


class TrainDataSample(AudioMidiSampleTemplate):

    """
    The class is used to retrieved segments in the training phase. The segment
    contains chord, piano-roll representation, audio, PianoTree representation
    and symbolic features.

    Three additional parameters should be used as follows:
    - Stage-0 (warmup): corrupt=False, autoregressive=False
    - Stage-1 (pre-training): corrupt=True, autoregressive=False
    - Stage-2a (fine-tuning, pure a2s): corrupt ignored, autoregressive=False
    - Stage-2b (fine-tuning, prev-2bar): corrupt ignored, autoregressive=True
    """

    def __init__(self, song_id, audio_pre_step=1000, corrupt=False,
                 autoregressive=False):
        """
        :param song_id: str or int in range(1, 910).
        :param audio_pre_step: how many extra frames to retrieve before the
        downbeat frame.
        :param corrupt: whether to apply random corruption to the symbolic
          piano-roll. Note: bridge track is always removed whatever the value
          of corrupt.
        :param autoregressive: whether to return the previous 2-bar prmat.
        """
        super(TrainDataSample, self).__init__(song_id)
        self.corrupt = corrupt
        self.autoregressive = autoregressive
        self.audio_pre_step = audio_pre_step

        self._nmat_dict = \
            dict(zip(self.filtered_db_pos, [None] * self.__len__()))

        self._pianotree_dict = \
            dict(zip(self.filtered_db_pos, [None] * self.__len__()))

        self._pr_mat_dict = \
            dict(zip(self.filtered_db_pos, [None] * self.__len__()))

        self._feat_dict = \
            dict(zip(self.filtered_db_pos, [None] * self.__len__()))


    def load(self, *params):
        # useless info is not loaded to save memory (e.g., melody).
        self.song_npz.load(bridge=True, piano=True, chord_lookup=True,
                           beat_secs=True,
                           filtered_db_pos=True, start_table=True)
        self.song_stretched.load()

    @property
    def reset_db_to_zeros(self):
        return self.song_npz.__class__.reset_db_to_zeros

    @property
    def format_reset_seg_mat(self):
        return self.song_npz.__class__.format_reset_seg_mat

    @property
    def cat_note_mat(self):
        return self.song_npz.__class__.cat_note_mats

    def store_brg_pno_mat(self, db):
        """
        Retrieve the bridge track and piano track and store them in matrices.
        """
        if self._nmat_dict[db] is not None:
            return

        brg_mat, pno_mat = self.song_npz.brg_pno_at_db(db)

        self.reset_db_to_zeros(brg_mat, db)
        self.reset_db_to_zeros(pno_mat, db)

        brg_mat = self.format_reset_seg_mat(brg_mat)
        pno_mat = self.format_reset_seg_mat(pno_mat)
        nmat = self.cat_note_mat([brg_mat, pno_mat])

        self._nmat_dict[db] = (brg_mat, pno_mat, nmat)

    def store_prmat(self, db):
        """
        Retrieve prmat from note matrices.
        """
        if self._pr_mat_dict[db] is not None:
            return

        brg_mat, pno_mat, nmat = self._nmat_dict[db]
        brg_prmat = nmat_to_pr_mat_repr(brg_mat)
        pno_prmat = nmat_to_pr_mat_repr(pno_mat)
        prmat = nmat_to_pr_mat_repr(nmat)
        self._pr_mat_dict[db] = (brg_mat, pno_prmat, prmat)

    def store_features(self, db):
        """
        Retrieve symbolic features from note matrices.
        """
        if self._feat_dict[db] is not None:
            return

        brg_rhy = nmat_to_rhy_array(self._nmat_dict[db][0])
        bass_prob, pno_intensity = \
            compute_pr_mat_feat(self._pr_mat_dict[db][1])
        self._feat_dict[db] = np.stack([bass_prob, pno_intensity, brg_rhy], -1)

    def store_pno_tree(self, db):
        """
        Retrieve PianoTree representation from note matrices.
        """
        if self._pianotree_dict[db] is not None:
            return

        nmat = self._nmat_dict[db][2]
        pno_tree = nmat_to_pianotree_repr(nmat)
        self._pianotree_dict[db] = pno_tree

    def get_seg_wave(self, db):
        """
        Retrieve the audio segment.
        """
        s_frame = int(self.stretched_dict[self.beat_secs[db]] * STRETCH_FPB)
        e_frame = int(s_frame + SEG_LGTH * STRETCH_FPB)

        # prepending pre steps to the audio segment.
        if s_frame >= self.audio_pre_step:
            return self.stretched_wav[s_frame - self.audio_pre_step: e_frame]
        else:
            pad_lgth = self.audio_pre_step - s_frame
            zero_pads = torch.zeros(pad_lgth, dtype=self.stretched_wav.dtype,
                                    device=self.stretched_wav.device)
            return torch.cat([zero_pads, self.stretched_wav[0: e_frame]], -1)

    def _store(self, db):
        self.store_brg_pno_mat(db)
        self.store_prmat(db)
        self.store_features(db)
        self.store_pno_tree(db)


    def _get_item_by_db(self, db):
        self._store(db)

        # chord
        seg_chd = self.chord_lookup[db: db + SEG_LGTH]

        # wave
        seg_wav = self.get_seg_wave(db)

        # pianotree
        seg_pno_tree = self._pianotree_dict[db]

        # prmat
        if self.autoregressive:
            db_id = np.where(self.filtered_db_pos == db)[0][0]

            # if there exists a previous complete 2 measure
            if db_id != 0 and self.filtered_db_pos[db_id - 1] + 4 == db:
                self._store(db - 4)
                seg_pr_mat = self._pr_mat_dict[db - 4][2]
            # if the previous 2 measure does not exist or incomplete.
            else:
                seg_pr_mat = np.zeros_like(self._pr_mat_dict[db][2])
        else:
            seg_pr_mat = self._pr_mat_dict[db][1]  # Note: piano track only
            if self.corrupt:  # random note masking
                seg_pr_mat = corrupt_pr_mat(seg_pr_mat)

        # symbolic feature
        seg_pr_feat = self._feat_dict[db]

        return seg_pno_tree, seg_chd, seg_wav, seg_pr_mat, seg_pr_feat
