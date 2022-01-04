from .amc_dl.torch_plus import PytorchModel
from .amc_dl.torch_plus.train_utils import kl_with_normal
import torch
from torch import nn
from .dl_modules import ChordEncoder, ChordDecoder, PianoTreeDecoder, \
    TextureEncoder, FrameEncoder3x153x88, FeatDecoder
from .onsets_and_frames.transcription_utils import \
    load_init_transcription_model
from .onsets_and_frames import OnsetsAndFrames


class Audio2Symb(PytorchModel):

    """ The proposed audio-to-symbolic C-VAE model. """

    writer_names = [
        'loss', 'pno_tree_l', 'pitch_l', 'dur_l', 'kl_l', 'kl_chd', 'kl_aud',
        'kl_sym', 'chord_l', 'root_l', 'chroma_l', 'bass_l', 'feat_l',
        'bass_feat_l', 'int_feat_l', 'rhy_feat_l', 'beta'
    ]

    def __init__(self, name, device,
                 chord_enc: ChordEncoder,
                 chord_dec: ChordDecoder,
                 transcriber: OnsetsAndFrames,
                 frame_enc: FrameEncoder3x153x88,
                 prmat_enc: TextureEncoder,
                 feat_dec: FeatDecoder,
                 pianotree_dec: PianoTreeDecoder,
                 stage=0):
        super(Audio2Symb, self).__init__(name, device)

        self.chord_enc = chord_enc
        self.chord_dec = chord_dec

        # transcriber + frame_enc = audio encoder
        self.transcriber = transcriber
        self.frame_enc = frame_enc

        # symbolic encoder
        self.prmat_enc = prmat_enc

        # feat_dec + pianotree_dec = symbolic decoder
        self.feat_dec = feat_dec  # for symbolic feature recon
        self.feat_emb_layer = nn.Linear(3, 64)
        self.pianotree_dec = pianotree_dec

        # {0: warmup, 1: pre-training, 2: fine-tuning-a, 3: fine-tuning-b}
        assert stage in range(0, 4)
        self.stage = stage

    @property
    def z_chd_dim(self):
        return self.chord_enc.z_dim

    @property
    def z_aud_dim(self):
        return self.frame_enc.z_dim

    @property
    def z_sym_dim(self):
        return self.prmat_enc.z_dim

    def transcriber_encode(self, spec):
        """
        Transcribe the input spectrogram to piano-roll predictions by calling
        Returns onset, frame, velocity predictions (B, 3, 153, 88).
        """

        onset_pred, _, _, frame_pred, velocity = \
            self.transcriber(spec.permute(0, 2, 1))
        frames = torch.stack([onset_pred, frame_pred, velocity], 1)
        return frames

    def audio_enc(self, spec):
        frames = self.transcriber_encode(spec)
        dist_aud = self.frame_enc(frames)
        return dist_aud

    def run(self, pno_tree, chd, spec, pr_mat, feat, tfr1, tfr2, tfr3):
        """
        Forward path of the model in training (w/o computing loss).
        """

        # compute chord representation
        dist_chd = self.chord_enc(chd)
        z_chd = dist_chd.rsample()

        # compute audio-texture representation
        dist_aud = self.audio_enc(spec)
        z_aud = dist_aud.rsample()

        # compute symbolic-texture representation
        if self.stage in [0, 1, 3]:
            dist_sym = self.prmat_enc(pr_mat)
            z_sym = dist_sym.rsample()
        else:  # self.stage == 2 (fine-tuning stage), dist_sym abandoned.
            with torch.no_grad():
                dist_sym = torch.distributions.Normal(
                    torch.zeros(z_aud.size(0), self.z_sym_dim,
                                device=z_aud.device, dtype=z_aud.dtype),
                    torch.ones(z_aud.size(0), self.z_sym_dim,
                               device=z_aud.device, dtype=z_aud.dtype) * 0.1
                )
                z_sym = dist_sym.sample()

        z = torch.cat([z_chd, z_aud, z_sym], -1)

        # reconstruction of chord progression
        recon_root, recon_chroma, recon_bass = \
            self.chord_dec(z_chd, False, tfr3, chd)

        # reconstruct symbolic feature using audio-texture repr.
        recon_feat = self.feat_dec(z_aud, False, tfr1, feat)

        # embed the reconstructed feature (without applying argmax)
        feat_emb = self.feat_emb_layer(recon_feat)

        # prepare the teacher-forcing data for pianotree decoder
        embedded_pno_tree, pno_tree_lgths = self.pianotree_dec.emb_x(pno_tree)

        # pianotree decoder
        recon_pitch, recon_dur = \
            self.pianotree_dec(z, False, embedded_pno_tree,
                               pno_tree_lgths, tfr1, tfr2, feat_emb)


        return recon_pitch, recon_dur, recon_root, recon_chroma, recon_bass, \
            recon_feat, dist_chd, dist_aud, dist_sym

    def loss_function(self, pno_tree, feat, chd, recon_pitch, recon_dur,
                      recon_root, recon_chroma, recon_bass, recon_feat,
                      dist_chd, dist_aud, dist_sym,
                      beta, weights):
        """ Compute the loss from ground truth and the output of self.run()"""
        # pianotree recon loss
        pno_tree_l, pitch_l, dur_l = \
            self.pianotree_dec.recon_loss(pno_tree, recon_pitch, recon_dur,
                                          weights, False)

        # chord recon loss
        chord_l, root_l, chroma_l, bass_l = \
            self.chord_dec.recon_loss(chd, recon_root,
                                      recon_chroma, recon_bass)

        # feature prediction loss
        feat_l, bass_feat_l, int_feat_l, rhy_feat_l = \
            self.feat_dec.recon_loss(feat, recon_feat)

        # kl losses
        kl_chd = kl_with_normal(dist_chd)
        kl_aud = kl_with_normal(dist_aud)
        kl_sym = kl_with_normal(dist_sym)

        if self.stage == 0:
            # beta keeps annealing from 0 - 0.01
            kl_l = beta * (kl_chd + kl_aud + kl_sym)
        elif self.stage == 1:
            # beta keeps annealing from 0.01 - 0.5
            kl_l = 0.01 * kl_chd + 0.01 * kl_aud + beta * kl_sym
        elif self.stage == 2:
            # kl_sym is not computed because symbolic input is abandoned.
            kl_l = 0.01 * kl_chd + 0.01 * kl_aud
        else:  # self.stage == 3
            # autoregressive fine-tuning
            kl_l = 0.01 * kl_chd + 0.01 * kl_aud + 0.5 * kl_sym

        loss = pno_tree_l + chord_l + feat_l + kl_l

        return loss, pno_tree_l, pitch_l, dur_l, kl_l, \
            kl_chd, kl_aud, kl_sym, chord_l, root_l, chroma_l, bass_l, \
            feat_l, bass_feat_l, int_feat_l, rhy_feat_l, torch.tensor(beta)

    def loss(self, pno_tree, chd, spec, pr_mat, feat, tfr1, tfr2, tfr3,
             beta=0.1, weights=(1, 0.5)):
        """
        Forward path during training with loss computation.

        :param pno_tree: (B, 32, 16, 6) ground truth for teacher forcing
        :param chd: (B, 8, 36) chord input
        :param spec: (B, 229, 153) audio input.
            Log mel-spectrogram. (n_mels=229)
        :param pr_mat: (B, 32, 128) (with proper corruption) symbolic input.
        :param feat: (B, 32, 3) ground truth for teacher forcing
        :param tfr1: teacher forcing ratio 1 (1st-hierarchy RNNs except chord)
        :param tfr2: teacher forcing ratio 2 (2nd-hierarchy RNNs except chord)
        :param tfr3: teacher forcing ratio 3 (for chord decoder)
        :param beta: kl annealing parameter
        :param weights: weighting parameter for pitch and dur in PianoTree.
        :return: losses (first argument is the total loss.)
        """

        recon_pitch, recon_dur, recon_root, recon_chroma, recon_bass, \
            recon_feat, dist_chd, dist_aud, dist_sym = \
            self.run(pno_tree, chd, spec, pr_mat, feat, tfr1, tfr2, tfr3)

        return self.loss_function(
            pno_tree, feat, chd, recon_pitch, recon_dur,
            recon_root, recon_chroma, recon_bass, recon_feat,
            dist_chd, dist_aud, dist_sym, beta, weights)

    @classmethod
    def init_model(cls, stage, z_chd_dim=128, z_aud_dim=192, z_sym_dim=192,
                   transcriber_path=None, model_path=None):
        """Fast model initialization."""

        name = 'audio2midi'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        chord_enc = ChordEncoder(36, 256, z_chd_dim)
        chord_dec = ChordDecoder(z_dim=z_chd_dim)

        if transcriber_path is None:
            transcriber = load_init_transcription_model(device)
        else:
            transcriber = load_init_transcription_model(device)
            dic = torch.load(transcriber_path, map_location=device)
            transcriber.load_state_dict(dic)

        frame_enc = FrameEncoder3x153x88(z_dim=z_aud_dim)

        prmat_enc = TextureEncoder(z_dim=z_sym_dim)

        feat_dec = FeatDecoder(z_dim=z_aud_dim)

        z_pt_dim = z_chd_dim + z_aud_dim + z_sym_dim
        pianotree_dec = PianoTreeDecoder(z_size=z_pt_dim, feat_emb_dim=64)

        model = cls(name, device, chord_enc, chord_dec,
                    transcriber, frame_enc, prmat_enc, feat_dec,
                    pianotree_dec, stage).to(device)
        if model_path is not None:
            model.load_model(model_path=model_path, map_location=device)
        return model


    def inference(self, audio, chord, sym_prompt=None):
        """
        Forward path during inference. By default, symbolic source is not used.

        :param audio: (B, 229, 153) audio input.
            Log mel-spectrogram. (n_mels=229)
        :param chord: (B, 8, 36) chord input
        :param sym_prompt: (B, 32, 128) symbolic prompt.
            By default, None.
        :return: pianotree prediction (B, 32, 15, 6) numpy array.
        """

        self.eval()
        with torch.no_grad():
            z_chd = self.chord_enc(chord).mean
            z_aud = self.audio_enc(audio).mean

            z_sym = \
                torch.zeros(z_aud.size(0), self.z_sym_dim,
                            dtype=z_aud.dtype, device=z_aud.device) \
                if sym_prompt is None else self.prmat_enc(sym_prompt).mean

            z = torch.cat([z_chd, z_aud, z_sym], -1)

            recon_feat = self.feat_dec(z_aud, True, 0., None)
            feat_emb = self.feat_emb_layer(recon_feat)
            recon_pitch, recon_dur = \
                self.pianotree_dec(z, True, None, None, 0., 0., feat_emb)

        # convert to (argmax) pianotree format, numpy array.
        pred = self.pianotree_dec.output_to_numpy(recon_pitch.cpu(),
                                                  recon_dur.cpu())[0]
        return pred


class Audio2SymbNoChord(PytorchModel):

    """
    A simplified proposed model w/o chord representation.
    The model should have comparable generation quality.
    - The advantage is there is no need to do chord extraction.
    - The disadvantage is compositional style transfer is no longer possible.
    """

    writer_names = [
        'loss', 'pno_tree_l', 'pitch_l', 'dur_l', 'kl_l', 'kl_aud', 'kl_sym',
        'feat_l', 'bass_feat_l', 'int_feat_l', 'rhy_feat_l', 'beta'
    ]

    def __init__(self, name, device,
                 transcriber: OnsetsAndFrames,
                 frame_enc: FrameEncoder3x153x88,
                 prmat_enc: TextureEncoder,
                 feat_dec: FeatDecoder,
                 pianotree_dec: PianoTreeDecoder,
                 stage=0):
        super(Audio2SymbNoChord, self).__init__(name, device)

        # transcriber + frame_enc = audio encoder
        self.transcriber = transcriber
        self.frame_enc = frame_enc

        # symbolic encoder
        self.prmat_enc = prmat_enc

        # feat_dec + pianotree_dec = symbolic decoder
        self.feat_dec = feat_dec  # for symbolic feature recon
        self.feat_emb_layer = nn.Linear(3, 64)
        self.pianotree_dec = pianotree_dec

        # {0: warmup, 1: pre-training, 2: fine-tuning-a, 3: fine-tuning-b}
        assert stage in range(0, 4)
        self.stage = stage

    @property
    def z_aud_dim(self):
        return self.frame_enc.z_dim

    @property
    def z_sym_dim(self):
        return self.prmat_enc.z_dim

    def transcriber_encode(self, spec):
        """
        Transcribe the input spectrogram to piano-roll predictions by calling
        Returns onset, frame, velocity predictions (B, 3, 153, 88).
        """

        onset_pred, _, _, frame_pred, velocity = \
            self.transcriber(spec.permute(0, 2, 1))
        frames = torch.stack([onset_pred, frame_pred, velocity], 1)
        return frames

    def audio_enc(self, spec):
        frames = self.transcriber_encode(spec)
        dist_aud = self.frame_enc(frames)
        return dist_aud

    def run(self, pno_tree, spec, pr_mat, feat, tfr1, tfr2):
        """
        Forward path of the model in training (w/o computing loss).
        """

        # compute audio-texture representation
        dist_aud = self.audio_enc(spec)
        z_aud = dist_aud.rsample()

        # compute symbolic-texture representation
        if self.stage in [0, 1, 3]:
            dist_sym = self.prmat_enc(pr_mat)
            z_sym = dist_sym.rsample()
        else:  # self.stage == 2 (fine-tuning stage), dist_sym abandoned.
            with torch.no_grad():
                dist_sym = torch.distributions.Normal(
                    torch.zeros(z_aud.size(0), self.z_sym_dim,
                                device=z_aud.device, dtype=z_aud.dtype),
                    torch.ones(z_aud.size(0), self.z_sym_dim,
                               device=z_aud.device, dtype=z_aud.dtype) * 0.1
                )
                z_sym = dist_sym.sample()

        z = torch.cat([z_aud, z_sym], -1)

        # reconstruct symbolic feature using audio-texture repr.
        recon_feat = self.feat_dec(z_aud, False, tfr1, feat)

        # embed the reconstructed feature (without applying argmax)
        feat_emb = self.feat_emb_layer(recon_feat)

        # prepare the teacher-forcing data for pianotree decoder
        embedded_pno_tree, pno_tree_lgths = self.pianotree_dec.emb_x(pno_tree)

        # pianotree decoder
        recon_pitch, recon_dur = \
            self.pianotree_dec(z, False, embedded_pno_tree,
                               pno_tree_lgths, tfr1, tfr2, feat_emb)

        return recon_pitch, recon_dur, recon_feat, dist_aud, dist_sym

    def loss_function(self, pno_tree, feat, recon_pitch, recon_dur,
                      recon_feat, dist_aud, dist_sym, beta, weights):
        """ Compute the loss from ground truth and the output of self.run()"""
        # pianotree recon loss
        pno_tree_l, pitch_l, dur_l = \
            self.pianotree_dec.recon_loss(pno_tree, recon_pitch, recon_dur,
                                          weights, False)

        # feature prediction loss
        feat_l, bass_feat_l, int_feat_l, rhy_feat_l = \
            self.feat_dec.recon_loss(feat, recon_feat)

        # kl losses
        kl_aud = kl_with_normal(dist_aud)
        kl_sym = kl_with_normal(dist_sym)

        if self.stage == 0:
            # beta keeps annealing from 0 - 0.01
            kl_l = beta * (kl_aud + kl_sym)
        elif self.stage == 1:
            # beta keeps annealing from 0.01 - 0.5
            kl_l = 0.01 * kl_aud + beta * kl_sym
        elif self.stage == 2:
            # kl_sym is not computed because symbolic input is abandoned.
            kl_l = 0.01 * kl_aud
        else:  # self.stage == 3
            # autoregressive fine-tuning
            kl_l = 0.01 * kl_aud + 0.5 * kl_sym

        loss = pno_tree_l + feat_l + kl_l

        return loss, pno_tree_l, pitch_l, dur_l, kl_l, kl_aud, kl_sym, \
            feat_l, bass_feat_l, int_feat_l, rhy_feat_l, torch.tensor(beta)

    def loss(self, pno_tree, chd, spec, pr_mat, feat, tfr1, tfr2, tfr3,
             beta=0.1, weights=(1, 0.5)):
        """
        Forward path during training with loss computation.

        :param pno_tree: (B, 32, 16, 6) ground truth for teacher forcing
        :param chd: ignored.
        :param spec: (B, 229, 153) audio input.
            Log mel-spectrogram. (n_mels=229)
        :param pr_mat: (B, 32, 128) (with proper corruption) symbolic input.
        :param feat: (B, 32, 3) ground truth for teacher forcing
        :param tfr1: teacher forcing ratio 1 (1st-hierarchy RNNs except chord)
        :param tfr2: teacher forcing ratio 2 (2nd-hierarchy RNNs except chord)
        :param tfr3: ignored.
        :param beta: kl annealing parameter
        :param weights: weighting parameter for pitch and dur in PianoTree.
        :return: losses (first argument is the total loss.)
        """

        recon_pitch, recon_dur, recon_feat, dist_aud, dist_sym = \
            self.run(pno_tree, spec, pr_mat, feat, tfr1, tfr2)

        return self.loss_function(
            pno_tree, feat, recon_pitch, recon_dur,
            recon_feat, dist_aud, dist_sym, beta, weights)

    @classmethod
    def init_model(cls, stage, z_aud_dim=320, z_sym_dim=192,
                   transcriber_path=None, model_path=None):
        """Fast model initialization."""

        name = 'audio2midi-nochord'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if transcriber_path is None:
            transcriber = load_init_transcription_model(device)
        else:
            transcriber = load_init_transcription_model(device)
            dic = torch.load(transcriber_path, map_location=device)
            transcriber.load_state_dict(dic)

        frame_enc = FrameEncoder3x153x88(z_dim=z_aud_dim)

        prmat_enc = TextureEncoder(z_dim=z_sym_dim)

        feat_dec = FeatDecoder(z_dim=z_aud_dim)

        z_pt_dim = z_aud_dim + z_sym_dim
        pianotree_dec = PianoTreeDecoder(z_size=z_pt_dim, feat_emb_dim=64)

        model = cls(name, device, transcriber, frame_enc, prmat_enc, feat_dec,
                    pianotree_dec, stage).to(device)
        if model_path is not None:
            model.load_model(model_path=model_path, map_location=device)
        return model

    def inference(self, audio, sym_prompt=None):
        """
        Forward path during inference. By default, symbolic source is not used.

        :param audio: (B, 229, 153) audio input.
            Log mel-spectrogram. (n_mels=229)
        :param sym_prompt: (B, 32, 128) symbolic prompt.
            By default, None.
        :return: pianotree prediction (B, 32, 15, 6) numpy array.
        """

        self.eval()
        with torch.no_grad():
            z_aud = self.audio_enc(audio).mean

            z_sym = \
                torch.zeros(z_aud.size(0), self.z_sym_dim,
                            dtype=z_aud.dtype, device=z_aud.device) \
                if sym_prompt is None else self.prmat_enc(sym_prompt).mean

            z = torch.cat([z_aud, z_sym], -1)

            recon_feat = self.feat_dec(z_aud, True, 0., None)
            feat_emb = self.feat_emb_layer(recon_feat)
            recon_pitch, recon_dur = \
                self.pianotree_dec(z, True, None, None, 0., 0., feat_emb)

        # convert to (argmax) pianotree format, numpy array.
        pred = self.pianotree_dec.output_to_numpy(recon_pitch.cpu(),
                                                  recon_dur.cpu())[0]
        return pred


class Audio2SymbSupervised(PytorchModel):

    """
    A naive supervised approach for audio-to-symbolic arrangement.
    Since audio is measured in frame while symbolic is measured in 16-th note,
      we still use a bottle-neck, or latent representation. The bottle-neck
      is or is not constrained by a kl loss.
    Equivalently, this is simply a seq2seq model structure with or without kl
      penalty.
    """

    writer_names = [
        'loss', 'pno_tree_l', 'pitch_l', 'dur_l', 'kl_l', 'beta'
    ]

    def __init__(self, name, device,
                 transcriber: OnsetsAndFrames,
                 frame_enc: FrameEncoder3x153x88,
                 pianotree_dec: PianoTreeDecoder,
                 kl_penalty=False):
        super(Audio2SymbSupervised, self).__init__(name, device)

        # transcriber + frame_enc = audio encoder
        self.transcriber = transcriber
        self.frame_enc = frame_enc

        self.pianotree_dec = pianotree_dec
        self.kl_penalty = kl_penalty

    @property
    def z_aud_dim(self):
        return self.frame_enc.z_dim

    def transcriber_encode(self, spec):
        """
        Transcribe the input spectrogram to piano-roll predictions by calling
        Returns onset, frame, velocity predictions (B, 3, 153, 88).
        """

        onset_pred, _, _, frame_pred, velocity = \
            self.transcriber(spec.permute(0, 2, 1))
        frames = torch.stack([onset_pred, frame_pred, velocity], 1)
        return frames

    def audio_enc(self, spec):
        frames = self.transcriber_encode(spec)
        dist_aud = self.frame_enc(frames)
        return dist_aud

    def run(self, pno_tree, spec, pr_mat, feat, tfr1, tfr2):
        """
        Forward path of the model in training (w/o computing loss).
        """

        # compute audio-texture representation
        dist_z = self.audio_enc(spec)
        z = dist_z.rsample() if self.kl_penalty else dist_z.mean

        # prepare the teacher-forcing data for pianotree decoder
        embedded_pno_tree, pno_tree_lgths = self.pianotree_dec.emb_x(pno_tree)

        # pianotree decoder
        recon_pitch, recon_dur = \
            self.pianotree_dec(z, False, embedded_pno_tree,
                               pno_tree_lgths, tfr1, tfr2, None)

        return recon_pitch, recon_dur, dist_z

    def loss_function(self, pno_tree, recon_pitch, recon_dur,
                      dist_z, beta, weights):
        """ Compute the loss from ground truth and the output of self.run()"""
        # pianotree recon loss
        pno_tree_l, pitch_l, dur_l = \
            self.pianotree_dec.recon_loss(pno_tree, recon_pitch, recon_dur,
                                          weights, False)

        # kl losses
        kl_l = kl_with_normal(dist_z)

        loss = pno_tree_l + beta * kl_l if self.kl_penalty else pno_tree_l

        return loss, pno_tree_l, pitch_l, dur_l, kl_l, torch.tensor(beta)

    def loss(self, pno_tree, chd, spec, pr_mat, feat, tfr1, tfr2, tfr3,
             beta=0.1, weights=(1, 0.5)):
        """
        Forward path during training with loss computation.

        :param pno_tree: (B, 32, 16, 6) ground truth for teacher forcing
        :param chd: ignored.
        :param spec: (B, 229, 153) audio input.
            Log mel-spectrogram. (n_mels=229)
        :param pr_mat: ignored
        :param feat: ignored
        :param tfr1: teacher forcing ratio 1 (1st-hierarchy RNNs except chord)
        :param tfr2: teacher forcing ratio 2 (2nd-hierarchy RNNs except chord)
        :param tfr3: ignored.
        :param beta: kl annealing parameter
        :param weights: weighting parameter for pitch and dur in PianoTree.
        :return: losses (first argument is the total loss.)
        """

        recon_pitch, recon_dur, dist_z = \
            self.run(pno_tree, spec, pr_mat, feat, tfr1, tfr2)

        return self.loss_function(
            pno_tree, recon_pitch, recon_dur, dist_z, beta, weights)

    @classmethod
    def init_model(cls, kl_penalty=False, z_dim=512,
                   transcriber_path=None, model_path=None):
        """Fast model initialization."""

        name = 'audio2midi-supervised'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if transcriber_path is None:
            transcriber = load_init_transcription_model(device)
        else:
            transcriber = load_init_transcription_model(device)
            dic = torch.load(transcriber_path, map_location=device)
            transcriber.load_state_dict(dic)

        frame_enc = FrameEncoder3x153x88(z_dim=z_dim)

        pianotree_dec = PianoTreeDecoder(z_size=z_dim, feat_emb_dim=0)

        model = cls(name, device, transcriber, frame_enc,
                    pianotree_dec, kl_penalty=kl_penalty).to(device)
        if model_path is not None:
            model.load_model(model_path=model_path, map_location=device)
        return model

    def inference(self, audio):
        """
        Forward path during inference. By default, symbolic source is not used.

        :param audio: (B, 229, 153) audio input.
            Log mel-spectrogram. (n_mels=229)
        :return: pianotree prediction (B, 32, 15, 6) numpy array.
        """

        self.eval()
        with torch.no_grad():
            z = self.audio_enc(audio).mean

            recon_pitch, recon_dur = \
                self.pianotree_dec(z, True, None, None, 0., 0., None)

        # convert to (argmax) pianotree format, numpy array.
        pred = self.pianotree_dec.output_to_numpy(recon_pitch.cpu(),
                                                  recon_dur.cpu())[0]
        return pred
