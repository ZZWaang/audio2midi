import os
import sys
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                 '..')))
import librosa
from .chord_and_beat import analyze_chord_and_beat
from .a2s_inference import audio_to_symbolic_prediction
from .perf_render import write_prediction
from scripts.a2s_config import prepare_model
import warnings

warnings.simplefilter('ignore', UserWarning)


# Model pt files directory
# Ensuring path is valid when importing from outside the project.
PARAM_PATH = \
    os.path.realpath(os.path.join(os.path.dirname(__file__), '..',
                                  'params'))

MODEL_PATHS = {
    ('a2s', True): os.path.join(PARAM_PATH, 'a2s-stage3b.pt'),
    ('a2s', False): os.path.join(PARAM_PATH, 'a2s-stage3a.pt'),
    ('a2s-alt', False): os.path.join(PARAM_PATH, 'a2s-stage3a-alt.pt'),
    ('a2s-nochd', True): os.path.join(PARAM_PATH, 'a2s-nochd-stage3b.pt'),
    ('a2s-nochd', False): os.path.join(PARAM_PATH, 'a2s-nochd-stage3a.pt'),
    ('supervised', False): os.path.join(PARAM_PATH, 'supervised.pt'),
    ('supervised+kl', False): os.path.join(PARAM_PATH, 'supervised+kl.pt')
}


def load_model(model_id, autoregressive, alt=False):
    """
    Load pre-trained proposed a2s models or baseline models.

    :param model_id: str in ['a2s', 'a2s-nochd', 'supervised', 'supervised+kl']
    :param autoregressive: bool.
        True for autoregressive mode (stage 3b), and stage 3a otherwise.
        Autoregressive can only be True for 'a2s' and 'a2s-nochd' model_id's.
    :param alt:  bool.
        We provide two stage 3a models. alt=True for the alternative model.
        alt can only be True for 'a2s' model_id.
    :return: stage 3b and 3a model if autoregressive (for transition and init),
        stage 3a model and None otherwise.
    """

    assert model_id in ['a2s', 'a2s-nochd', 'supervised', 'supervised+kl']

    if model_id in ['supervised', 'supervised+kl'] and autoregressive:
        raise ValueError("Autoregressive mode can only applied to a2s "
                         "and a2s-nochd model_id's.")
    if model_id != 'a2s' and alt:
        raise ValueError("Alt model can only be applied to a2s model_id.")

    model_id_ = '-'.join([model_id, 'alt']) if alt else model_id

    model = prepare_model(model_id,
                          stage=0,
                          model_path=MODEL_PATHS[model_id if autoregressive
                                                 else model_id_,
                                                 autoregressive])

    model0 = prepare_model(model_id, stage=0,
                           model_path=MODEL_PATHS[model_id_, False]) \
        if autoregressive else None

    return model, model0


def acc_audio_to_midi(input_acc_audio_path, output_midi_path,
                      model, model0=None,
                      input_audio_path=None, input_analysis_npy_path=None,
                      save_analysis_npy_path=None, batch_size=32):
    """
    Convert an input accompaniment audio to its piano arrangement in MIDI.
    The output MIDI file can be combined with the original vocal track
    (e.g., in a DAW) and produce a piano cover song.

    - If model0 is None, we apply non-autoregressive conversion, i.e., the audio
      is split into 2-bar segments and apply the model independently.
    - If model0 is not None, we apply autoregressive conversion, i.e., the
      current 2-bar segment is arranged based on previous outputs and current
      audio input. Here, model is the "transition operator" and model0 computes
      the "initial probability".
    - When input_audio_path is None, the chord and beat extraction is based on
      the split accompaniment audio. Otherwise, chord and beat extraction is
      applied on the given original audio.
    - Should an audio be converted using multiple models, the chord and beat
      extraction/analysis result can be stored at specific places and used
      again. Specify them in `save_analysis_npy_path` to save the result or
      `input_analysis_npy_path` to load the result.

    :param input_acc_audio_path: input accompaniment audio path
        (i.e., after source separation)
    :param output_midi_path: output MIDI file name.
    :param model: one of the proposed or baseline models.
    :param model0: None or the proposed or baseline models.
    :param input_audio_path: None or the original audio without source
      separation.
    :param input_analysis_npy_path: chord and beat analysis npy path.
    :param save_analysis_npy_path: chord and beat analysis npy path.
    :param batch_size: batch size in model inference.
    :return: None
    """

    # analyze chord and beat
    print('Extracting chords and beats...')

    # Either use split accompaniment for mir analysis or the original audio.
    audio_to_analyze = input_acc_audio_path if input_audio_path is None else \
        input_audio_path

    analysis = analyze_chord_and_beat(audio_to_analyze,
                                      input_analysis_npy_path,
                                      save_analysis_npy_path)

    print('Audio-to-symbolic conversion...')
    # load audio
    audio, sr = librosa.load(input_acc_audio_path)

    # segment data into 2-bar segments and call the audio2midi models.
    predictions, to_notes_func = \
        audio_to_symbolic_prediction(model, model0, analysis, audio,
                                     batch_size)

    # render performance and output a midi file
    print('Writing MIDI and rendering performance...')

    write_prediction(output_midi_path, to_notes_func, analysis,
                     predictions, audio, sr,
                     autoregressive=model0 is None)
