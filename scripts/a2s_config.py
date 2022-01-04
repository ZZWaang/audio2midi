import sys
import os
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                 '..')))
from src.models import Audio2Symb, Audio2SymbNoChord, Audio2SymbSupervised
from src.dataset import create_data_loaders, AudioMidiDataset, \
    AudioMidiDataLoaders
from src.constants import AUG_P
from src.dirs import RESULT_PATH
from src.train import train_model


A2S_CONFIG = {
    'z_chd_dim': 128,
    'z_aud_dim': 192,
    'z_sym_dim': 192
}


NOCHD_CONFIG = {
    'z_aud_dim': 320,
    'z_sym_dim': 192
}


SUPERVISED_CONFIG = {
    'z_dim': 512
}


# Ensuring path is valid when importing from outside the project.
PARAM_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                           '..', 'params'))
TRANSCRIBER_PATH = os.path.join(PARAM_PATH, 'pretrained_onsets_and_frames.pt')


TRAIN_CONFIG = {
    'batch_size': 64,
    'num_workers': 0,
    'meter': 2,
    'n_subdiv': 2,
    'parallel': False,
    'load_data_at_start': False
}

LR = {0: 5e-4, 1: 5e-5, 2: 5e-5, 3: 5e-5}
BETA = {0: 0.01, 1: 0.5, 2: 0.5, 3: 0.5}
CORRUPT = {0: False, 1: True, 2: True, 3: False}
AUTOREGRESSIVE = {0: False, 1: False, 2: False, 3: True}
N_EPOCH = {0: 120, 1: 90, 2: 90, 3: 90}


def prepare_model(model_id, stage=0, model_path=None):
    transcriber_path = TRANSCRIBER_PATH if model_path is None else None
    if model_id == 'a2s':
        model = Audio2Symb.init_model(
            stage=stage,
            z_chd_dim=A2S_CONFIG['z_chd_dim'],
            z_aud_dim=A2S_CONFIG['z_aud_dim'],
            z_sym_dim=A2S_CONFIG['z_sym_dim'],
            transcriber_path=TRANSCRIBER_PATH,
            model_path=model_path
        )
    elif model_id == 'a2s-nochd':
        model = Audio2SymbNoChord.init_model(
            stage=stage,
            z_aud_dim=NOCHD_CONFIG['z_aud_dim'],
            z_sym_dim=NOCHD_CONFIG['z_sym_dim'],
            transcriber_path=TRANSCRIBER_PATH,
            model_path=model_path
        )
    elif model_id == 'supervised':
        model = Audio2SymbSupervised.init_model(
            kl_penalty=False,
            z_dim=SUPERVISED_CONFIG['z_dim'],
            transcriber_path=TRANSCRIBER_PATH,
            model_path=model_path
        )
    elif model_id == 'supervised+kl':
        model = Audio2SymbSupervised.init_model(
            kl_penalty=True,
            z_dim=SUPERVISED_CONFIG['z_dim'],
            transcriber_path=TRANSCRIBER_PATH,
            model_path=model_path
        )
    else:
        raise NotImplementedError
    return model


def prepare_data_loaders(test_mode, stage):
    if test_mode:
        tv_song_ids = ([1, 2, 3], [4])
        train_set, valid_set = \
            AudioMidiDataset.load_with_train_valid_ids(tv_song_ids,
                                                       corrupt=CORRUPT[stage])
        batch_size = 16
        aug_p = AUG_P / AUG_P.sum()
        return AudioMidiDataLoaders.get_loaders(
            batch_size, batch_size, train_set, valid_set, True, False,
            aug_p, TRAIN_CONFIG['num_workers']
        )

    return create_data_loaders(
        batch_size=TRAIN_CONFIG['batch_size'],
        num_workers=TRAIN_CONFIG['num_workers'],
        meter=TRAIN_CONFIG['meter'],
        n_subdiv=TRAIN_CONFIG['n_subdiv'],
        corrupt=CORRUPT[stage],
        autoregressive=AUTOREGRESSIVE[stage]
        )


def result_path_folder_path(model_id, stage=0):
    if model_id in ['a2s', 'a2s-nochd']:
        folder_name = '-'.join([model_id, str(stage)])
    else:
        folder_name = model_id
    folder_path = os.path.join(RESULT_PATH, folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


class TrainingCall:

    def __init__(self, model_id, stage=0):
        assert model_id in ['a2s', 'a2s-nochd', 'supervised', 'supervised+kl']

        self.model_id = model_id
        self.stage = stage
        self.result_path = result_path_folder_path(model_id, stage)

    def __call__(self, test_mode, model_path, run_epochs, readme_fn):
        model = prepare_model(self.model_id, self.stage, model_path)
        data_loaders = prepare_data_loaders(test_mode, self.stage)
        train_model(
            model=model,
            data_loaders=data_loaders,
            stage=self.stage,
            readme_fn=readme_fn,
            n_epoch=N_EPOCH[self.stage],
            parallel=TRAIN_CONFIG['parallel'],
            lr=LR[self.stage],
            writer_names=model.writer_names,
            load_data_at_start=TRAIN_CONFIG['load_data_at_start'],
            beta=BETA[self.stage],
            run_epochs=run_epochs,
            result_path=self.result_path
        )
