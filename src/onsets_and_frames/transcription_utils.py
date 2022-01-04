import torch
from .transcriber import OnsetsAndFrames


def load_transcription_model(model_path, device=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if device is None else device
    model = torch.load(model_path, map_location=device)
    return model


def load_init_transcription_model(device=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if device is None else device
    return OnsetsAndFrames(229, 108 - 21 + 1, 48).to(device)
