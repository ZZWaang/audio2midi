from a2s_config import TrainingCall

# If test_mode is True, will load a mini dataset to debug the code.
test_mode = False

# model_id in ['a2s', 'a2s-nochd', 'supervised', 'supervised+kl']
model_id = 'supervised'

# stage in [0, 1, 2, 3], ignored in 'supervised' or 'supervised+kl']
stage = 0

# A pre-trained a2s model path. Used when stage > 0.
model_path = None

# In case the training is terminated, checkpoint saving is not implemented in
# amc_dl. Manually enter the number of epochs trained so far. Default None.
run_epochs = None

if __name__ == '__main__':
    training = TrainingCall(model_id=model_id, stage=stage)
    training(test_mode=test_mode, model_path=model_path,
             run_epochs=run_epochs, readme_fn=__file__)
