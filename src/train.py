from .dirs import *
from .amc_dl.torch_plus import LogPathManager, SummaryWriters, \
    ParameterScheduler, OptimizerScheduler, MinExponentialLR, \
    TeacherForcingScheduler, ConstantScheduler
from .amc_dl.torch_plus.module import TrainingInterface
import torch
from torch import optim
import os
from .utils import beta_annealing, scheduled_sampling


class TrainingVAE(TrainingInterface):

    def _batch_to_inputs(self, batch):
        return batch


def train_model(model, data_loaders, stage, readme_fn, n_epoch, parallel,
                lr=1e-3, writer_names=None, load_data_at_start=False,
                beta=0.1, run_epochs=None, result_path=None):
    """
    :param model: A2S model. Possibly loaded with pre-trained parameters.
    :param data_loaders: dataset.AudioMidiDataLoaders
    :param stage: training stage in range(0, 4).
    :param readme_fn: the fn to copy as log.
    :param n_epoch: total epochs to train.
    :param parallel: pytorch data parallel.
    :param lr: learning rate.
    :param writer_names: tensorboard writers.
    :param load_data_at_start: whether to load all the data before training.
    :param beta: target kl annealing value.
    :param run_epochs: the epoches already endured.
    :param result_path: the output log path.
    :return: None
    """

    def pre_load_dataset(dst):
        for item in range(len(dst)):
            _ = dst[item]

    # constants
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clip = 3

    weights = [1, 0.5]

    tf_rates = [(0.8, 0.3), (0.8, 0.5), (0.5, 0)]

    parallel = parallel if torch.cuda.is_available() and \
        torch.cuda.device_count() > 1 else False

    if load_data_at_start:
        for dataset in [data_loaders.train_set, data_loaders.val_set]:
            pre_load_dataset(dataset)

    result_path = RESULT_PATH if result_path is None else result_path
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    log_path_mng = \
        LogPathManager(readme_fn,
                       log_path_name=os.path.join(result_path, 'result'))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = MinExponentialLR(optimizer, gamma=0.9999, minimum=lr * 1e-2)
    optimizer_scheduler = OptimizerScheduler(optimizer, scheduler, clip)

    tags = {'loss': None}
    summary_writers = SummaryWriters(writer_names, tags,
                                     log_path_mng.writer_path)
    tfr1_scheduler = TeacherForcingScheduler(*tf_rates[0],
                                             f=scheduled_sampling)
    tfr2_scheduler = TeacherForcingScheduler(*tf_rates[1],
                                             f=scheduled_sampling)
    tfr3_scheduler = TeacherForcingScheduler(*tf_rates[2],
                                             f=scheduled_sampling)
    weights_scheduler = ConstantScheduler(weights)

    if stage >= 1:
        beta_scheduler = TeacherForcingScheduler(beta, 0.01, f=beta_annealing)
    else:
        beta_scheduler = TeacherForcingScheduler(beta, 0., f=beta_annealing)

    params_dic = dict(tfr1=tfr1_scheduler, tfr2=tfr2_scheduler,
                      tfr3=tfr3_scheduler,
                      beta=beta_scheduler,
                      weights=weights_scheduler)
    param_scheduler = ParameterScheduler(**params_dic)

    training = TrainingVAE(device, model, parallel, log_path_mng,
                           data_loaders, summary_writers, optimizer_scheduler,
                           param_scheduler, n_epoch)

    # Run through the epochs already endured. Similar to (but still different
    # from) loading specific training checkpoints.
    if run_epochs is not None:
        steps_per_epoch = len(data_loaders.train_loader)
        for _ in range(run_epochs * steps_per_epoch):
            training.opt_scheduler.optimizer_zero_grad()
            training.opt_scheduler.step()
            training.param_scheduler.step()

    training.run()
