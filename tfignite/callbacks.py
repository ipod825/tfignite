'''
@package tfignite.callbacks
Built-in callbacks.
'''

from pathlib import Path

from . import util
from .engine import Events


class Callback(object):
    """ Base class of all callbacks. A Callback adds to an
    `tfignite.engine.Engine` a set of related `tfignite.engine.Events`
    handlers.  `tfignite.callbacks` implements some common Callbacks such as
    progress bar, checkpointer, each of which is composed of several event
    handlers that are usually boilplate code. Therefore, instead of
    implementing these event handlers and add them to the engine, users can
    simply use `tfignite.engine.Engine.add_callbacks` with a list of existing
    """
    def register(self, engine):
        """ Subclass must implement this function.
        @param engine: `tfignite.engine.Engine`. The engine to adds the event
                       handles to.
        """
        raise NotImplementedError


class TqdmProgressBar(Callback):
    """ Progress bar powered by tqdm
    """
    def __init__(self, epochs, iterations):
        """
        @param epochs: int. Total number of epochs.
        @param iterations: int. Total number of iterations.

        """
        self.epochs = epochs
        self.iterations = iterations

    def register(self, engine):
        from tqdm import tqdm
        pbar = tqdm(initial=engine.epoch_val, total=self.epochs, desc="Epoch")
        qbar = tqdm(initial=0, total=self.iterations, desc="Iter")

        @engine.on(Events.EPOCH_COMPLETED)
        def print_per_epoc(engine):
            nonlocal qbar
            pbar.update(1)
            qbar.close()
            qbar = tqdm(initial=0,
                        leave=False,
                        total=self.iterations,
                        desc="iteration")

        @engine.on(Events.ITERATION_COMPLETED)
        def print_per_iteration(engine):
            qbar.update(1)

        @engine.on(Events.COMPLETED)
        def complete(engine):
            qbar.close()
            pbar.close()


class Checkpointer(Callback):
    """ A callback that saves checkpoint during training and restore the model
    from the checkpoint during evaluation.
    """
    def __init__(
            self,
            ckpt_dir,
            objs_to_save={},
            save_interval=1,
            max_to_keep=None,
            is_training=True,
    ):
        """
        @param ckpt_dir: str. The directory to store checkpoint information.
        @param objs_to_save: dict. The objects to write to checkpoint. Usually
                             users just need to pass model.gen_ckpt_objs if
                             `tfignite.model.Model` is used.
        @param save_interval: int. The number of epochs to wait before saving
                              to checkpoints.
        @param max_to_keep: int. Maximum number of checkpoint to save. If not
                            specified, all checkpoints are preserved.
        @param is_training: boolean. Indicating whether the current engine is
                            in traing or evaluation mode. In both mode,
                            CheckPointer try to load the latest checkpoint from
                            `ckpt_dir`. In addition, in training mode,
                            CheckPointer stores checkpoints into `ckpt_dir`.
        """

        self._ckpt_dir = ckpt_dir
        self._objs_to_save = objs_to_save
        self._save_interval = save_interval
        self._max_to_keep = max_to_keep
        self._is_training = is_training

    def register(self, engine):
        assert len(engine._callbacks) == 0, (
            "CheckPointer must be the first registered callback as it"
            "alter the engine's state when registering.")

        import tensorflow as tf

        if self._is_training:
            self._objs_to_save['epoch'] = engine.epoch
            self._objs_to_save['iteration'] = engine.iteration

        ckpt = tf.train.Checkpoint(**self._objs_to_save)
        manager = tf.train.CheckpointManager(ckpt,
                                             f'{self._ckpt_dir}',
                                             max_to_keep=self._max_to_keep)

        if self._is_training:

            @engine.on(Events.EPOCH_COMPLETED)
            def save_per_epoch(engine):
                if tf.equal(engine.epoch % self._save_interval, 0):
                    manager.save()

            # if in last epoch we did not save the the model in save_per_epoch,
            # we save it here
            @engine.on(Events.COMPLETED)
            def save_completed(engine):
                if tf.not_equal(engine.epoch % self._save_interval, 0):
                    manager.save()

        if manager.latest_checkpoint:
            if self._is_training:
                print(f'Resume training using {manager.latest_checkpoint}')
                # This might modify engine's epoch tensor.
                ckpt.restore(manager.latest_checkpoint)
            else:
                print(f'Load model: {manager.latest_checkpoint}')
                ckpt.restore(manager.latest_checkpoint).expect_partial()


class ModelArgsSaverLoader(Callback):
    """ A callback that saves / loads the model arguments from the command
    line.

    For first time training, we saves the model-specif arguments from the
    command line (defined by the model's add_parser function) into a file. For
    resumed training / evaluation, we check and ask user to confirm if saved
    arguments are different than those given in the command line.
    """
    def __init__(self, model, is_training, save_dir):
        assert hasattr(model, 'args'), (
            "model must has the attribute 'args' to use ModelArgsSaver")

        self._model = model
        self._is_training = is_training
        self._save_dir = Path(save_dir).expanduser()
        self._save_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_args(self):
        import argparse
        model_parser = argparse.ArgumentParser()
        self._model.__class__.add_parser(model_parser)
        model_arg_names = vars(model_parser.parse_known_args()[0]).keys()

        model_args = {}
        for name in model_arg_names:
            model_args[name] = self._model.args.__dict__[name]
        return model_args

    def register(self, engine):
        import json

        model_args = self._get_model_args()
        model_args_file = self._save_dir / 'model_args.json'

        if self._is_training:
            if model_args_file.is_file():
                saved_args = json.loads(model_args_file.read_text())
                assert len(model_args) == len(saved_args), (
                    "Number of model arguments mismatch. "
                    "Saved:\n {}\n Arguments:\n {}\n".format(
                        list(saved_args.keys()), list(model_args.keys())))

                for key, value in saved_args.items():
                    if model_args[key] != value:
                        util.stop_if_no(
                            "Previous run of this model version use different"
                            " arguments. Use the saved argument and proceed to"
                            " train anyway? By pressing ycheckpoint might be"
                            " loaded. By pressing n you might want to delete "
                            " the old model argsfile.\nPrevious saved "
                            " arguments:\n{}\n Current Arguments:\n{}".format(
                                saved_args, model_args))
                        break
            elif len(model_args) != 0:
                model_args_file.write_text(json.dumps(model_args))
        else:
            if model_args_file.is_file():
                saved_args = json.loads(model_args_file.read_text())

                for key, value in saved_args.items():
                    if model_args[key] != value:
                        util.stop_if_no(
                            f"You specify\n{key} = {model_args[key]} which "
                            f"is different from the saved value: {value}.\nBy "
                            f"pressing y {key} = {model_args[key]} will be "
                            f"used. \nBy pressing n you might want to delete"
                            f"  {key} from you command line.")
