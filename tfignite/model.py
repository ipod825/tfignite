'''
@package tfignite.model
Model defines interface to the training script.
'''
from tensorflow.keras import layers


class Model(layers.Layer):
    def create_trainer(self):
        """ Return an tfignite.engine that runs the training forward pass for a
        single batch.
        """
        raise NotImplementedError

    def create_evaluator(self):
        """ Return an tfignite.engine that runs the evaluation forward pass for
        a single batch.
        """
        raise NotImplementedError

    @classmethod
    def add_additional_args(self, args, meta_info):
        """ Let the model set data related arguments.
        @param args: argparse.Namespace. The argument namespace to be set.
        @param meta_info: dict. Data related meta information that can be used
                          to set args.

        For example, a CNN model might need to define its archtecture based on
        the image height/width which is only know after the dataset is loaded
        in a training/evaluation script. The training script can then use this
        function to ask the model to modify args as needed. Such design frees
        users from modifying model-specific arguments in the
        training/evaluation script.

        """
        pass

    @classmethod
    def add_parser(cls, parser):
        """ Add model specific arguments to command line arguments.
        @param parser: argparse.ArgumentParser. The argument parser.

        This function is used by `tfignite.parser.ArgumentParser`. It frees
        users from adding model-specific arguments to `argparse.ArgumentParser`
        in training/evaluation script.

        """
        pass

    def gen_ckpt_objs(self):
        """ Function to integrate with `tfignite.callbacks.Checkpointer`.
        """
        return dict(model=self)
