'''
@package tfignite.dataset
Dataset defines a unified interface to wrap tf.data.Dataset.
'''
import math

import tensorflow as tf


def calc_len(self):
    return self.num_examples // self.batch_size


tf.data.Dataset.__len__ = calc_len


class Dataset(tf.data.Dataset):
    @classmethod
    def add_parser(cls, parser):
        """ Add dataset specific arguments to command line arguments.
        @param parser: `argparse.ArgumentParser`. The argument parser.

        This function is used by `tfignite.parser.ArgumentParser`. It frees
        users from adding dataset-specific arguments to
        `argparse.ArgumentParser` in the training/evaluation script.
        """
        pass

    @classmethod
    def build(cls, args, is_training, batch_size=128, shuffle=True, **kwargs):
        """ Subclass inheriting `Dataset` should implement this function to
        return a `tf.data.Datset`.
        @param args: argparse.Namespace. The argument to the dataset.
        @param is_training: boolean. Indicating whether to load training/tesing
                            data split.
        @param batch_size: int. The batch size.
        @param shuffle: boolean. Whether to shuffle the dataset.
        @param kwargs: dict. Optional arguments for specific dataset.
        @return tuple of (`tf.data.Dataset`, dict): The dict serves as
                meta-information for model to add additional arguments, for
                e.g. the image height and width.
        """
        raise NotImplementedError

    @classmethod
    def create(cls,
               args,
               batch_size=32,
               is_training=True,
               prefetch_batch=1,
               shuffle=True,
               **kwargs):
        """ The unified interface for different dataset.
        This function delegates the real construction of the dataset to
        `Dataset.build`, while keeping some common behaviors such as
        prefetching. Also, we require the meta-information returned by
        `Dataset.build` contains `num_examples` so that we can calculate number
        of batches.
        @param args: argparse.Namespace. The argument to the dataset.
        @param batch_size: int. The batch size.
        @param is_training: boolean. Indicating whether to load training/tesing
                            data split.
        @param prefetch_batch: int. Number of batch to be prefectched.
        @param shuffle: boolean. Whether to shuffle the dataset.
        @param kwargs: dict. Optional arguments for specific dataset.
        @return tuple of (`tf.data.Dataset`, dict): The dict serves as
                meta-information for model to add additional arguments, for
                e.g. the image height and width.
        """

        dataset, meta_info = cls.build(args,
                                       is_training=is_training,
                                       batch_size=batch_size,
                                       shuffle=shuffle)
        dataset = dataset.prefetch(math.floor(prefetch_batch * batch_size))

        assert 'num_examples' in meta_info,\
            "Need number of examples to calculate lengh of dataset"
        meta_info['batch_size'] = batch_size

        dataset.__dict__.update(meta_info)

        return dataset, meta_info
