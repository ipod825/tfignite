import math

import tensorflow as tf


def calc_len(self):
    return self.num_examples // self.batch_size


tf.data.Dataset.__len__ = calc_len


class Dataset(tf.data.Dataset):
    @classmethod
    def add_parser(cls, parser):
        pass

    @classmethod
    def build(cls, args, train, batch_size=128, shuffle=True, **kwargs):
        raise NotImplementedError

    @classmethod
    def create(cls,
               args,
               batch_size=32,
               train=True,
               prefetch_batch=1,
               shuffle=True,
               cache_in_memory=True,
               **kwargs):

        dataset, meta_info = cls.build(args,
                                       train=train,
                                       batch_size=batch_size,
                                       shuffle=shuffle)
        dataset = dataset.prefetch(math.floor(prefetch_batch * batch_size))

        assert 'num_examples' in meta_info,\
            "Need number of examples to calculate lengh of dataset"
        meta_info['batch_size'] = batch_size

        dataset.__dict__.update(meta_info)

        return dataset, meta_info
