import tensorflow as tf
import tensorflow_datasets as tfds

from tfignite import Dataset


class MNIST(Dataset):
    @classmethod
    def build(cls, args, train, batch_size=128, shuffle=True, **kwargs):

        if train:
            dataset = tfds.load(name="mnist", split=tfds.Split.TRAIN)
            num_examples = 55000
        else:
            dataset = tfds.load(name="mnist", split=tfds.Split.TEST)
            num_examples = 5000

        if shuffle:
            dataset = dataset.shuffle(int(num_examples / 100))

        dataset = dataset.map(
            lambda d: (tf.cast(d['image'], tf.float32) / 255., d['label']))

        dataset = dataset.batch(batch_size)

        return dataset, dict(img_shape=[1, 28, 28],
                             num_classes=10,
                             num_examples=num_examples)
