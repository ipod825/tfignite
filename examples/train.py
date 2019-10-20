import os

import tensorflow as tf

from datasets.mnist import MNIST as Dataset
from models.mlp import MLPModel as Model
from tfignite import ArgumentParser, callbacks


def parse_args():
    parser = ArgumentParser(Model, Dataset, description='')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        metavar='N',
                        help='input batch size for training')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--root_dir',
                        default='/tmp/mnist_example',
                        help='Directory where to store checkpointslogs.')

    return parser.parse_args()


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    args = parse_args()

    dataset, meta_info = Dataset.create(args, batch_size=args.batch_size)

    Model.add_additional_args(args, meta_info)
    model = Model(args)

    trainer = model.create_trainer()

    trainer.add_callbacks([
        callbacks.Checkpointer(args.root_dir + '/ckpt',
                               model.gen_ckpt_objs(),
                               save_interval=1,
                               max_to_keep=10),
        callbacks.ModelArgsSaverLoader(model, True, args.root_dir),
        callbacks.TqdmProgressBar(args.epochs, len(dataset))
    ])

    train_summary_writer = tf.summary.create_file_writer(args.root_dir +
                                                         '/log')
    with train_summary_writer.as_default():
        trainer.run(dataset, args.epochs)
