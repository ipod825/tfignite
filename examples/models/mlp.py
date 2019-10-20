from functools import reduce

import tensorflow as tf
from tensorflow.keras import layers, metrics, optimizers

from tfignite import Engine, Events, Model


class MLPModel(Model):
    def __init__(self, args, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.args = args

        self.nets = []

        for unit in eval(self.args.hidden_units):
            self.nets.append(layers.Dense(unit, activation='relu'))
        self.nets.append(layers.Dense(args.num_class))

        self.optimizer = optimizers.SGD(lr=args.lr)

    @classmethod
    def add_additional_args(cls, args, meta_info):
        args.img_shape = meta_info['img_shape']
        args.num_class = meta_info['num_classes']

    @classmethod
    def add_parser(cls, parser):
        parser.add_argument('--lr',
                            type=float,
                            default=0.001,
                            help='learning rate')
        parser.add_argument('--hidden_units',
                            type=str,
                            default="[256,256]",
                            help="MLP's hidden units")
        parser.add_argument(
            '--summary_interval',
            type=int,
            default=1,
            help='how many batches to wait before logging scalar during'
            'training')

    def create_trainer(self):
        @tf.function
        def update(engine, batch):
            x, y = batch
            with tf.GradientTape(persistent=True) as tape:
                out = tf.reshape(
                    x, (-1, reduce((lambda a, b: a * b), self.args.img_shape)))

                for net in self.nets:
                    out = net(out)

                y_onehot = tf.one_hot(y, depth=10)
                loss = tf.square(out - y_onehot)
                loss = tf.reduce_sum(loss) / 32

            grads = tape.gradient(loss, self.trainable_variables)
            del tape
            self.optimizer.apply_gradients(zip(grads,
                                               self.trainable_variables))

            if engine.iteration % self.args.summary_interval == 0:
                tf.summary.scalar('training/loss', loss, engine.iteration)

        trainer = Engine(update)
        return trainer

    def create_evaluator(self):
        acc = metrics.Accuracy()

        @tf.function
        def evaluate(engine, batch):
            x, y = batch
            out = tf.reshape(
                x, (-1, reduce((lambda a, b: a * b), self.args.img_shape)))
            for net in self.nets:
                out = net(out)

            acc(tf.argmax(out, axis=1), y)

        evaluator = Engine(evaluate)

        @evaluator.on(Events.EPOCH_COMPLETED)
        # @tf.function
        def summary_per_epoch(engine):
            tf.summary.scalar('testing/acc', acc.result(), engine.iteration)
            print(f'Accuracy: {acc.result()}')
            acc.reset_states()

        return evaluator
