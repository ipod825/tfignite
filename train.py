import tensorflow as tf
from tensorflow.keras import Sequential, datasets, layers, metrics, optimizers

from ignite.engine import Engine, Events
from ignite.tfmetrics import Accuracy

(xs, ys), _ = datasets.mnist.load_data()
print('datasets:', xs.shape, ys.shape, xs.min(), xs.max())

xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255.
db = tf.data.Dataset.from_tensor_slices((xs, ys))
db = db.batch(128).repeat(1)

network = Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)
])
network.build(input_shape=(None, 28 * 28))

optimizer = optimizers.SGD(lr=0.01)


@tf.function
def update(engine, batch):
    x, y = batch
    with tf.GradientTape() as tape:
        x = tf.reshape(x, (-1, 28 * 28))
        out = network(x)
        y_onehot = tf.one_hot(y, depth=10)
        loss = tf.square(out - y_onehot)
        loss = tf.reduce_sum(loss) / 32

    grads = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))


@tf.function
def evaluate(engine, batch):
    x, y = batch
    x = tf.reshape(x, (-1, 28 * 28))
    out = network(x)
    return {'y_pred': out, 'y_true': y}


trainer = Engine(update)
evaluator = Engine(evaluate)
acc = Accuracy(lambda e: (e['y_pred'], e['y_true']))
acc.attach(evaluator, "accuracy")


@evaluator.on(Events.EPOCH_COMPLETED)
def calc_acc(engine):
    print(engine.state.metrics['accuracy'])


trainer.run(db, max_epochs=1)
evaluator.run(db)
