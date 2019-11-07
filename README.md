# Tfignite

Tfignite is a project that stems from [ignite](https://github.com/pytorch/ignite). Ignite is a high-level library to help with training neural networks in PyTorch, while Tfignite bares similar design / api and is designed dedicated to tensorflow 2.0.

For the full API reference, read the [online documentation](https://ipod825.github.io/tfignite/docs/html/index.html). For example usage, check the [examples](https://github.com/ipod825/tfignite/tree/master/examples) directory.

# Why tfignite?
Tfignite separates training/evaluation loop from model/dataset computation graph. This makes a single training/evaluation script highly portable to different project and developers only need to focus on how to build the model and dataset for their tasks. The difference from [Keras](https://keras.io) is that the training/evaluation loop is not part of Model's APIs; instead, developers define model forward pass function, which is then injected into the loop defined by an `Engine`. Users can also register event handlers in different phases of a training/evaluation loop (for e.g. `ITERATION_STARTED`).

Apart from the aforementioned separation of model forward pass function and boilerplate loop in `Engine` ([ignite](https://github.com/pytorch/ignite) has full credits for this). Tfignite further reduce boilerplate code by defining the `Model`, `Dataset`, `Callback`, `ArgumentParser` interfaces:
1. `Model`: Defines the `create_trainer` and the `create_evaluator` function, both of which injects a forward pass function to an `Engine` and returns it to the training/evaluation script.
2. `Dataset`: Defines an unified interface `Dataset.create` wrapping over `tf.data.Dataset`.
3. `Callback`: Defines a interface to group related `Engine` event handlers in different phases. For example, `Checkpointer` loads the checkpoint at the beginning of training and stores the checkpoint on `EPOCH_COMPLETED`.
4. `ArgumentParser`: Inherited from `argparse.ArgumentParser`, the parser pass itself to the Model and Dataset classes for parsing Model-specific and Dataset-specific arguments. This further separates the Model development and the training/evaluation script.




# Installation

~~~{.bash}
pip install tfignite
~~~
