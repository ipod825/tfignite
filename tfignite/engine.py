'''
@package tfignite.engine
Training / evaluation loop boilplate code.
'''
import inspect
import logging
from collections import defaultdict
from enum import Enum

import tensorflow as tf


class Events(Enum):
    """Events that are fired by the `tfignite.engine.Engine`
    during execution.
    """
    EPOCH_STARTED = "epoch_started"
    EPOCH_COMPLETED = "epoch_completed"
    STARTED = "started"
    COMPLETED = "completed"
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"
    EXCEPTION_RAISED = "exception_raised"


class State(object):
    """An object that is used to pass internal and user-defined state between
    event handlers.
    """
    event_to_attr = {
        Events.ITERATION_STARTED: "iteration",
        Events.ITERATION_COMPLETED: "iteration",
        Events.EPOCH_STARTED: "epoch",
        Events.EPOCH_COMPLETED: "epoch",
        Events.STARTED: "epoch",
        Events.COMPLETED: "epoch"
    }

    def __init__(self, epoch, iteration, **kwargs):
        self.iteration = tf.Variable(iteration, dtype=tf.int64)
        self.epoch = tf.Variable(epoch, dtype=tf.int64)

    def get_event_attrib_value(self, event_name):
        if event_name not in State.event_to_attr:
            raise RuntimeError("Unknown event name '{}'".format(event_name))
        return getattr(self, State.event_to_attr[event_name])


class Engine(object):
    """Runs a given `process_function` over each batch of a dataset, emitting
    events as it goes.

    @note It is more efficient to process the logging (e.g. `tf.summary` inside
    the `process_function` then passing it back to the training script), as
    tensorflow might need to copy tensors from gpu back to cpu. So it is
    recommended not to return any output for further processing.

    Also, instead of writing just a function, users can consider inherit
    and implement a `tfignite.model.Model`.
    """
    def __init__(self, process_function):
        """
        @param process_function: callable. A function receiving a handle to the
                                 engine and the current batch in each
                                 iteration, and returns arbitray output.
        """
        self._event_handlers = defaultdict(list)
        self._callbacks = []
        self._logger = logging.getLogger(__name__ + "." +
                                         self.__class__.__name__)
        self._logger.addHandler(logging.NullHandler())
        self._process_function = process_function
        self.should_terminate = False
        self.should_terminate_single_epoch = False
        self._allowed_events = []

        self.state = State(0, 0)

        self.register_events(*Events)

        if self._process_function is None:
            raise ValueError(
                "Engine must be given a processing function in order to run.")

        self._check_signature(process_function, 'process_function', None)

    @property
    def epoch_val(self):
        """ Return the python interger value of the engine's epoch tensor.

        Usually used in callbacks.
        """
        return self.state.epoch.numpy()

    @property
    def iteration_val(self):
        """ Return the python interger value of the engine's iteration tensor.

        Usually used in callbacks.
        """
        return self.state.iteration.numpy()

    @property
    def epoch(self):
        """ Return the engine's epoch tensor.
        Usually used in training/evaluation functions.
        """
        return self.state.epoch

    @property
    def iteration(self):
        """ Return the engine's iteration tensor.
        Usually used in training/evaluation functions.
        """
        return self.state.iteration

    def register_events(self, *event_names):
        """Add events that can be fired.
        @param *event_names: An object (ideally a string or int). Define the
                             name of the event being supported.

        Registering an event will let the user fire these events at any point.
        This opens the door to make the `Engine.run` loop even more
        configurable.

        By default, the events from `Events` are registerd.
        """
        for name in event_names:
            self._allowed_events.append(name)

    def add_event_handler(self, event_name, handler, *args, **kwargs):
        """Add an event handler to be executed when the specified event is
        fired.
        @param event_name: An event to attach the handler to. Valid events are
                           from `Events` or any event_name added by
                           `Engine.register_events`.
        @param handler: callable. The callable event handler that should be
                        invoked.
        @param *args: optional args to be passed to handler.
        @param **kwargs: optional keyword args to be passed to handler.

        @note The handler function's first argument will be self, the `Engine`
              object it was bound to.
        @note Other arguments can be passed to the handler in addition to the
              `*args` and  `**kwargs` passed here, for example during
              `Events.EXCEPTION_RAISED`.
        """
        if event_name not in self._allowed_events:
            self._logger.error(
                "attempt to add event handler to an invalid event %s.",
                event_name)
            raise ValueError(
                "Event {} is not a valid event for this Engine.".format(
                    event_name))

        event_args = (
            Exception(), ) if event_name == Events.EXCEPTION_RAISED else ()
        self._check_signature(handler, 'handler', *(event_args + args),
                              **kwargs)

        self._event_handlers[event_name].append((handler, args, kwargs))
        self._logger.debug("added handler for event %s.", event_name)

    def add_callbacks(self, callbacks):
        """ Add `tfignite.callbacks.Callback` to be exeuted.
        @param callbacks: list of `tfignite.callbacks.Callback`. Callbacks to
                          be executed.
        """
        for cbk in callbacks:
            cbk.register(self)
            self._callbacks.append(cbk)

    def has_event_handler(self, handler, event_name=None):
        """Check if the specified event has the specified handler.
        @param handler: callable. the callable event handler.
        @param event_name: The event the handler attached to. Set this to
                           `None` to search all events.
        """
        if event_name is not None:
            if event_name not in self._event_handlers:
                return False
            events = [event_name]
        else:
            events = self._event_handlers
        for e in events:
            for h, _, _ in self._event_handlers[e]:
                if h == handler:
                    return True
        return False

    def remove_event_handler(self, handler, event_name):
        """Remove event handler handler from registered handlers of the
        engine.
        @param handler: callable. The callable event handler that should be
                        removed.
        @param event_name: The event the handler attached to.
        """
        if event_name not in self._event_handlers:
            raise ValueError(
                "Input event name '{}' does not exist".format(event_name))

        new_event_handlers = [
            (h, args, kwargs)
            for h, args, kwargs in self._event_handlers[event_name]
            if h != handler
        ]
        if len(new_event_handlers) == len(self._event_handlers[event_name]):
            raise ValueError("Input handler '{}' is not found among \
                registered event handlers".format(handler))
        self._event_handlers[event_name] = new_event_handlers

    def _check_signature(self, fn, fn_description, *args, **kwargs):
        exception_msg = None

        signature = inspect.signature(fn)
        try:
            signature.bind(self, *args, **kwargs)
        except TypeError as exc:
            fn_params = list(signature.parameters)
            exception_msg = str(exc)

        if exception_msg:
            passed_params = [self] + list(args) + list(kwargs)
            raise ValueError("Error adding {} '{}': "
                             "takes parameters {} but will be called with {} "
                             "({}).".format(fn, fn_description, fn_params,
                                            passed_params, exception_msg))

    def on(self, event_name, *args, **kwargs):
        """Decorator shortcut for add_event_handler.
        @param event_name: An event to attach the handler to. Valid events are
                           from `Events` or any event_name added by
                           `Engine.register_events`.
        @param *args: optional args to be passed to the event handler.
        @param **kwargs: optional keyword args to be passed to the handler.
        """
        def decorator(f):
            self.add_event_handler(event_name, f, *args, **kwargs)
            return f

        return decorator

    def _fire_event(self, event_name, *event_args, **event_kwargs):
        """Execute all the handlers associated with given event.
        @param event_name: event for which the handlers should be executed.
        @param event_name: Valid events are from `Events` or any event_name
                           added by `Engine.register_events`.
        @param *event_args: optional args to be passed to all handlers.
        @param **event_kwargs: optional keyword args to be passed to all
                               handlers.

        This method executes all handlers associated with the event event_name.
        Optional positional and keyword arguments can be used to pass arguments
        to **all** handlers added with this event. These aguments updates
        arguments passed using `Engine.add_event_handler`.

        """
        if event_name in self._allowed_events:
            self._logger.debug("firing handlers for event %s ", event_name)
            for func, args, kwargs in self._event_handlers[event_name]:
                kwargs.update(event_kwargs)
                func(self, *(event_args + args), **kwargs)

    def fire_event(self, event_name):
        """Execute all the handlers associated with given event.
        @param event_name: event for which the handlers should be executed.
                           Valid events are from `Events` or any event_name
                           added by `Engine.register_events`.

        This method executes all handlers associated with the event event_name.
        This is the method used in `Engine.run` to call the core events found
        in `Events`.

        Custom events can be fired if they have been registered before with
        `Engine.register_events`. The engine state attribute should be used to
        exchange "dynamic" data among `process_function` and handlers.

        This method is called automatically for core events. If no custom
        events are used in the engine, there is no need for the user to call
        the method.

        """
        return self._fire_event(event_name)

    def terminate(self):
        """Sends terminate signal to the engine, so that it terminates
        completely the run after the current iteration.
        """
        self._logger.info("Terminate signaled. Engine will stop after current\
            iteration is finished.")
        self.should_terminate = True

    def terminate_epoch(self):
        """Sends terminate signal to the engine, so that it terminates the
        current epoch after the current iteration.
        """
        self._logger.info(
            "Terminate current epoch is signaled. "
            "Current epoch iteration will stop after current iteration"
            "is finished.")
        self.should_terminate_single_epoch = True

    def _run_once_on_dataset(self, dataset):
        try:
            for batch in dataset:
                self.state.iteration.assign_add(1)
                self._fire_event(Events.ITERATION_STARTED)
                self.output = self._process_function(self, batch)
                self._fire_event(Events.ITERATION_COMPLETED)
                if self.should_terminate or self.should_terminate_single_epoch:
                    self.should_terminate_single_epoch = False
                    break

        except BaseException as e:
            self._logger.error(
                "Current run is terminating due to exception: %s.", str(e))
            self._handle_exception(e)

    def _handle_exception(self, e):
        if Events.EXCEPTION_RAISED in self._event_handlers:
            self._fire_event(Events.EXCEPTION_RAISED, e)
        else:
            raise e

    def run(self, dataset, max_epochs=1, start_epoch=None):
        """Runs the `process_function` over the passed data.
        @param dataset: Iterable. Collection of batches allowing repeated
                        iteration.
        @param max_epochs: int: Max epochs to run for.
        @param start_epoch: The starting epoch (default 0).
        """

        try:
            if start_epoch is not None:
                self.state.epoch.assign(start_epoch)

            self._logger.info(
                "Engine run starting with max_epochs={}.".format(max_epochs))
            self._fire_event(Events.STARTED)
            while self.epoch_val < max_epochs and not self.should_terminate:
                self.state.epoch.assign_add(1)
                self._fire_event(Events.EPOCH_STARTED)
                self._run_once_on_dataset(dataset)
                if self.should_terminate:
                    break
                self._fire_event(Events.EPOCH_COMPLETED)

            self._fire_event(Events.COMPLETED)

        except BaseException as e:
            self._logger.error(
                "Engine run is terminating due to exception: %s.", str(e))
            self._handle_exception(e)
