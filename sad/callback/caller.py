#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from abc import ABC, abstractmethod
from typing import Dict, List

from sad.generator import GeneratorBase
from sad.model import ModelBase
from sad.task.base import TaskBase


class CallerProtocol(ABC):
    """A caller protocol that defines a set of interfaces that will be used to interact
    with instances of ``sad.callback.CallbackBase``. Currently
    ``sad.trainer.TrainerBase`` is respecting this protocol.
    """

    @property
    @abstractmethod
    def config(self) -> Dict:
        """Configuration dictionary that is used to initialize instances of classes
        compliant with ``CallerProtocal``."""

    @property
    @abstractmethod
    def spec(self) -> Dict:
        """A reference to ``"spec"`` field in ``self.config``."""

    @property
    @abstractmethod
    def n_iters(self) -> int:
        """An integer suggests how many iterations the caller will perform."""

    @property
    @abstractmethod
    def stop(self) -> bool:
        """A flag to indicate caller if early stop is needed."""

    @property
    @abstractmethod
    def model(self) -> ModelBase:
        """An instance of ``sad.model.ModelBase``. A reference to such an instance
        that will be trained by the caller."""

    @property
    @abstractmethod
    def generator(self) -> GeneratorBase:
        """An instance of ``sad.model.GeneratorBase``. A reference to such an instance
        that will be used to generate data to train ``self.model``."""

    @property
    @abstractmethod
    def task(self) -> TaskBase:
        """An instance of ``sad.task.TaskBase``. It is a reference to a task instance
        in which current caller is initialized."""

    @property
    def callbacks(self) -> List["sad.callback.CallbackBase"]:
        """A list of callback instances this caller owns."""
        return self._callbacks

    @callbacks.setter
    def callbacks(self, callbacks: List["sad.callback.CallbackBase"]):
        self._callbacks = callbacks

    def initialize_callback(self):
        """Initialize callbacks. Callback configurations are supplied under
        ``trainer:spec:callbacks`` field in ``self.config``. ``self.spec`` holds a
        reference to ``self.config["spec"]``.

        Initialized instances of ``sad.callback.CallbackBase`` will be pushed to
        ``self.callbacks``, with the same order as their appear in configuration
        ``caller:spec:callbacks``."""
        self.callbacks = []
        callback_configs = (
            [] if not self.spec.get("callbacks") else self.spec.get("callbacks")
        )

        from .base import CallbackFactory

        for callback_config in callback_configs:
            CallbackFactory.produce(callback_config, self)

    def register_callback(self, callback: "CallbackBase"):
        """Callback registration. The actual place where a callback instance is pushed
        to ``self.callbacks`` list. This function will be called when a callback
        instance is initialized - newly created callback instances will register
        themselves to their caller.

        Args:
            callback (:obj:`CallbackBase`): An instance of
                ``sad.callback.CallbackBase``. It is at the initialization of
                ``callback`` argument when this method is called.

        """
        self.callbacks.append(callback)

    def on_loop_begin(self, **kwargs):
        """Will be called when main loop begins."""
        for callback in self.callbacks:
            callback.on_loop_begin(**kwargs)

    def on_loop_end(self, **kwargs):
        """Will be called when main loop finishes."""
        for callback in self.callbacks:
            callback.on_loop_end(**kwargs)

    def on_iter_begin(self, iter_idx: int, **kwargs):
        """Will be called when an iteration begins. An iteration could be an epoch
        during training.

        Args:
            iter_idx (:obj:`int`): The index of iteration, 0-based.

        """
        for callback in self.callbacks:
            callback.on_iter_begin(iter_idx, **kwargs)

    def on_iter_end(self, iter_idx: int, **kwargs):
        """Will be called when an iteration finishes.

        Args:
            iter_idx (:obj:`int`): The index of an iteration. 0-based.

        """
        for callback in self.callbacks:
            callback.on_iter_end(iter_idx, **kwargs)

    def on_step_begin(self, iter_idx: int, step_idx: int, **kwargs):
        """Will be called when step begins. A step could be a gradient update from
        a minibatch during training loop.

        Args:
            iter_idx (:obj:`int`): The index of iteration. 0-based.
            step_idx (:obj:`int`): The index of step. 0-based.

        """
        for callback in self.callbacks:
            callback.on_step_begin(iter_idx, step_idx, **kwargs)

    def on_step_end(self, iter_idx: int, step_idx: int, **kwargs):
        """Will be called when a step finishes.

        Args:
            iter_idx (:obj:`int`): The index of iteration. 0-based.
            step_idx (:obj:`int`): The index of step. 0-based.

        """
        for callback in self.callbacks:
            callback.on_step_end(iter_idx, step_idx, **kwargs)
