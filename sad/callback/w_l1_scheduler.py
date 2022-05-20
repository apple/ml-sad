#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import logging

import numpy as np

from .base import CallbackBase, CallbackFactory

logger = logging.getLogger("callback.w_l1_scheduler")


def exp_rise(w_l1: float, rate: float) -> float:
    """A scheduling function to calculate new weight of L1 regularization with
    exponential rise.

    Args:
        w_l1 (:obj:`float`): Current weight of L1 regularization.
        rate (:obj:`float`): The rate of rise. When activated, ``w_l1`` will be
            changed by multiplying ``exp(rate)``.

    Returns:
        :obj:`float`: Updated weight of L1 regularization.
    """
    new_w_l1 = w_l1 * np.exp(rate)
    logger.info(f"w_l1 updated {w_l1:.02e} -> {new_w_l1:.02e} " "by exponenetial rise.")

    return new_w_l1


def step(w_l1: float, new_w_l1: float) -> float:
    """A scheduling function to update learning rate..

    Args:
        w_l1 (:obj:`float`): Current weight of L1 regularization.
        new_w_l1 (:obj:`float`): New weight of L1 regularization.

    Returns:
        :obj:`float`: Updated weight of L1 regularization.
    """
    if w_l1 == new_w_l1:
        return w_l1

    logger.info(f"w_l1 updated {w_l1:.02e} -> {new_w_l1:.02e} by step scheme.")
    w_l1 = new_w_l1

    return w_l1


@CallbackFactory.register
class WeightL1SchedulerCallback(CallbackBase):
    """A callback class that is responsible to update weight of L1 regularization during
    training.

    Instance of this class will be managed by instances compliant with
    ``sad.caller.CallerProtocol`` instances, during caller's' initialization.
    Configurations for this callback is provided under
    ``trainer:spec:callbacks:``. An example is shown below::

        trainer:
          name: SGDTrainer
          spec:
            n_iters: 20
            w_l1: 0.1
            w_l2: 0.0
            u_idxs: [0, 1, 2, 3]
            callbacks:
            - name: "WeightL1SchedulerCallback"
              spec:
                scheme: "exp_rise"
                rate: -0.1
                start: 0.5

    """

    @property
    def scheme(self) -> str:
        """The scheme of how weight of L1 regularization will be changed. Currently can
        take ``"exp_rise"|"step"``. Will read directly from ``"scheme"`` field from
        ``self.spec``.
        """
        return self.spec.get("scheme", "exp_rise")

    @property
    def start(self) -> int:
        """A positive number suggesting when to start to apply changes to weight of L1
        regularization. When ``start < 1``, it will be treated as a proportion,
        suggesting ``w_l1`` will subject to change when
        ``iter_idx >= int(n_iters * start)``. Otherwise, ``iter_idx >= int(start)`` will
        be the condition.
        """
        start = self.spec.get("start", 0)
        if start > 0 and start < 1:  # assume it is a proportion
            start = int(self.caller.n_iters * start)
        return int(start)

    @property
    def every(self) -> int:
        """Number of iterations every update is performed. ``1`` means weight of L1
        regularization is subject to change for every iteration. Will read directly from
        ``"every"`` field in ``self.spec``.
        """
        every = self.spec.get("every", 1)
        return every

    @property
    def rate(self) -> float:
        """The rate of rise. Effective when ``self.scheme`` is set to ``"exp_rise"``.
        When activated,  weight of L1 regularization will be changed by multiplying its
        value by ``exp(rate)``. Will read directly from ``"rate"`` field in
        ``self.spec``.
        """
        return self.spec.get("rate", 0)

    @property
    def new_w_l1(self) -> float:
        """The new weight of L1 regularization. Effective when ``self.scheme`` is set to
        ``"step"``. When activated, ``w_l1`` will be changed to ``self.new_w_l1``. Will
        read directly from ``"new_w_l1"`` field under ``self.spec``.
        """
        return self.spec.get("new_w_l1", self.caller.w_l1)

    def on_loop_begin(self, **kwargs):
        """Not applicable to this class."""
        pass

    def on_loop_end(self, **kwargs):
        """Not applicable to this class."""
        pass

    def on_iter_begin(self, iter_idx: int, **kwargs):
        """Will be called to determine whether to attempt to update the weight of L1
        regulation when an iteration begins.

        Args:
            iter_idx (:obj:`int`): The index of iteration, 0-based.
        """
        start = self.start
        every = self.every
        caller = self.caller
        if (iter_idx >= start) and (iter_idx % every == 0):
            if self.scheme == "exp_rise":
                new_w_l1 = exp_rise(caller.w_l1, self.rate)
            elif self.scheme == "step":
                new_w_l1 = step(caller.w_l1, self.new_w_l1)
            else:
                new_w_l1 = caller.w_l1
            caller.w_l1 = new_w_l1

    def on_iter_end(self, iter_idx: int, **kwargs):
        """Not applicable to this class."""
        pass

    def on_step_begin(self, iter_idx: int, step_idx: int, **kwargs):
        """Not applicable to this class."""
        pass

    def on_step_end(self, iter_idx: int, step_idx: int, **kwargs):
        """To be determined."""
        pass

    def save(self, folder: str):
        pass

    def load(self, folder: str):
        pass
