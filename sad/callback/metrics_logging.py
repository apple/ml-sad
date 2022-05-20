#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from .base import CallbackBase, CallbackFactory
from .caller import CallerProtocol


@CallbackFactory.register
class MetricsLoggingCallback(CallbackBase):
    """A callback class that is responsible to log metrics during caller's main loop.

    Instance of this class will be managed by caller instances that is compliant with
    ``sad.caller.CallerProtocol``, during caller's initialization. Configurations
    for this callback is provided under ``trainer:spec:callbacks:``. An example is shown
    below::

        trainer:
          name: SGDTrainer
          spec:
            n_iters: 20
            w_l1: 0.1
            w_l2: 0.0
            u_idxs: [0, 1, 2, 3]
            callbacks:
            - name: "MetricsLoggingCallback"
              spec:
                every_iter: 1
                every_step: 2

    """

    def __init__(self, config: Dict, caller: CallerProtocol):
        """Instance __init__ method.

        Args:
            config (:obj:`dict`): Configuration dictionary that will be used to
                initialize a ``MetricsLoggingCallback`` instance.
            caller (:obj:`sad.callback.CallerProtocol`): A caller instance that is
                compliant with ``CallerProtocol``.
        """
        super().__init__(config, caller)
        self.history = defaultdict(list)

    @property
    def history(self) -> Dict[str, List[Tuple]]:
        """A dictionary that holds metrics history. It has following fields::

            history = {
                "step_end": [(iter_idx, step_idx, metrics), ... ],
                "iter_end": [(iter_idx, metrics), ... ]
            }

        The ``metrics`` in the list is a dictionary by itself, with following fields::

            metrics = {
                "ll": float,  // log likelihood of trained model
                "t_sparsity": float,  // sparsity of right item matrix
                "mse": float,  // MSE wrt true parameter X, available in simulation
                "ll0": float   // True log likelihood, available in simulation
            }

        This information will be saved to ``metricsloggingcallback_history.json``.
        """
        return self._history

    @history.setter
    def history(self, history: Dict[str, List[Tuple]]):
        self._history = history

    @property
    def every_iter(self) -> int:
        """Number of iterations every logging event happens. Read directly from
        ``"every_iter"`` field in ``self.spec``. A negative number suggests no metrics
        logging will happen at ``iteration`` ends."""
        return self.spec.get("every_iter", 1)

    @property
    def every_step(self) -> int:
        """Number of steps every logging event happens. Read directly from
        ``"every_step"`` field in ``self.spec``. A negative number suggests this
        callback will not log metrics at ``step`` ends."""
        return self.spec.get("every_step", -1)

    def on_loop_begin(self, **kwargs):
        pass

    def on_loop_end(self, **kwargs):
        """Will be called when caller's main loop finishes. When this method is
        triggered, the metrics history will be saved to a Json file with name
        ``metricsloggingcallback_history.json`` in the model folder.
        """
        working_dir = self.caller.task.output_dir
        model_s3_key_path = self.caller.model.s3_key_path
        class_name = self.__class__.__name__.lower()
        os.makedirs(os.path.join(working_dir, model_s3_key_path), exist_ok=True)
        with open(
            os.path.join(working_dir, model_s3_key_path, f"{class_name}_history.json"),
            "w",
        ) as fout:
            json.dump(self.history, fout)

    def on_iter_begin(self, iter_idx: int):
        pass

    def on_iter_end(
        self,
        iter_idx: int,
        ll: float = -1,
        t_sparsity: float = -1,
        mse: float = -1,
        ll0: float = -1,
        **kwargs,
    ):
        """Will be called to determine whether to log metrics at the end of an iteration.
        After confirming, it will organize metrics to a dictionary and push the
        dictionary into a history queue. The format of the dictionary is shown below::

            metrics = {
                "ll": float,  // log likelihood of trained model
                "t_sparsity": float,  // sparsity of right item matrix
                "mse": float,  // MSE wrt true parameter X, available in simulation
                "ll0": float   // True log likelihood, available in simulation
            }

        Args:
            iter_idx (:obj:`int`): The index of iteration, 0-based.
            ll (:obj:`float`): Log likelihood at current iteration.
            t_sparsity (:obj:`float`): The proportion of elements that are close to ``1``
                in ``T`` matrix.
            mse (:obj:`float`): The mean squared error between estimated item preference
                tensor and true tensor. Only available in simulation.
            ll0 (:obj:`float`): Log likelihood under true parameter values. Only
                available in simulation.
        """
        if self.every_iter <= 0 or (iter_idx % self.every_iter) != 0:
            return

        n_iters = self.caller.n_iters
        self.logger.info(
            f"Iteration: {iter_idx}/{n_iters}, "
            f"ll: {ll:.2e}, t_sparsity: {t_sparsity:.2e}, mse: {mse:.2e}, "
            f"ll0: {ll0:.2e}."
        )

        metrics = {"ll": ll, "t_sparsity": t_sparsity, "mse": mse, "ll0": ll0}
        self.history["iter_end"].append((iter_idx, metrics))

    def on_step_begin(self, iter_idx: int, step_idx: int, **kwargs):
        pass

    def on_step_end(
        self,
        iter_idx: int,
        step_idx: int,
        ll: float = -1,
        t_sparsity: float = -1,
        mse: float = -1,
        ll0: float = -1,
        **kwargs,
    ):
        """Will be called to determine whether to log metrics at end of a step. After
        confirming, it will organize metrics to a dictionary and push the dictionary
        into history queue. The format of the dictionary is shown below::

            metrics = {
                "ll": float,  // log likelihood of trained model
                "t_sparsity": float,  // sparsity of right item matrix
                "mse": float,  // MSE wrt true parameter X, available in simulation
                "ll0": float   // True log likelihood, available in simulation
            }

        Args:
            iter_idx (:obj:`int`): The index of iteration, 0-based.
            step_idx (:obj:`int`): The index of step, 0-based.
            ll (:obj:`float`): Log likelihood at current step.
            t_sparsity (:obj:`float`): The proportion of elements that are close to ``1``
                in ``T`` matrix.
            mse (:obj:`float`): The mean squared error between estimated item preference
                tensor and true tensor. Only available in simulation.
            ll0 (:obj:`float`): Log likelihood under true parameter values. Only
                available in simulation.
        """
        if self.every_step <= 0 or (step_idx % self.every_step) != 0:
            return

        n_iters = self.caller.n_iters
        self.logger.info(
            f"Iteration: {iter_idx}/{n_iters}, Step: {step_idx}, "
            f"ll: {ll:.2e}, t_sparsity: {t_sparsity:.2e}, mse: {mse:.2e}, "
            f"ll0: {ll0:.2e}."
        )

        metrics = {"ll": ll, "t_sparsity": t_sparsity, "mse": mse, "ll0": ll0}
        self.history["step_end"].append((iter_idx, step_idx, metrics))
