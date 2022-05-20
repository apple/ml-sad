#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import copy
import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Dict


class TaskBase(ABC):
    """A task base class that all task subclasses will inherit from.

    A task is the main component in our workflow. For example, when training a model,
    an instance of ``sad.tasks.training.TrainingTask`` will be responsible to launch
    the training job.

    """

    def __init__(self, config: Dict, input_dir: str = None, output_dir: str = None):
        self.config = copy.deepcopy(config)
        self._logger = logging.getLogger(f"task.{self.__class__.__name__}")

        prefix = "" if self.is_local else "/mnt/"
        self.input_dir = input_dir or tempfile.mktemp(prefix=prefix)
        self.output_dir = output_dir or tempfile.mktemp(prefix=prefix)

    def show_config(self):
        """A function to print the configuration of a running task."""
        self.logger.info(
            f"{self.__class__.__name__} config: \n {json.dumps(self.config, indent=2)}"
        )
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")

    @property
    def input_dir(self) -> str:
        """An absolute path that points to input directory of a running task."""
        return self._input_dir

    @input_dir.setter
    def input_dir(self, input_dir: str):
        self._input_dir = os.path.abspath(os.path.expanduser(input_dir))

    @property
    def output_dir(self) -> str:
        """An absolute path that points to output directory of a running task."""
        return self._output_dir

    @output_dir.setter
    def output_dir(self, output_dir: str):
        self._output_dir = os.path.abspath(os.path.expanduser(output_dir))

    @property
    def artifact_dir(self) -> str:
        """A path points to an artifact directory. Will be the artifact directory
        from Bolt when running as a Bolt job. Otherwise, will be the same as
        ``self.output_dir``."""
        artifact_dir = self.config.get("artifact_dir") or self.output_dir
        return artifact_dir

    @property
    def logger(self) -> logging.Logger:
        """A logger instance to manage logging during the life-cycle of a task."""
        return self._logger

    @property
    def is_local(self) -> bool:
        """A boolean flag to indicate whether the task is running in local mode."""
        return self.config.get("is_local", True)

    @property
    def is_hc(self) -> bool:
        """A boolean flag to indicate if the task is running in HC."""
        return self.config.get("is_hc", False)

    @property
    def task_id(self) -> str:
        """A unique string to identify a running task."""
        return self.config.get("task_id")

    @task_id.setter
    def task_id(self, task_id: str):
        self.config["task_id"] = task_id

    @abstractmethod
    def run(self):
        raise NotImplementedError
