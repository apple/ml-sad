#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import datetime
import os
import sys
from typing import Dict

from sad.generator import GeneratorBase, GeneratorFactory
from sad.model import ModelBase, ModelFactory
from sad.task.base import TaskBase
from sad.trainer import TrainerBase, TrainerFactory
from sad.utils.job import read_from_yaml
from sad.utils.logging import setup_module_level_logger


class TrainingTask(TaskBase):
    """A concrete task class that will be responsible to train a model.

    This class inherits all existing properties in ``sed.task.base.TaskBase``.

    """

    @property
    def filename(self) -> str:
        """A relative path pointing to where user-item interaction data are located.
        The path is relative to ``self.input_dir``.
        """
        return self.config.get("filename")

    @property
    def model_id(self) -> str:
        """A string that uniquely identifies a trained model. It is usually set to
        ``"model_{self.task_id}"``."""
        model_id = self.config.get("model_id") or f"model_{self.task_id}"
        return model_id

    @property
    def trainer_config(self) -> Dict:
        """A dictionary read from configuration of the task. It specifies the
        configuration to initialize a trainer of type
        ``sad.trainer.TrainerBase``. Will read directly from ``"trainer"`` field from
        ``self.config``.

        An example is shown below::

            name: SGDTrainer
            spec:
              n_iters: 50
              u_idxs: [0, 1, 2, 3, 4, 5]
              w_l1: 0.01
              w_l2: 0.01

              callbacks:
              - name: "MetricsLoggingCallback"
                spec:
                  every_iter: 1
                  every_step: 2

        """
        return self.config.get("trainer", None)

    @property
    def model_config(self) -> Dict:
        """A dictionary read from configuration of the task. It specifies the
        configuration to initialize a model of type ``sad.model.ModelBase``. Will read
        directly from ``"model"`` field from ``self.config``.

        An example is shown below::

            name: SADModel
            spec:
              n: 200
              m: 500
              k: 100

        """
        return self.config.get("model", None)

    @property
    def generator_config(self) -> Dict:
        """A dictionary read from configuration of the task. It specifies the
        configuration to initialize a generator of type
        ``sad.generator.GeneratorBase``. Will read directly from ``"generator"`` field
        from ``self.config``.

        An example is shown below::

            name: ImplicitFeedbackGenerator
            spec:
              u_batch: 50
              i_batch: 100

        """
        return self.config.get("generator", None)

    def create_model(self) -> ModelBase:
        """Instance method to initialize a model for training.

        Returns:
            :obj:`sad.model.ModelBase`: An instance of model class that will be
            trained in current task.

        Raises:
            RuntimeError: When a model instance is not able to initialize from
                configuration in ``self.model_config``.

        """
        model_config = self.model_config

        # setup relative path
        now = datetime.datetime.now()
        s3_key_path = os.path.join(
            "model",
            f"{now.year}",
            f"{now.month:02d}",
            f"{now.day:02d}",
            self.model_id,
        )
        model_config["spec"]["s3_key_path"] = s3_key_path

        try:
            model = ModelFactory.produce(model_config, task=self)
        except Exception as ex:
            self.logger.error(
                f"Unable to create model with config {model_config}: {ex}"
            )
            raise RuntimeError

        return model

    def create_generator(self, model: ModelBase) -> GeneratorBase:
        """Instance method to create a generator for training.

        Args:
            model (:obj:`sad.model.ModelBase`): An instance of model that will be
                associated with the generator.

        Returns:
            :obj:`sad.generator.GeneratorBase`: An instance of generator class that
            will be used to train the model in a trainer, an instance of
            ``sad.trainer.TrainerBase``.

        Raises:
            RuntimeError: When a generator instance is not able to create from
                configuration in ``self.generator_config``.

        """
        generator_config = self.generator_config
        try:
            generator = GeneratorFactory.produce(
                generator_config, model=model, task=self
            )
        except Exception as ex:
            self.logger.error(
                f"Unable to create generator with config {generator_config}: {ex}"
            )
            raise RuntimeError
        return generator

    def create_trainer(self, model: ModelBase, generator: GeneratorBase) -> TrainerBase:
        """Instance method to create a trainer for training. Require an instance of
        ``sad.model.ModelBase`` and a ``sad.generator.GeneratorBase``.

        Args:
            model (:obj:`sad.model.ModelBase`): An instance of model that will be
                associated with the trainer.
            generator (:obj:`sad.generator.GeneratorBase`): An instance of generator
                that will be used by trainer.

        Returns:
            :obj:`sad.trainer.TrainerBase`: An instance of trainer class that will be
            used in current task.

        Raises:
            RuntimeError: When a trainer instance is not able to initialize from
                configuration in ``self.trainer_config``.

        """
        trainer_config = self.trainer_config
        try:
            trainer = TrainerFactory.produce(
                trainer_config, model=model, generator=generator, task=self
            )
        except Exception as ex:
            self.logger.error(
                f"Unable to create trainer with config {trainer_config}: {ex}"
            )
            raise RuntimeError

        return trainer

    def run(self):
        """Run training task."""
        self.show_config()

        model = self.create_model()
        generator = self.create_generator(model)

        # prepare data for training
        filepath = os.path.join(self.input_dir, self.filename)
        if not os.path.exists(filepath):
            self.logger.warning(f"{filepath} for training doesn't exist. Aborting.")
            return

        generator.add(filepath)

        trainer = self.create_trainer(model, generator)
        trainer.train()

        self.logger.info("Task succeed!")


def run_task(config_file: str = None):
    """Main function that will be called when running a processing task.

    Args:
        config_file (:obj:`str`): A ``yml`` file that contains configurations for
            running the processing task. Optional, when ``None`` a default file at
            ``./ppgflow/tasks/processing/config.yml`` will be used.

    """
    setup_module_level_logger(["tasks", "utils", "processor", "data"])
    if not config_file:
        config_file = "./ppgflow/tasks/processing/config.yml"

    config = read_from_yaml(config_file)

    input_dir = config.get("input_dir")
    output_dir = config.get("output_dir")
    task = TrainingTask(config, input_dir, output_dir)
    task.run()


if __name__ == "__main__":
    config_file = None
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    run_task(config_file)
