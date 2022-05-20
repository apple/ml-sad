#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from sad.task.base import TaskBase


class DummyTask(TaskBase):
    """A concrete dummy task class that will be used to create a default task instance.
    This class inherits all existing properties in ``sed.task.base.TaskBase``.

    """

    def run(self):
        self.logger.info("Running a dummy task means nothing will be done.")
