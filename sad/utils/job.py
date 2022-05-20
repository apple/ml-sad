#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import logging
import os
import random
import string
from lib2to3.pytree import Base
from typing import Dict

import yaml

logger = logging.getLogger("utils.job")


def id_generator(size: int = 6, chars: str = string.ascii_uppercase + string.digits):
    """Randomly generate an id string.

    Args:
        size (:obj:`int`): The length of the id, in characters.
        chars (:obj:`str`): The set of characters that the random id string will be
            choosing from.

    Returns:
        :obj:`str`: A randomly generated string.

    """
    return "".join(random.choice(chars) for _ in range(size))


def read_from_yaml(filename: str) -> Dict:
    """Read a local yml configuration file, and extract its ``parameters`` field and
    return.

    Args:
        config_file (:obj:`str`): A local yml file. The ``parameters`` field in the yml
            file contains configurations.

    Returns:
        :obj:`dict`: A dictionary containing configurations to initialize a task.

    Raises:
        IOError: When provided ``filename`` argument does not point to a valid yml file.
        AssertionError: When ``parameters`` is not contained in the yml file.
    """
    config = {}
    if filename is None:
        raise IOError("Unable to read yaml file. No file is provided")
    filename = os.path.expanduser(filename)
    if not os.path.exists(filename):
        logger.error(f"Unable to read yaml file, {filename} does not exist")
        raise IOError

    try:
        with open(filename, "r") as fin:
            yaml_dict = yaml.load(fin, yaml.FullLoader)
    except Exception as ex:
        logger.error(f"Failed while reading yaml file: {ex}")
        raise IOError

    assert "parameters" in yaml_dict, "There are must be a parameters field in yaml."
    config.update(yaml_dict["parameters"])
    config["is_local"] = True
    config["task_id"] = f"local_{id_generator()}"
    return config
