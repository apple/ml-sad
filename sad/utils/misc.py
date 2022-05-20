#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import logging
from typing import Any, Dict

from scipy.special import logit

_logger = logging.getLogger("utils.misc")


def my_logit(value: float, EPS: float = 1e-10) -> float:
    """Take logit of a given value. Input value will be restricted to ``[EPS, 1-EPS]``
    interval.

    Args:
        value (:obj:`float`): A value is between ``(0, 1)``. Due to numerical
            consideration, the value will be truncated to ``[EPS, 1-EPS]`` where ``EPS``
            is a small number.
        EPS (:obj:`float`): A small positive number that will be used to maintain
            numerical stability. Default to ``1e-10``.

    Returns:
        :obj:`float`: The logit of input ``value``.

    """
    if value < EPS:
        value = EPS
    elif value > 1 - EPS:
        value = 1 - EPS
    return logit(value)


def update_dict_recursively(dict_a: Dict, dict_b: Dict):
    """A helper function to absorb contents in ``dict_b`` into ``dict_a``, recursively.
    ``dict_a`` will be modified in place.

    Args:
        dict_a (:obj:`dict`): First dictionary that absorbs.
        dict_b (:obj:`dict`): Second dictionary in which all fields will be absorbed
            into ``dict_a``.

    Return:
        Modified input dictionary ``dict_a``.
    """

    for kb, vb in dict_b.items():
        if kb in dict_a:
            if isinstance(dict_a[kb], dict) and isinstance(vb, dict):
                dict_a[kb] = update_dict_recursively(dict_a[kb], vb)
            else:
                dict_a[kb] = vb  # overwrite existing values in dict_a otherwise
        else:
            dict_a[kb] = vb
    return dict_a
