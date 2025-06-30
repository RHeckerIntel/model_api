#
# Copyright (C) 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

try:
    from openvino import Core

    _ = Core()  # Triggers loading of shared libs like libopenvino.so
except Exception as e:
    raise ImportError(f"Failed to initialize OpenVINO runtime: {e}")


from ._model_api import ClassificationModel

__all__ = [ClassificationModel]
