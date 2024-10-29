# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pre-processing
Post-processing utilities for question answering.
"""
import json
import logging
import os, time, random, collections
from typing import Any, Optional, Tuple
from contextlib import contextmanager

import pickle

logger = logging.getLogger(__name__)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def save_pickle(target, path):
    with open(path, "wb") as file:
        pickle.dump(target, file)
