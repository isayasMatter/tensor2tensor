# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

_ENTI_TRAIN_DATASETS = [
    [
        "https://www.cse.unr.edu/~iadhanom/parallel-en-ti/training_set.tar.gz",
        ("jw300.train.en", "jw300.train.ti")
    ]
]
_ENTI_TEST_DATASETS = [
    [
        "https://www.cse.unr.edu/~iadhanom/parallel-en-ti/test_set.tar.gz",
        ("jw300.test.en", "jw300.test.ti")
    ],
]


@registry.register_problem
class TranslateEnti(translate.TranslateProblem):
  """En-ti translation."""
  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  @property
  def additional_training_datasets(self):
    """Allow subclasses to add training datasets."""
    return []

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    train_datasets = _ENTI_TRAIN_DATASETS + self.additional_training_datasets
    return train_datasets if train else _ENTI_TEST_DATASETS
