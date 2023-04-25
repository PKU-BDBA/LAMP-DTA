# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ml_collections

def get_PMMA_config():
    """Returns the PMMA configuration."""
    config = ml_collections.ConfigDict()
    config.hidden_size = 128
    config.transformer = ml_collections.ConfigDict()
    config.transformer.num_heads = 4
    config.transformer.num_p_plus_s_layers = 12
    config.transformer.attention_dropout_rate = 0.2 # 0.0 - 0.2
    config.transformer.dropout_rate = 0.3 # 0.1 - 0.3
    config.classifier = 'token'
    config.representation_size = None
    config.mol_len = 128
    # config.mol_len = 64
    config.feat_len = 140
    # config.feat_len = 54
    return config
