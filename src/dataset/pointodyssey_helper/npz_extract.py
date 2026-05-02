"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import os
import glob


def process_path(sample_dir):
    """Read anno.npz and save annotation for each frame."""
    print(f"processing {sample_dir}")
    for seq_dir in glob.glob(os.path.join(sample_dir, "*")):
      
      if not os.path.isdir(seq_dir):
        continue
      print(f" processing {seq_dir}")
      anno_name = os.path.join(seq_dir, "anno.npz")
      info_name = os.path.join(seq_dir, "info.npz")
      if not os.path.isfile(anno_name):
        print(f'{anno_name} not found')
        continue
      # Load data
      data_dict = {}
      anno = np.load(anno_name)
      for key in anno.keys():
        data_dict[key] = anno[key]

      # for key in data_dict.keys():
      #   print(key, data_dict[key].shape)

      # Sample and save
      seq_len = data_dict['trajs_2d'].shape[0]
      save_dir = os.path.join(seq_dir, "annos")
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f'creating {save_dir}')

      sample_dict = {}
      for t in range(seq_len):
        for key, val in data_dict.items():
          data_key = data_dict[key]
          if data_key.shape == ():
            sample_data_t = None
          else:
            sample_data_t = data_key[t]
          sample_dict[key] = sample_data_t 
        save_filename = os.path.join(save_dir, f"anno_{t:05d}.npz")
        np.savez(save_filename, **sample_dict)

for key in ['train', 'val', 'test']:
# for key in ['sample']:

    sample_dir = f"YOUR_PATH_TO_POINT_ODYSSEY/point_odyssey/{key}"
    if not os.path.isdir(sample_dir):
      raise ValueError(f'{sample_dir} is not Found.')
    process_path(sample_dir)

