# Differential Gaussian Rasterization with camera pose, point maps, and motion maps

This code provides differentiable Gaussian rasterization that supports:

* Analytical gradient for SE(3) camera poses
* Analytical gradient for rendered point maps
* Analytical gradient for rendered motion maps

The code is built on top of the two great implementations, [Differential Gaussian Rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) and [diff-gaussian-rasterization-w-pose](https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose).

If you find this implementation useful for your projects, please cite the following three papers:


```bibtex
@inproceedings{Hur:2025:UFO,
  title={{UFO}-4{D}: Unposed Feedforward 4{D} Reconstruction from Two Images},
  author={Junhwa Hur and Charles Herrmann and Songyou Peng and Philipp Henzler and Zeyu Ma and Todd Zickler and Deqing Sun},
  booktitle={under review},
  year={2025},
}
```

```bibtex
@article{kerbl20233d,
  title={3{D} Gaussian splatting for real-time radiance field rendering.},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal={ACM Trans. Graph.},
  volume={42},
  number={4},
  pages={139--1},
  year={2023},
}
```

```bibtex
@inproceedings{Matsuki:Murai:etal:CVPR2024,
  title={{G}aussian {S}platting {SLAM}},
  author={Hidenobu Matsuki and Riku Murai and Paul H. J. Kelly and Andrew J. Davison},
  booktitle={CVPR},
  year={2024},
}
```