# HREM Module Implementation 🧩

This repository provides an implementation of the **Hybrid Residual Enhancement Module (HREM)**, inspired by two key research papers. 📑 The code integrates essential deep learning components to enhance feature extraction and fusion.

## 🔗 Dependencies on Key Papers
Our implementation is primarily based on the following two papers:

1. **[Paper 1]**:  "MD³Net: Integrating Model-Driven and Data-Driven Approaches for Pansharpening" 📄

 ```bibtex
@ARTICLE{9851415,
  author={Yan, Yinsong and Liu, Junmin and Xu, Shuang and Wang, Yicheng and Cao, Xiangyong},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={MD³Net: Integrating Model-Driven and Data-Driven Approaches for Pansharpening}, 
  year={2022},
  volume={60},
  number={},
  pages={1-16},
  keywords={Pansharpening;Task analysis;Spatial resolution;Neural networks;Deep learning;Convolutional neural networks;Remote sensing;Deep learning (DL);deep prior;model-driven and data-driven;pansharpening;remote sensing;unfolding algorithm},
  doi={10.1109/TGRS.2022.3196427}}
 ```

2. **[Paper 2]**: "CNN-based remote sensing pan-sharpening: a critical review" 📄

 ```bibtex
@ARTICLE{deng2022grsm,
author={L.-J. Deng, G. Vivone, M. E. Paoletti, G. Scarpa, J. He, Y. Zhang, J. Chanussot, and A. Plaza},
booktitle={IEEE Geoscience and Remote Sensing Magazine},
title={Machine Learning in Pansharpening: A Benchmark, from Shallow to Deep Networks},
year={2022},
pages={},
}
```


These works introduce the core ideas behind **CFEM (Contrast Feature Enhancement Module)**, **CMCM (Cross-Modal Compensation Module)**, and **HREM**, all of which are directly incorporated into this codebase.





## 📌 Key Features

- **Unified Training & Testing Process**:
  - The **train** and **test** procedures have been seamlessly integrated into a single workflow for simplicity. 🚀
- **HDF5 Data Format Support**:
  - The current implementation processes datasets stored in **.h5** format, ensuring efficient storage and access. 📂
- **Module-to-Paper Mapping**:
  - The **CFEM, CMCM, and HREM** modules directly correspond to those described in the referenced research papers, maintaining conceptual consistency. 🔍

## 🛠 Installation & Usage

To run the code, ensure you have the necessary dependencies installed:

## Requirements
* Python3.7+, Pytorch>=1.6.0
* NVIDIA GPU + CUDA
* h5py


## ❤Acknowledgments
A huge thanks to the authors of the referenced papers for their invaluable contributions to the field of deep learning and remote sensing! 🌍📡

For any questions or suggestions, feel free to reach out. Happy coding! 🎉


