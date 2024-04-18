# <p align="center">DunHuangStitch: Unsupervised Deep Image Stitching of Dunhuang Murals</p>
<p align="center">Yuan Mei*, Lichun Yang*, Mengsi Wang*, Tianxiu Yu`, Kaijun Wu*</p>
<p align="center">* the School of Electronic and Information Engineering, Lanzhou Jiaotong University</p>
<p align="center">` Institute of Digitization of Cultural Relics, Dunhuang Research Institute</p>

## Dataset (Synthetic dunhuang dataset)
We use the synthetic dunhuang dataset to train and evaluate our method. Please refer to [S-DunHuang]() for more details about this dataset.

## Dataset (Real dunhuang dataset)
We use the real dunhuang dataset to train and evaluate our method. Please refer to [R-DunHuang]() for more details about this dataset.

## Dataset (Natural scene dataset)
We use the UDIS-D dataset to train and evaluate our method. Please refer to [UDIS]() for more details about this dataset.


## Code
#### Requirement
* numpy 1.19.5
* pytorch 1.7.1
* scikit-image 0.15.0
* tensorboard 2.9.0

We implement this work with Ubuntu, 2080Ti, and CUDA11. Refer to [environment.yml]() for more details.

#### How to run it
Similar to UDIS, we also implement this solution in two stages:
* Stage 1 (unsupervised warp): please refer to  [Warp/readme.md]().
* Stage 2 (unsupervised composition): please refer to [Composition/readme.md]().



## Meta
If you have any questions about this project, please feel free to drop me an email.

Yuan Mei -- 2551161628@qq.com

