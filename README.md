# Anonymous Repository

**This code is provided solely for AAAI 2026 anonymous review.**

**Any other use, distribution, or publication is strictly prohibited.**

## Dataset

We use publicly available datasets for all experiments:  

- synthetic dataset from [D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html).
- real-world dataset from [NeRF-DS](https://jokeryan.github.io/projects/nerf-ds/).

We organize the datasets as follows:

```shell
├── data
│   | D-NeRF 
│     ├── Hell Warrior
│     ├── Mutant
│     ├── Hook
│     ├── Bouncing Balls
│     ├── Lego
│     ├── T-Rex
│     ├── Stand Up
│     ├── Jumping Jacks
│   | NeRF-DS
│     ├── Sieve
│     ├── Plate
│     ├── Bell
│     ├── Press
│     ├── Cup
│     ├── As
│     ├── Basin
```

> Dataset Note: D-NeRF Lego Scene
We found an **inconsistency in the D-NeRF Lego dataset**:
The training and test sets contain different scenes, which can be verified by the orientation of the Lego shovel.
To ensure fair evaluation, **we exclude the Lego scene when reporting average results in our comparisons**.
For the Lego scene, we use its validation set as the test set instead.
Detailed results for the Lego scene are provided in the appendix.

## Submodules

This project depends on two submodules:
```shell
depth-diff-gaussian-rasterization
simple-knn
```
> The compressed package for these submodules is provided as GARO.zip in the supplementary materials.

## Run

### Environment

```shell
git clone xxxxxxxxxxxxxxx --recursive
cd GARO

conda create -n garo_env python=3.10
conda activate garo_env

# install pytorch
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121

# install dependencies
pip install -r requirements.txt
```

### Train

**D-NeRF:**

```shell
python train.py -s path/to/your/d-nerf/dataset -m output/exp-name --eval --is_blender --use_garo
```

**NeRF-DS:**

```shell
python train.py -s path/to/your/real-world/dataset -m output/exp-name --eval --iterations 20000 --use_garo
```

### Render

```shell
python render.py -m output/exp-name --mode render
```

### Evaluation

```shell
python metrics.py -m output/exp-name
```
