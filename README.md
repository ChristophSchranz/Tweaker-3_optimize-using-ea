# Tweaker-3's Optimization using an Evolutionary Algorithm

This repo is used to optimize the parameters for the [Tweaker-3](https://github.com/ChristophSchranz/Tweaker-3) 3D print auto-rotation module.

## Setup

Cloning and checking out the correct branch:
```bash
git clone --recurse-submodules https://github.com/ChristophSchranz/Tweaker-3_optimize-using-ea.git
cd Tweaker-3_optimize-using-ea/Tweaker-3
git checkout ea-optimize
cd ..
```

Creating the virtualenv and installing the requirements
```bash
conda create -n .venv_deap python=3.7.6 anaconda
conda activate .venv_deap
pip install -r requirements.txt
```
Don't forget to deactivate the virtualenv afterwards:
```bash
conda deactivate
```

## Running

Set the desired configs in `optimize_tweaker.py`, like 
```python
individuals = 50
n_generations = 50
n_objects = 50
```
Then, run with the `scoop` parameter that allows multiprocessing:
```bash
python3 optimize_tweaker.py -m scoop
```

