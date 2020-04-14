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

## Evaluation

The reference value of the parameters are:
```python
# python3 -i optimize_tweaker.py 
original_individual = [0.001, 0.2, 0.25, 1, 100, 1, 0.5]
evaluate(original_individual, verbose=False, is_phenotype=True)
# yields (4.052568577348239, 3.25)
```

Where the first item in the tuple specifies the total error that 
consists of the deviations and the miss-classifications, and the 
second item is the miss-classification for all objects.
