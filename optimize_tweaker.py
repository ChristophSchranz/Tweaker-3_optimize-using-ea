# This script tests the installation of the deap-project that is used for ea
# https://github.com/deap/deap
import os
import json
import random
import logging
import time

import numpy as np
import pandas as pd
from datetime import datetime

from deap import creator, base, tools, algorithms
from tweaker_phenotype import evaluate_tweaker

# Define the parameters name and default values. The phenotype mapping is set in tweaker_phenotype
# Orig: [0.001, 0.2, 0.25, 1, 100, 1, 0.5] (4.052568577348239, 3.25)
# 2020-04-11 = [0.00132, 0.1035, 0.1055, 2.44, 340.0, 0.1109, 0.52] (2.1221719828604946, 1.75)
# 2020-04-13 [0.001451, 0.291153, 0.028855, 1.400084, 669.452018, 1.566669, 0.088707] (1.7734233947572995, 1.5)
# 2020-04-14: [0.001241, 0.220733, 0.027727, 1.396286, 222.797949, 0.368569, 0.145722] (1.8986392001671926, 1.5)
# From here on, it is a 0.25 harder.
# 2020-04-16: [1.0949166432153108, 1.3801130552407321, 4.212270839789636, 2.9124195997415634, -0.12123768217392668, 1.9189241901924767, 0.851190555042431, 1.0501212774303932, 0.0619619118477204, 0.3218715070457661, 2.2793963527249073, -0.0006922836147128827, 0.4803757083992928, 0.02205278899619667, 0.00031301424728405193,
# 0.30571098484924314, 0.09500435168604665, 0.42768694157524495, 0.010720139799460952] (2,28859922694892, 2.0)
# of 100 objects:
# 2020-04-18: ... (7, 6.25)
# 2020-04-19: ... (9, 8)
# 2020-04-22_100gen_150ind_100obj, branch ea-optimize_20200414: (5,36413058858435, 4.24)
# [98.5883768434822, 1.162524130132582, 0.1605628698074317, 0.08473208766649207, 0.7015860182950739, 0.26931582120058184, 1.554247674370683, 0.44833952635629537, 0.8840613107383717, 0.24174313621949237, 0.7254421358435629, 119.03812433302157, 0.43859512908527554, 0.012893512521961371]
# 2020-04-23: [0.0042576279028611365, 0.33219207773978865, 4.60902879923442, 0.7511151864983084, 2.1286462051546766, 3.3973820298013355, 1.667284602578321, 0.02185198515157545, 0.1029284730801404, 0.011883209962576614, 0.41527230467417364, 0.014979849261818057, 0.8804181849453453, 0.024725248634437477]
# with a fitness of (4.83322, 4.0)

chrome_map = [("TAR_A", lambda x: 0.02+0.01*x),
              ("TAR_B", lambda x: 0.2+0.01*x),
              ("RELATIVE_F", lambda x: 2.6+0.3*x),
              ("CONTOUR_F", lambda x: 1.+0.1*x),
              ("BOTTOM_F", lambda x: 0.85+0.1*x),
              ("TAR_C", lambda x: 1.+0.1*x),
              ("TAR_D", lambda x: 1.1+0.1*x),
              ("TAR_E", lambda x: 0.02+0.003*x),

              ("FIRST_LAY_H", lambda x: 0.08 + 0.01 * x),
              ("VECTOR_TOL", lambda x: 0.001 + 0.0005 * x),

              ("NEGL_FACE_SIZE", lambda x: 0.3+0.01*x),
              ("ASCENT", lambda x: -0.5+0.01*x),
              ("PLAFOND_ADV", lambda x: 0.1+0.01*x),
              ("CONTOUR_AMOUNT", lambda x: 0.014+0.001*x),
              ("OV_H", lambda x: 2+0.05*x)]
chrome_dict = dict(chrome_map)
CHROMOSOMES = [chrome[0] for chrome in chrome_map]

min_volume = True
n_individuals = 200
n_generations = 150
n_objects = 100
# Phases: 1: use search space, 2: min. miss-class., 3: min. exec. time too, 4: min. all


# Create class to store  model-wise statistics about errors and miss-classifications
class StatPos:
    def __init__(self):
        self.pos = 0
        self.max_pos = n_individuals * n_generations - 1
        self.eval_unprintablity = False
        self.eval_times = False

    def increase(self):
        # Increment with max_pos as upper bound
        if self.pos < self.max_pos:
            self.pos += 1

    def get(self):
        return self.pos


stat_pos = StatPos()
stats = pd.DataFrame(0.0, index=range(n_individuals * n_generations),
                     columns=sorted([f"Model{i}.stl_error" for i in range(1, n_objects + 1)] +
                                    [f"Model{i}.stl_miss" for i in range(1, n_objects + 1)]))

# Read reference file that holds all grades for the models
ref_file = os.path.join("data", "ref_fitness.json")
with open(ref_file) as f:
    ref = json.loads(f.read())


def evaluate(individual, verbose=False, is_phenotype=False):
    """
    This function evaluates an individual. In the first step, the raw alleles of the chromosome (individual) are
    mapped to the phenotype's parameters. Then, for multiple files the auto orientation is called and the final result
    is returned. In the final step, the multiple orientation-unprintability mappings for multiple files are aggregated
    to a single fitness value that is returned.
    :param individual: A chromosome that consists of multiple normalized alleles for each gene
    :param verbose: whether or not additional messages are printed
    :param is_phenotype: specifies whether the given individual is a phenotype that should not be mapped
    :return: The fitness value that grades the individual's ability to auto-rotate the files successfully.
    """

    parameter = dict()
    for i, cr_name in enumerate(CHROMOSOMES):
        if is_phenotype:
            parameter[cr_name] = individual[i]
        else:
            parameter[cr_name] = map_parameters(cr_name, individual[i])
        # # Some parameter can't be zero or below
        # if individual[i] <= 0:
        #     logger.info("Non-positive parameter in phenotype.")
        #     return n_objects * (1 + 2 * abs(parameter[cr_name])),

    if verbose:
        print("Evaluating with parameter:")
        for key, val in parameter.items():
            print(f"  {key}:    \t{val}")

    error = 0
    miss = 0
    st = time.time()
    # iterate through multiple objects and compare to real values
    for model_number, model in enumerate(ref["models"][:n_objects + 1]):
        error_per_model = 0
        miss_per_model = 0
        # extract the filename and run the tweaker
        input_file = os.path.join("data", "Models", model["name"])
        try:
            result = evaluate_tweaker(parameter, input_file, min_volume=min_volume, verbose=verbose)
        except RuntimeWarning:
            print("A RuntimeWarning occurred, returning.")
            return 2 * n_objects, 2 * n_objects

        # Compare the resulting best alignment with the reference alignment
        referred_flag = False
        for ref_a in model["alignments"]:
            v = [ref_a["x"], ref_a["y"], ref_a["z"]]
            if sum([(result.alignment[i] - v[i]) ** 2 for i in range(3)]) < 1e-5:
                # print(f"found alignment {v} with grade {ref_a['grade']}")
                referred_flag = True
                if ref_a['grade'] == "A":
                    miss_per_model += 0
                elif ref_a['grade'] == "B":
                    miss_per_model += 0.25  # Here the error can be lower, as B ist still a good value
                    stats.loc[stat_pos.get()][f"{model['name']}_miss"] = 0.25
                elif ref_a['grade'] == "C":
                    miss_per_model += 1
                    stats.loc[stat_pos.get()][f"{model['name']}_miss"] = 1
                break
        # Add error if alignment is not found
        if not referred_flag:
            miss_per_model += 1
            stats.loc[stat_pos.get()][f"{model['name']}_miss"] = 1

        if stats.eval_unprintablity:
            # Weight the values of the unprintablities
            for alignment in result.best_5:
                # Return negative slope for negative and therefore invalid unprintability. Do it for each alignment
                if alignment[4] < 0:  # Alignment[4] is its unprintability
                    error += 1 + abs(alignment[4])
                # Compare each unprintability to grades and score it. Bad alignments are rated as C per default
                else:
                    referred_value = 1  # Could be weighted lower
                    for ref_a in model["alignments"]:
                        v = [ref_a["x"], ref_a["y"], ref_a["z"]]
                        if sum([(alignment[0][i] - v[i]) ** 2 for i in range(3)]) < 1e-5:
                            # print(f"found alignment {v} with grade {ref_a['grade']}")
                            if ref_a['grade'] == "A":
                                referred_value = 0
                            elif ref_a['grade'] == "B":
                                referred_value = 1 / 2
                            elif ref_a['grade'] == "C":
                                referred_value = 1
                            break
                    # Increase the error, compute the squared residual and normalize with 1/(|results|*|n_objects|
                    inc = 2 / (len(result.best_5) * n_objects) * (
                            referred_value - 1 / (1 + np.exp(0.5 * (10 - alignment[4])))) ** 2
                    error_per_model += inc
                    # Adding high error on negative unprintability
                    if alignment[4] < 0.0:
                        error_per_model += 1 + abs(alignment[4])

        if stats.eval_unprintablity:
            # logistic transformation with a turning point in (10, 1), low value for x=0 and a maximum of 3
            # normalized with 0.5/|n_objects|
            error_per_model += 1/n_objects * 1/(1 + np.exp(0.5*(10-result.unprintability)))

        stats.loc[stat_pos.get()][f"{model['name']}_error"] = error_per_model
        error += error_per_model
        miss += miss_per_model

    if stats.eval_times:
        time_required = time.time() - st
        error += 5 * time_required / n_objects

    # Update positions as the individual was evaluated on each model file
    stat_pos.increase()
    # Miss in the second item is not used, but useful for explicit individual evaluation
    return error + miss, miss


def map_parameters(name, allele):
    """
    This functions maps the raw allele that is around 1 into an appropriate scale. Therefore, it maps the genotype
    onto the phenotype
    :param name: name of the gene
    :param allele: value for the specified gene
    :return: value that can be used by the algorithm
    """
    return chrome_dict[name](allele)


def map_all_parameters(chromosome, exact=False):
    """
    This functions maps each allele of the chromosome that is around 1 into the appropriate scales.
    Therefore, it maps the genotype to the phenotype.
    :param exact: rounds the chromosomes to 6 digits if true
    :param chromosome: chromosome that contains all genes
    :return: list of the real values
    """
    if exact:
        return [allele[1](chromosome[i]) for i, allele in enumerate(chrome_map)]
    else:
        return [float("%.4g" % allele[1](chromosome[i])) for i, allele in enumerate(chrome_map)]


def print_parameters(individual, is_phenotype=True):
    if not is_phenotype:
        individual = map_all_parameters(individual)
    chr_dict = {ch: individual[i] for i, ch in enumerate(CHROMOSOMES)}
    print(json.dumps(chr_dict, indent=4))


if __name__ == "__main__":
    # configure the logger
    logger = logging.getLogger("optimize Tweaker")
    logger.setLevel(logging.INFO)
    logging.basicConfig()

    # Set to a minimization problem
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # allow parallel computing.
    # These lines must be within the __main__ and the program must be started with a "-m scoop" parameter.
    from scoop import futures

    toolbox.register("map", map)


    def about_one():
        return np.random.normal(0, 2)


    # Draw random variables for the initial population
    toolbox.register("attr_float", random.random)
    toolbox.register("attr_one", about_one)

    # Length of the chromosome is specified with n
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_one, n=len(CHROMOSOMES))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Define the genetic operations
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)

    # Create hall of fame of size 1
    hall_of_fame = tools.HallOfFame(maxsize=1)

    # Create a population and update the history
    population = toolbox.population(n=n_individuals)

    today = datetime.now().date().isoformat()
    df = pd.DataFrame(index=range(1, n_generations + 1), columns=["top", "median", "best"])
    df.index.name = "gen"
    fittest_ever = None

    for gen in range(1, n_generations + 1):
        if gen == 1:
            print("Phase 1: search space to find single minimum value and use high mutation rate.")
            # Define for phase 1:
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2.5, indpb=0.75)
            toolbox.register("select", tools.selBest, k=int(0.4 * n_generations))
            stats.eval_times = False
            stats.eval_unprintablity = False
        elif gen == int(0.2 * n_generations):
            print("Phase 1.5: search space to find single minimum value and use high mutation rate.")
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.5, indpb=0.7)
        elif gen == int(0.4 * n_generations):
            print("Phase 2: Find minimal miss-classifications and use tournament selection with medium mutation rate.")
            toolbox.register("select", tools.selTournament, tournsize=3)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.6)
            hall_of_fame = tools.HallOfFame(maxsize=1)
        elif gen == int(0.6 * n_generations):
            print("Phase 3: Find minimal miss-classifications with fast execution times with medium mutation rate.")
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.5)
            hall_of_fame = tools.HallOfFame(maxsize=1)
            stats.eval_times = True
        elif gen == int(0.7 * n_generations):
            print("Phase 4: Find minimal composition with dominant miss-classifications with small mutation rate.")
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.35)
            hall_of_fame = tools.HallOfFame(maxsize=1)
            fits = toolbox.map(toolbox.evaluate, hall_of_fame)
            for fit, ind in zip(fits, hall_of_fame):
                print(f"Re-evaluate phenotype {map_all_parameters(ind)} \t has a fitness of: ({float('%.6g' % fit[0])}, {fit[1]})")
                ind.fitness.values = fit
            stats.eval_unprintablity = True
        elif gen == int(0.85 * n_generations):
            print("Phase 5: Fine-tune minimal composition value with very small mutation rate.")
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.15, indpb=0.2)
            hall_of_fame = tools.HallOfFame(maxsize=1)
            fits = toolbox.map(toolbox.evaluate, hall_of_fame)
            for fit, ind in zip(fits, hall_of_fame):
                print(f"Re-evaluate phenotype {map_all_parameters(ind)} \t has a fitness of: ({float('%.6g' % fit[0])}, {fit[1]})")
                ind.fitness.values = fit
        print(f"Generation {gen} of {n_generations}:")

        offspring = algorithms.varAnd(population, toolbox, cxpb=0.6, mutpb=0.4)  # set cxpb to not split the unpr. fct
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            print(f"The phenotype {map_all_parameters(ind)} \t has a fitness of: ({float('%.6g' % fit[0])}, {fit[1]})")
            ind.fitness.values = fit

        # Clone the best individuals from hall_of_fame into the offspring for the selection
        for fame in hall_of_fame:
            print(f"  loading fame {map_all_parameters(fame)} with fitness of {round(fame.fitness.values[0], 6)} "
                  f"into offspring")
            offspring.append(toolbox.clone(fame))

        # Individuals from the hall_of_fame are selected into the next generation
        population = toolbox.select(offspring, k=len(population))
        hall_of_fame.update(population)  # This must be after the selection

        df.loc[gen]["top"] = np.min([ind.fitness.values[0] for ind in population])
        df.loc[gen]["median"] = np.median([ind.fitness.values[0] for ind in population])
        df.loc[gen]["best"] = map_all_parameters(hall_of_fame[0], exact=True)

        # Saving results
        df.to_csv(os.path.join("data", f"{today}_{n_generations}gen_{n_individuals}ind_{n_objects}obj-df.csv"))
        stats.to_csv(os.path.join("data", f"{today}_{n_generations}gen_{n_individuals}ind_{n_objects}"
                                          f"obj-stats.csv"))
        print(f"Best phenotype so far is: {map_all_parameters(hall_of_fame[0])} with a fitness of "
              f"{hall_of_fame[0].fitness.values}.\n")

    print(df)
    print(stats.head())

    results = evaluate(hall_of_fame[0], verbose=False)
    print(f"Best phenotype is {map_all_parameters(hall_of_fame[0], exact=True)} with a fitness of "
          f"({round(hall_of_fame[0].fitness.values[0], 5)}, {results[1]})")
