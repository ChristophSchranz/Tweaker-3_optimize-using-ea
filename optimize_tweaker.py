# This script tests the installation of the deap-project that is used for ea
# https://github.com/deap/deap
import os
import json
import random
import logging
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

chromosome_mapping = [("ABSOLUTE_F", 100.0), ("RELATIVE_F", 1.0), ("CONTOUR_F", 1.0), ("FIRST_LAY_H", 0.1),
                    ("TAR_A", 1.0), ("TAR_B", 0.01), ("TAR_C", 1.0), ("TAR_D", 1.0), ("BOTTOM_F", 1),
                    ("PLAFOND_ADV_A", 0.01), ("PLAFOND_ADV_B", 0.2), ("PLAFOND_ADV_C", 0.01),
                    ("ANGLE_SCALE", 0.1), ("ASCENT", 0.1), ("NEGL_FACE_SIZE", 1.0), ("CONTOUR_AMOUNT", 0.01)]
chromosome_dict = dict(chromosome_mapping)

# CHROMOSOMES = ["VECTOR_TOL", "PLAFOND_ADV", "FIRST_LAY_H", "NEGL_FACE_SIZE", "ABSOLUTE_F", "RELATIVE_F", "CONTOUR_F"]
CHROMOSOMES = [chrome[0] for chrome in chromosome_mapping]

n_individuals = 10  # 25 was good
n_generations = 5
n_objects = 5


# Create class to store  model-wise statistics about errors and miss-classifications
class StatPos:
    def __init__(self):
        self.pos = 0
        self.max_pos = n_individuals * n_generations - 1

    def increase(self):
        # Increment with max_pos as upper bound
        if self.pos < self.max_pos:
            self.pos += 1

    def get(self):
        return self.pos


stat_pos = StatPos()
statistics = pd.DataFrame(0.0, index=range(n_individuals * n_generations),
                          columns=sorted([f"Model{i}.stl_error" for i in range(1, n_objects+1)] +
                                         [f"Model{i}.stl_miss" for i in range(1, n_objects+1)]))

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
    # iterate through multiple objects and compare to real values
    for model_number, model in enumerate(ref["models"][:n_objects+1]):
        error_per_model = 0
        miss_per_model = 0
        # extract the filename and run the tweaker
        input_file = os.path.join("data", "Models", model["name"])
        try:
            result = evaluate_tweaker(parameter, input_file, verbose=verbose)
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
                    statistics.loc[stat_pos.get()][f"{model['name']}_miss"] = 0.25
                elif ref_a['grade'] == "C":
                    miss_per_model += 1
                    statistics.loc[stat_pos.get()][f"{model['name']}_miss"] = 1
                break
        # Add error if alignment is not found
        if not referred_flag:
            miss_per_model += 1
            statistics.loc[stat_pos.get()][f"{model['name']}_miss"] = 1

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
                            referred_value = 1/2
                        elif ref_a['grade'] == "C":
                            referred_value = 1
                        break
                # Increase the error, compute the squared residual and normalize with 1/(|results|*|n_objects|
                inc = 2/(len(result.best_5)*n_objects) * (referred_value - 1/(1 + np.exp(0.5*(10-alignment[4]))))**2
                error_per_model += inc
                # Adding high error on negative unprintability
                if alignment[4] < 0.0:
                    error_per_model += 1 + abs(alignment[4])

        # logistic transformation with a turning point in (10, 1), low value for x=0 and a maximum of 3
        # normalized with 0.5/|n_objects|
        error_per_model += 1/n_objects * 1/(1 + np.exp(0.5*(10-result.unprintability)))

        statistics.loc[stat_pos.get()][f"{model['name']}_error"] = error_per_model
        error += error_per_model
        miss += miss_per_model

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
    return chromosome_dict[name] * allele


def map_all_parameters(chromosome, exact=False):
    """
    This functions maps each allel of the chromosome that is around 1 into the appropriate scales.
    Therefore, it maps the genotype to the phenotype.
    :param exact: rounds the chromosomes to 6 digits if true
    :param chromosome: chromosome that contains all genes
    :return: list of the real values
    """
    if exact:
        return [chromosome[i] * allele[1] for i, allele in enumerate(chromosome_mapping)]
    else:
        return [round(chromosome[i] * allele[1], 6) for i, allele in enumerate(chromosome_mapping)]



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
        return np.random.normal(1, 0.2)

    # Draw random variables for the initial population
    toolbox.register("attr_float", random.random)
    toolbox.register("attr_one", about_one)

    # Length of the chromosome is specified with n
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_one, n=len(CHROMOSOMES))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Define the genetic operations
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.25, indpb=0.75)  # sigma of 0.25 is the best
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create hall of fame of size ceiling(2.5%)
    hall_of_fame = tools.HallOfFame(maxsize=int(n_individuals * 0.025) + 1)

    # Create a population and update the history
    population = toolbox.population(n=n_individuals)

    df = pd.DataFrame(index=range(1, n_generations+1), columns=["top", "median", "best"])
    df.index.name = "gen"
    fittest_ever = None
    for gen in range(1, n_generations+1):
        print(f"Generation {gen} of {n_generations}:")
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.4)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            print(f"The phenotype {map_all_parameters(ind)} \t has a fitness of: {round(fit[0], 6)}")
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

        print(f"Best phenotype so far is: {map_all_parameters(hall_of_fame[0])} with a fitness of "
              f"{round(hall_of_fame[0].fitness.values[0], 6)}.\n")

    df.to_csv(os.path.join("data", f"{datetime.now().date().isoformat()}_"
                                   f"{n_generations}gen_{n_individuals}ind_{n_objects}obj-df.csv"))
    print(df)

    statistics.to_csv(os.path.join("data", f"{datetime.now().date().isoformat()}_"
                                           f"{n_generations}gen_{n_individuals}ind_{n_objects}obj-stats.csv"))
    print(statistics.head())

    results = evaluate(hall_of_fame[0], verbose=True)
    print(results)
