# This script tests the installation of the deap-project that is used for ea
# https://github.com/deap/deap
import os
import json
import random
import logging
import numpy as np
import pandas as pd

from deap import creator, base, tools, algorithms
from tweaker_phenotype import evaluate_tweaker, map_all_parameters, map_parameters

# Define the parameters name and default values. The phenotype mapping is set in tweaker_phenotype
CHROMOSOMES = ["VECTOR_TOL", "PLAFOND_ADV", "FIRST_LAY_H", "NEGL_FACE_SIZE",
               "ABSOLUTE_F", "RELATIVE_F", "CONTOUR_F"]

n_individuals = 100  # 25 was good
n_generations = 250
n_objects = 50


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


def evaluate(individual, verbose=False):
    """
    This function evaluates an individual. In the first step, the raw alleles of the chromosome (individual) are
    mapped to the phenotype's parameters. Then, for multiple files the auto orientation is called and the final result
    is returned. In the final step, the multiple orientation-unprintability mappings for multiple files are aggregated
    to a single fitness value that is returned.
    :param individual: A chromosome that consists of multiple normalized alleles for each gene
    :param verbose: whether or not additional messages are printed
    :return: The fitness value that grades the individual's ability to auto-rotate the files successfully.
    """

    parameter = dict()
    for i, cr_name in enumerate(CHROMOSOMES):
        parameter[cr_name] = map_parameters(cr_name, individual[i])
        # Some parameter can't be zero or below
        if individual[i] <= 0:
            logger.info("Non-positive parameter in phenotype.")
            return n_objects * (1 + 2 * abs(parameter[cr_name])),

    if verbose:
        print("Evaluating with parameter:")
        for key, val in parameter.items():
            print(f"  {key}:    \t{val}")

    error = 0
    # iterate through multiple objects and compare to real values
    for model_number, model in enumerate(ref["models"][:n_objects+1]):
        error_per_model = 0
        # extract the filename and run the tweaker
        inputfile = os.path.join("data", "Models", model["name"])
        result = evaluate_tweaker(parameter, inputfile, verbose=verbose)

        # Compare the resulting best alignment with the reference alignment
        referred_flag = False
        for ref_a in model["alignments"]:
            v = [ref_a["x"], ref_a["y"], ref_a["z"]]
            if sum([(result.alignment[i] - v[i]) ** 2 for i in range(3)]) < 1e-5:
                # print(f"found alignment {v} with grade {ref_a['grade']}")
                referred_flag = True
                if ref_a['grade'] == "A":
                    error += 0
                elif ref_a['grade'] == "B":
                    error += 0.25  # Here the error can be lower, as B ist still a good value
                    statistics.loc[stat_pos.get()][f"{model['name']}_miss"] = 0.25
                elif ref_a['grade'] == "C":
                    error += 1
                    statistics.loc[stat_pos.get()][f"{model['name']}_miss"] = 1
                break
        # Add error if alignment is not found
        if not referred_flag:
            error += 1
            statistics.loc[stat_pos.get()][f"{model['name']}_miss"] = 1

        # Weight the values of the unprintablities
        for alignment in result.best_5:
            # Return negative slope for negative and therefore invalid unprintability. Do it for each alignment
            if alignment[4] < 0:  ## Alignment[4] is its unprintability
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
                error_per_model += 1/(len(result.best_5)*n_objects) * (referred_value - 1/(1 + np.exp(0.5*(10-alignment[4]))))**2

        # logistic transformation with a turning point in (10, 1), low value for x=0 and a maximum of 3
        # normalized with 0.5/|n_objects|
        error_per_model += 0.5/n_objects * 1/(1 + np.exp(0.5*(10-result.unprintability)))

        statistics.loc[stat_pos.get()][f"{model['name']}_error"] = error_per_model
        error += error_per_model

    # Update positions as the individual was evaluated on each model file
    stat_pos.increase()
    return error,


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
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_one, n=7)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Define the genetic operations
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.25, indpb=0.6)  # sigma of 0.25 is the best
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
        df.loc[gen]["best"] = map_all_parameters(hall_of_fame[0])

        print(f"Best phenotype so far is: {map_all_parameters(hall_of_fame[0])} with a fitness of {round(hall_of_fame[0].fitness.values[0], 6)}.\n")

    df.to_csv(os.path.join("data", f"DataFrame_{n_generations}gen_{n_individuals}inds_{n_objects}objects.csv"))
    print(df)

    statistics.to_csv(os.path.join("data", f"DataFrame_{n_generations}gen_{n_individuals}inds_{n_objects}objects_model-stats.csv"))
    print(statistics.head())

    result = evaluate(hall_of_fame[0], verbose=True)
    print(result)
