# This script tests the installation of the deap-project that is used for ea
# https://github.com/deap/deap
import json
import logging
import os
import random
import numpy as np
from deap import creator, base, tools, algorithms
from tweaker_phenotype import evaluate_tweaker, map_all_parameters, map_parameters

# Define the parameters name and default values
CHROMOSOMES = [("VECTOR_TOL", 0.001), ("PLAFOND_ADV", 0.2), ("FIRST_LAY_H", 0.25), ("NEGL_FACE_SIZE", 1),
                    ("ABSOLUTE_F", 100), ("RELATIVE_F", 1), ("CONTOUR_F", 0.5)]

n_objects = 5
reffile = os.path.join("data", "ref_fitness.json")
with open(reffile) as f:
    ref = json.loads(f.read())

# configure the logger
logger = logging.getLogger("optimize Tweaker")
logger.setLevel(logging.INFO)
logging.basicConfig()

# Set to a minimization problem
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Draw random variables for the initial population
toolbox.register("attr_bool", random.random)

# Length of the chromosome is specified with n
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=7)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    """
    This function evaluates an individual. In the first step, the raw alleles of the chromosome (individual) are
    mapped to the phenotype's parameters. Then, for multiple files the auto orientation is called and the final result
    is returned. In the final step, the multiple orientation-unprintability mappings for multiple files are aggregated
    to a single fitness value that is returned.
    :param individual: A chromosome that consists of multiple normalized alleles for each gene
    :return: The fitness value that grades the individual's ability to auto-rotate the files successfully.
    """
    parameter = dict()
    for i, kv in enumerate(CHROMOSOMES):
        parameter[kv[0]] = map_parameters(kv[0], individual[i])
        if parameter[kv[0]] < 0:
            logger.info("Non-positive parameter in phenotype.")
            return n_objects * 3 * (1 + abs(parameter[kv[0]])),

    error = 0
    # iterate through multiple objects and compare to real values
    for model_number, model in enumerate(ref["models"]):
        # extract the filename and run the tweaker
        inputfile = model["name"]
        result = evaluate_tweaker(parameter, os.path.join("data", "Models", inputfile))

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
                    error += 1
                elif ref_a['grade'] == "C":
                    error += 5
                break
        # Add error if alignment is not found
        if not referred_flag:
            error += 5

        # Add minor weights for the value of the unprintablity
        # Return negative slope for negative and therefore invalid scores
        if result.unprintability < 0:
            error += 1 + 10 * abs(result.unprintability)

        # logit transformation with a turning point in (10, 1), low value for x=0 and a maximum of 3
        error += 3/(1 + np.exp(0.5*(10-result.unprintability)))
        # TODO decrease the latest error because unprintable orientations should have a high unprintablity

    return error,


# Define the genetic operations
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.60)
toolbox.register("select", tools.selTournament, tournsize=3)

# Create a population of 10 individuals
population = toolbox.population(n=20)

n_generations = 10
for gen in range(n_generations):
    print(f"Generation {gen+1} for {n_generations}:")
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.4)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        print(f"The phenotype {map_all_parameters(ind)} \t has a fitness of: {round(fit[0], 6)}")
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
    top = tools.selBest(population, k=1)[0]
    print(f"  Best phenotype of generation {gen}: {map_all_parameters(top)} with a fitness of {top.fitness.values[0]}.\n")

