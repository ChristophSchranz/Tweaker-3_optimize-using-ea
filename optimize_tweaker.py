# This script tests the installation of the deap-project that is used for ea
# https://github.com/deap/deap

import random
import numpy as np
from deap import creator, base, tools, algorithms
from tweaker_phenotype import evaluate_tweaker, map_all_parameters

# Set to a minimization problem
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Draw random variables for the initial population
toolbox.register("attr_bool", random.random)

# Length of the chromosome is specified with n
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=7)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the genetic operations
toolbox.register("evaluate", evaluate_tweaker)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Create a population of 10 individuals
population = toolbox.population(n=20)

n_generations = 10
for gen in range(n_generations):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.4)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        print(f"The phenotype {map_all_parameters(ind)} \t has a fitness of: {round(fit[0], 6)}")
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
    top = tools.selBest(population, k=1)[0]
    print(f"  Best phenotype of generation {gen}: {map_all_parameters(top)} with a fitness of {top.fitness.values[0]}.\n")

