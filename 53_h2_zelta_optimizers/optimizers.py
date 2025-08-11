
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import ParameterGrid
import random
from deap import base, creator, tools
from sklearn.gaussian_process import GaussianProcessRegressor



def genetic_optimizer(objective_func, bounds, population_size=50, generations=100, cxpb=0.5, mutpb=0.2):
    # Define the individual and fitness function
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimization problem
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Initialize the toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_float", lambda: [random.uniform(b[0], b[1]) for b in bounds])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register the evaluation, selection, crossover, and mutation functions
    toolbox.register("evaluate", lambda ind: (objective_func(ind),))
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create the initial population
    population = toolbox.population(n=population_size)

    # Evolution process
    for gen in range(generations):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the old population with the offspring
        population[:] = offspring

    # Get the best individual from the population
    best_individual = tools.selBest(population, 1)[0]
    return best_individual, objective_func(best_individual)



def grid_search_optimizer(objective_func, param_grid):
    grid = list(ParameterGrid(param_grid))
    best_score = float('inf')
    best_solution = None
    for params in grid:
        score = objective_func(params)
        if score < best_score:
            best_score = score
            best_solution = params
    return best_solution, best_score

def random_search_optimizer(objective_func, bounds, n_iter=100):
    best_score = float('inf')
    best_solution = None
    for _ in range(n_iter):
        candidate = [np.random.uniform(b[0], b[1]) for b in bounds]
        score = objective_func(candidate)
        if score < best_score:
            best_score = score
            best_solution = candidate
    return best_solution, best_score




def bayesian_optimizer(objective_func, bounds, n_iters=25):
    kernel = Matern()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    X_sample = np.array([np.random.uniform(b[0], b[1], 1) for b in bounds])
    Y_sample = np.array([objective_func(x) for x in X_sample])

    def acquisition(X, X_sample, Y_sample, gp):
        mu, sigma = gp.predict(X, return_std=True)
        opt_value = np.max(Y_sample)
        with np.errstate(divide='warn'):
            Z = (mu - opt_value) / sigma
            return (mu - opt_value) * norm.cdf(Z) + sigma * norm.pdf(Z)

    for _ in range(n_iters):
        gp.fit(X_sample, Y_sample)
        X_next = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(100, len(bounds)))
        X_next = X_next[np.argmax(acquisition(X_next, X_sample, Y_sample, gp))]
        Y_next = objective_func(X_next)
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.vstack((Y_sample, Y_next))

    best_index = np.argmin(Y_sample)
    return X_sample[best_index], Y_sample[best_index]

