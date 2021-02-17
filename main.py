import numpy as np
from numpy.random import randint


class GenAlgorithm:

    def __init__(self, k=4, reduce_k=False, pop_size=132, offspring_ratio=1, mutation_rate=0.1, max_iterations=4000,
                 stop_with_mean=False, nearest_neighbors=True, localsearch=True, localsearch_rate=0.5,
                 diversity_promotion=False, introduce_new=True, diversity_when_introduce=True, fitness_sharing=True,
                 efficient_large_tours=True, debug=False):
        self.k = k
        self.reduce_k = reduce_k  # if k has to be decreased in every iteration
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.offspring_size = int(pop_size * offspring_ratio)
        self.max_iterations = max_iterations
        self.stop_with_mean = stop_with_mean  # if True, it will stop if the mean is the same for a number of iterations
        self.nearest_neighbors = nearest_neighbors  # if True, will use nearest neighbors heuristic for the population initialization
        self.localsearch = localsearch  # if True, will apply local search
        self.localsearch_rate = localsearch_rate  # only used if localsearch=True.
        self.diversity_promotion = diversity_promotion  # if True, will apply crowding to the elimination
        self.introduce_new = introduce_new  # if True, will replace part of the population with new one,
                                        # if the mean is the same than the best
        self.diversity_when_introduce = diversity_when_introduce  # if diversity promotion in elimination has to be activated for a
                                    # number of iterations after introducing new population (both after
                                    # introduce_new and the initialization). Only used if diversity_promotion=False.
        self.fitness_sharing = fitness_sharing  # if fitness sharing has to be applied in selection
        self.efficient_large_tours = efficient_large_tours  # if True, will set diversity_when_introduce
                                                            # and diversity_promotion to False for large tours, to make
                                                            # the algorithm more exploitative.
        self.debug = debug

    def evaluate(self, population, distance_matrix):
        """ Calculate the distance of all paths in the given population, giving the objective function (fitness).
        Less is better """
        return distance_matrix[population[:, -1], population[:, 0]] + \
               distance_matrix[population[:, :-1], population[:, 1:]].sum(1)

    def initialize(self, pop_size, n_cities, seed=None, nearest_neighbors=False, distance_matrix=None,
                   pop_portion_nearest=0.1):
        """ Randomly initialize a population of paths. @seed can be an int (will seed the randomizer) or None. If
        @nearest_neighbors is True, a portion of the population will be initialized using nearest neighbors greedy
        heuristic, which consists in selecting one city, then the one with shortest path to it, and so on. It then needs
        @distance_matrix to not be None. @pop_portion_nearest (only used if nearest_neighbors=True) is
        the portion of the population that will be initialized with nearest neighbors."""

        if (nearest_neighbors):
            pop_size_nearest = int(pop_size * pop_portion_nearest)
            pop_size = pop_size - pop_size_nearest

            nearest_pop = np.zeros((pop_size_nearest, n_cities), dtype=int)
            # select a vertex/city randomly as the first
            nearest_pop[:, 0] = randint(0, n_cities, pop_size_nearest)
            max_ = distance_matrix.max() + 1
            for i in range(pop_size_nearest):
                distance_matrix2 = distance_matrix.copy()
                distance_matrix2[:,
                nearest_pop[i, 0]] = max_  # so that the first city is not selected again as minimumm
                for c in range(n_cities - 1):
                    # select city closest to current (arc with least weight)
                    closest = np.argmin(distance_matrix2[nearest_pop[i, c]])
                    nearest_pop[i, c + 1] = closest
                    distance_matrix2[:, closest] = max_  # so that this city is not selected again as minimum

                # check that all cities are in ind
                unique, counts = np.unique(nearest_pop[i], return_counts=True)
                if unique.size != n_cities:
                    # some cities are repeated and some missing, maybe because some distances in the distance_matrix are equal
                    isin = np.isin(np.arange(n_cities), nearest_pop[i])
                    notin = np.where(isin == False)
                    for notin2 in notin:
                        # replace by a repeated value
                        repeated = np.where(counts > 1)[0][0]
                        replace = np.where(nearest_pop[i] == unique[repeated])[0][0]
                        nearest_pop[i][replace] = notin2
                        counts[repeated] -= 1

        # initialize randomly
        init_pop = np.zeros((pop_size, n_cities), dtype=int)
        if (seed is not None):
            np.random.seed(seed)  # fix randomizer
        for i in range(pop_size):
            init_pop[i] = np.random.permutation(n_cities)
        if (seed is not None):
            np.random.seed(None)

        if (nearest_neighbors):
            init_pop = np.concatenate((init_pop, nearest_pop), axis=0)

        return init_pop

    def order_crossover(self, p1, p2, n_cities):
        """ Perform order crossover with 2 given parents. """
        cpa = randint(0, n_cities - 1)
        cpb = randint(0, n_cities - 1)
        if cpa > cpb:
            temp = cpb
            cpb = cpa
            cpa = temp
        offspring = list(p1[cpa:cpb])
        i = cpb
        while len(offspring) != n_cities:
            if not (p2[i] in offspring):
                offspring.append(p2[i])
            i = i + 1
            if i == n_cities:
                i = 0
        return np.array(offspring)

    def dpx_crossover(self, p1, p2, n_cities, distance_matrix):
        """ Distance preserving crossover """

        def find_c2(c1, parent):
            where = np.where(parent == c1)[0][0]
            if where == n_cities - 1:
                return parent[0]
            return parent[where + 1]

        offspring = np.ones(n_cities, int) * -1
        available_cities_complete = np.arange(n_cities)
        available_cities_bool = np.ones(n_cities, int)
        off_i = 0
        off_i_same = 0

        # add to offspring the edges that are in both parents (note: in specific occasions, an edge be added that is only in one)
        # (does not contemplate inversed edge)
        for i in range(n_cities - 1):
            c1 = p1[i]
            c2_in_p1 = p1[i + 1]
            if c2_in_p1 == find_c2(c1, p2):
                # same edge in both parents. Keep edge
                offspring[off_i] = c1
                offspring[off_i + 1] = c2_in_p1
                available_cities_bool[c1] = 0
                available_cities_bool[c2_in_p1] = 0
                off_i += 1
                off_i_same = 1
            else:
                # edge doesn't appear in both parents. Delete
                off_i += off_i_same
                off_i_same = 0

        where = np.where(offspring == -1)[0]

        if off_i == 0:
            # no edge found shared by both parents. Perform order_crossover
            offspring = self.order_crossover(p1, p2, n_cities)

        elif len(where) > 0:
            # still cities to fill. Else, means that p1=p2.
            # add rest of the edges. c2_greedy will be the city with least distance to c1, provided that
            # c1-c2_greedy edge does not appear in either parent.
            off_i = where[0] - 1
            for j in range(off_i, n_cities - 1):
                c1 = offspring[j]
                available_cities_bool[c1] = 0
                c2_in_p1 = find_c2(c1, p1)
                c2_in_p2 = find_c2(c1, p2)
                available_cities_bool_temp = available_cities_bool.copy()
                available_cities_bool_temp[c2_in_p1] = 0  # do not allow edge that appears in p1
                available_cities_bool_temp[c2_in_p2] = 0  # do not allow edge that appears in p2
                available_cities_temp = available_cities_complete[available_cities_bool_temp == 1]
                if available_cities_temp.size == 0:
                    # allow edges that appear in some parent
                    available_cities_temp = available_cities_complete[available_cities_bool == 1]
                closest_idx = np.argmin(distance_matrix[c1][available_cities_temp])
                c2_greedy = available_cities_temp[closest_idx]
                offspring[j + 1] = c2_greedy
                available_cities_bool[c2_greedy] = 0

        return offspring

    def mutate(self, pop, n_cities, pop_size):
        """ Randomly swap two pairs of cities, on a subset (depending on mutation_rate) of the population. A pair
        of cities are two connected cities. @pop is the entire population. """

        # choose which individuals to mutate. Use permutation to avoid repetition of individuals
        n_mutate = int(self.mutation_rate * pop_size)
        i_mutate = np.random.permutation(np.arange(pop_size))[:n_mutate]

        # randomly choose cities to swap, different for each individual. Note: it's possible that some pair of cities
        # are the same, and also that the edge is not maintained (if rands[x,1] = rands[x,0] +- 1)
        rands1 = randint(0, n_cities,
                         size=(n_mutate, 2))  # indexes of cities that will be swapped, 1st cities of the pair
        rands2 = rands1 + 1  # 2nd cities of the pair
        rands2[rands2 >= n_cities] = 0  # close cycle

        # swap
        pop[i_mutate, rands1[:, 0]], pop[i_mutate, rands1[:, 1]] = pop[i_mutate, rands1[:, 1]], pop[
            i_mutate, rands1[:, 0]]
        pop[i_mutate, rands2[:, 0]], pop[i_mutate, rands2[:, 1]] = pop[i_mutate, rands2[:, 1]], pop[
            i_mutate, rands2[:, 0]]

        return pop

    def eliminate(self, population, fitness, lambda_):
        """ Keeps best lambda_ individuals from the population. @fitness is the fitness of each individual """
        sorted_fitnesses = np.argsort(fitness)
        population2 = population[sorted_fitnesses[:lambda_]]
        fitness2 = fitness[sorted_fitnesses[:lambda_]]
        return population2, fitness2

    def eliminate_with_diversity(self, population, fitness, lambda_, n_cities):
        """ Use Crowding diversity promotion technique for the elimination. Keeps best lambda_ individuals from
        the population, but for every promoted individual, the one more similar to it will be eliminated. """

        size = population.shape[0]

        # prepare for calculating diversity metric
        # make all individuals start from the city 0
        population = self._roll_individuals(population, size)

        # promote
        fitness_promoted = np.zeros(lambda_)
        promoted = np.zeros((lambda_, n_cities), dtype=int)
        for i in range(lambda_):
            # promote individual
            best_fit = np.argmin(fitness)
            best_ind = population[best_fit]
            promoted[i] = best_ind
            fitness_promoted[i] = fitness[best_fit]
            population = np.delete(population, best_fit, axis=0)
            fitness = np.delete(fitness, best_fit, axis=0)

            # calculate diversity of current best individual compared to the rest, using hamming distance
            #  (number of cities in same index that are different; the higher, the more diverse).
            diff = best_ind - population
            diff[diff != 0] = 1
            diversity = np.sum(diff,
                               axis=1)  # diversity of current individual with respect to each of the others. "distance"

            # delete the individual with less diversity with respect to current
            more_similar = np.argmin(diversity)
            population = np.delete(population, more_similar, axis=0)
            fitness = np.delete(fitness, more_similar, axis=0)

        return promoted, fitness_promoted

    def select(self, population, fitness, k, pop_size):
        """ Select an individual from the given population based on k-tournament selection """
        indices = randint(0, pop_size, k)
        winner_index = indices[fitness[indices].argmin()]
        return population[winner_index]

    def fitness_sharing_objval(self, population, fitness, n_cities, pop_size):
        """ Returns the fitness of the population with fitness sharing applied. """

        radius_prop = 0.16  # min distance (diversity) required to apply 1+beta. Proportional to n_cities
        alpha = 3.2

        population = self._roll_individuals(population, pop_size)

        # calculate new fitness
        min_radius = min(n_cities, 4)
        radius = max(int(radius_prop * n_cities), min_radius)
        div_fitness = np.zeros(pop_size)
        arange = np.arange(pop_size)
        for i in range(pop_size):
            # calculate diversity of current best individual compared to the rest, using hamming distance
            #  (number of cities in same index that are different; the higher, the more diverse).
            diff = population[i] - population[arange != i]
            diff[diff != 0] = 1
            diversity = np.sum(diff,
                               axis=1)  # diversity of current individual with respect to each of the others. "distance"

            # apply 1+beta function with individuals with less diversity than radius. Note: sign is +1 for all fitnesses
            if np.any(diversity <= radius):
                div_fitness[i] = fitness[i] * (np.sum(1 - np.power(diversity[diversity <= radius] / radius, alpha)))
            else:
                div_fitness[i] = fitness[i]
        return div_fitness

    def local_search(self, ind, n_cities, distance_matrix, localsearch_rate, method=1):
        """ Perform 1-opt local search. It can use 4 possible methods:
            @method=1. Neighbors will be created by swapping two consecutive cities.
            @method=2. Not strictly a local search method, as it includes some random component. Creates an arbitrary
                number of random neighbors, instead of searching through all the possible neighbors. Neighbors will be
                created by swapping two cities (not necessarily consecutive).
            @method=3. Same as method 2, but neighbors will be created by inverting a random subpath.
            @method=4. rn times, select a random city and replace the city next to it for the closest city.
        """

        if (np.random.uniform() < localsearch_rate):

            if method == 1:
                # Create all neighbors possible by swapping two consecutive cities
                n_neighbors = n_cities
                neighbors = np.ones((n_neighbors + 1, n_cities), int) * ind  # add ind as well
                n = 1
                for i in range(0, n_cities - 1):
                    neighbors[n, i], neighbors[n, i + 1] = neighbors[n, i + 1], neighbors[n, i]
                    n += 1
                neighbors[n, 0], neighbors[n, -1] = neighbors[n, -1], neighbors[n, 0]

            elif method == 2:

                # Create neighbors by swapping two cities (not only consecutive)
                n_neighbors = n_cities * 20
                neighbors = np.ones((n_neighbors, n_cities), int) * ind
                rands = randint(0, n_cities, size=(n_neighbors, 2))  # indexes of cities that will be swapped
                arange = np.arange(n_neighbors)
                # Randomly swap to cities of all the neighbos. Note: some indexes may be the same (no swapping),
                #   and some neighbors may be the same
                neighbors[arange, rands[:, 0]], neighbors[arange, rands[:, 1]] = neighbors[arange, rands[:, 1]], \
                                                                                 neighbors[arange, rands[:, 0]]

                # add the original individual
                neighbors = np.concatenate((neighbors, [ind]))

            elif method == 3:

                # Create neighbors by inverting a subpath
                n_neighbors = 102
                neighbors = np.ones((n_neighbors + 1, n_cities), int) * ind  # add ind as well
                for i in range(n_neighbors):
                    cpa = randint(0, n_cities)
                    cpb = randint(cpa,
                                  n_cities + cpa)  # allow to surpass n_cities (to give uniform probability to all subpaths)
                    if cpb >= n_cities:
                        # cycle, so that it gets subpath between cpa and cpb
                        neighbors[i] = np.roll(neighbors[i], -cpa)
                        cpb = cpb - cpa
                        cpa = 0
                    neighbors[i, cpa:cpb] = np.flip(neighbors[i, cpa:cpb])

            elif method == 4:

                # rn times, select a random city and replace the city next to it for the closest city
                ind2 = ind.copy()
                rn = int(n_cities / 10)
                available_cities_bool = np.ones(n_cities, int)
                available_cities = np.arange(n_cities)
                for i in range(rn):
                    pos1 = randint(n_cities - 1)
                    pos2 = pos1 + 1
                    city1 = ind2[pos1]
                    available_cities_bool_temp = available_cities_bool.copy()
                    available_cities_bool_temp[city1] = 0  # do not allow this same city to turn as min
                    available_cities_temp = available_cities[available_cities_bool_temp == 1]
                    closest_idx = np.argmin(distance_matrix[city1][available_cities_temp])
                    city2 = available_cities_temp[closest_idx]
                    where = np.where(ind2 == city2)[0][0]
                    ind2[pos2], ind2[where] = city2, ind2[pos2]
                neighbors = np.array([ind, ind2])

            # check best neighbor (including the individual)
            fitness = self.evaluate(neighbors, distance_matrix)
            best_fitness_i = np.argmin(fitness)
            best_ind = neighbors[best_fitness_i]

            return best_ind

        return ind

    def introduce_new_population(self, population, fitness, pop_size, n_cities):
        """ Population re-initiation. Thought to be called when the population is stuck at a local minima or hasn't
        changed in several iterations. Keeps a subset with the best individuals, the rest is replaced by a new
        initialization of individuals. This way, the new population will theoretically have more diversity. """

        keep_best_p = 0.05  # percentage of num of best individuals to keep

        keep_best = int(pop_size * keep_best_p)
        sorted_fitnesses = np.argsort(fitness)
        keep_population = population[sorted_fitnesses[:keep_best]]

        create_new_n = pop_size - keep_best
        # do not use nearest neighbors for this initialization, as, in case it had to be used, it was already used
        # in the first initialization
        new_population = self.initialize(create_new_n, n_cities, nearest_neighbors=False)

        return np.concatenate((keep_population, new_population))

    def _roll_individuals(self, population, n_individuals):
        """ Shift cities in individuals so that they start from the city 0. """
        for i in range(n_individuals):
            population[i] = np.roll(population[i], -np.where(population[i] == 0)[0][0])
        return population

    def get_statistics(self, population, fitness):
        """ Calculate the mean fitness, best fitness and best individual of the given population"""
        mean_fitness = np.mean(fitness)
        i = np.argmin(fitness)
        best_fitness = fitness[i]
        best_individual = self._roll_individuals(np.array([population[i]]), 1)[
            0]  # shift best individual so that the first city is the 0
        return mean_fitness, best_fitness, best_individual

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        n_cities = distanceMatrix.shape[0]

        method = 3 #local search method

        if self.debug or self.stop_with_mean:
            # Initialize arrays to return for visualization
            means = []
            bests = []
            iterations = []
            same_with_mean_var = 0.0000001  # minimum variance for the means to be different
            same_with_mean_i = 0  # number of iterations the mean has been the same
            stop_with_mean_max = 100  # max consecutive iterations that the mean can be the same before it stops

        if self.efficient_large_tours:
            # for very large tours, it's better to try more exploitative techniques. Will deactivate some
            #  that are more explorative
            min_large_tour = 500  # minimum number of cities considered to be large
            if n_cities >= min_large_tour:
                self.diversity_promotion = False
                self.diversity_when_introduce = False

        if self.introduce_new:
            same_with_mean_var = 0.0000001  # minimum variance for the best and mean to be different

        if self.reduce_k:
            # determine how much will k be reduced in each iteration
            if n_cities < 100:
                k_decrease = 0.007
            elif n_cities < 700:
                k_decrease = 0.03
            else:
                k_decrease = 3
            min_k = 3  # minimum value the k can have
            k_float = float(self.k)

        if self.diversity_promotion:
            diversity_promotion_active = True
        elif self.diversity_when_introduce:
            # only activate diversity promotion for the first #iterations_div_max after introducing new population, both
            # with introduce_new_population and
            iterations_div_passed = 0  # number of iterations diversity_promotion_active has been True
            diversity_promotion_active = True
            if n_cities < 300:
                iterations_div_max = 20  # number of iterations diversity promotion will be active each time
            elif n_cities < 700:
                iterations_div_max = 3
            else:
                iterations_div_max = 1
        else:
            diversity_promotion_active = False

        if self.nearest_neighbors:
            if n_cities > 500:
                pop_portion_nearest = 0.3
            else:
                pop_portion_nearest = 0.16
        else:
            pop_portion_nearest = None

        # Initialization
        iteration = 0
        pop = self.initialize(self.pop_size, n_cities, nearest_neighbors=self.nearest_neighbors,
                              distance_matrix=distanceMatrix, pop_portion_nearest=pop_portion_nearest)
        fit = self.evaluate(pop, distanceMatrix)
        continue_iterating = True
        bestObjectiveOverall = None  # fitness of best individual found in the entire execution
        bestSolutionOverall = None

        # optimization loop
        while continue_iterating:

            if self.reduce_k and iteration > 0:
                k_float = max(min_k, k_float - k_decrease)
                self.k = int(k_float)

            if self.fitness_sharing:
                # calculate fitness using fitness sharing
                fit = self.fitness_sharing_objval(pop, fit, n_cities, self.pop_size)

            # Selection + crossover (+ local search)
            offspring = np.zeros((self.offspring_size, n_cities), dtype=int)
            for i in range(self.offspring_size):
                p1, p2 = self.select(pop, fit, self.k, self.pop_size), self.select(pop, fit, self.k, self.pop_size)
                offspring[i] = self.order_crossover(p1, p2, n_cities)
                if self.localsearch:
                    # local search (only to offsprings)
                    offspring[i] = self.local_search(offspring[i], n_cities, distanceMatrix, self.localsearch_rate,
                                                     method=method)
            joined_pop = np.concatenate((pop, offspring))

            # Mutation
            mutated_pop = self.mutate(joined_pop, n_cities, self.pop_size)

            # Elimination
            if not self.diversity_promotion and diversity_promotion_active and self.diversity_when_introduce:
                # check whether diversity promotion has to be deactivated
                iterations_div_passed += 1
                if iterations_div_passed >= iterations_div_max:
                    diversity_promotion_active = False
            if diversity_promotion_active:
                pop, fit = self.eliminate_with_diversity(mutated_pop, self.evaluate(mutated_pop, distanceMatrix),
                                                         self.pop_size, n_cities)
            else:
                pop, fit = self.eliminate(mutated_pop, self.evaluate(mutated_pop, distanceMatrix), self.pop_size)

            if (self.localsearch):
                # perform local search to current best individual
                best_fit_i = np.argmin(fit)
                pop[best_fit_i] = self.local_search(pop[best_fit_i], n_cities, distanceMatrix, localsearch_rate=1,
                                                    method=method)
                fit[best_fit_i] = self.evaluate(np.array([pop[best_fit_i]]), distanceMatrix)

            meanObjective, bestObjective, bestSolution = self.get_statistics(pop, fit)

            # keep best individual. Note that it won't necessarily be part of the population
            if iteration == 0:
                bestObjectiveOverall = bestObjective
                bestSolutionOverall = bestSolution
            elif bestObjectiveOverall > bestObjective:
                bestObjectiveOverall = bestObjective
                bestSolutionOverall = bestSolution

            iteration += 1

            if self.max_iterations is not None and iteration >= self.max_iterations:
                continue_iterating = False

            if self.debug or self.stop_with_mean:
                if self.debug:
                    print("i: {}, mean: {}, best: {}, bestOverall: {}".format(iteration, meanObjective,
                                                                              bestObjective, bestObjectiveOverall))
                    # Save values to plot (visualization)
                    bests.append(bestObjectiveOverall)
                    iterations.append(iteration)
                means.append(meanObjective)

                if self.stop_with_mean:
                    # Check if the mean has been (approximately) the same for the last iterations
                    if (means[-1] == np.inf):
                        curr_mean_var1 = curr_mean_var2 = means[-1]
                    else:
                        curr_mean_var1 = means[-1] + means[-1] * same_with_mean_var
                        curr_mean_var2 = means[-1] - means[-1] * same_with_mean_var
                    if (iteration >= 2 and curr_mean_var2 <= means[-2] <= curr_mean_var1):
                        same_with_mean_i += 1
                    else:
                        same_with_mean_i = 0
                    if self.stop_with_mean and same_with_mean_i >= stop_with_mean_max:
                        # stopping criterion
                        continue_iterating = False

            if self.introduce_new:
                # check if mean value is same as best value, in which case will introduce new population
                if (meanObjective == np.inf):
                    curr_mean_var1 = curr_mean_var2 = meanObjective
                else:
                    curr_mean_var1 = meanObjective + meanObjective * same_with_mean_var
                    curr_mean_var2 = meanObjective - meanObjective * same_with_mean_var
                if curr_mean_var2 <= bestObjective <= curr_mean_var1:
                    # introduce new population, intended to diversify again
                    pop = self.introduce_new_population(pop, fit, self.pop_size, n_cities)
                    fit = self.evaluate(pop, distanceMatrix)
                    if not self.diversity_promotion and self.diversity_when_introduce:
                        # temporally activate diversity promotion
                        diversity_promotion_active = True
                        iterations_div_passed = 0


        # Your code here.
        if (self.debug):
            return means, bests, iterations, bestSolutionOverall, timeLeft, fit
        return 0

#About filename:
#Should be a .csv, with a square matrix, where each row represents a city and each column a city. The same cities should be in the rows and in
# the columns and ordered the same way. Each value will be the distance between the x-th city and the y-th city, therefore the diagonal should be zeros.

GenAlgorithm.optimize(filename)

