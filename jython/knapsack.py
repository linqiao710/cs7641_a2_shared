import time
from array import array
from itertools import product
from time import clock

import sys

from java.lang import Math

from dist import DiscreteDependencyTree
from opt import SimulatedAnnealing
from opt.ga import DiscreteChangeOneMutation, SingleCrossOver, GenericGeneticAlgorithmProblem, StandardGeneticAlgorithm
from opt.prob import GenericProbabilisticOptimizationProblem, MIMIC

sys.path.append("./ABAGAIL.jar")

import java.util.Random as Random

from shared import ConvergenceTrainer
from opt.example import KnapsackEvaluationFunction
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor

import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.RandomizedHillClimbing as RandomizedHillClimbing

random = Random()
maxIters = [2, int(2e4 + 1)]
numTrials = 5
# Problem Sizes - NUM_ITEMS
N_list = [80, 90, 100, 110]

OUTPUT_DIRECTORY = "output"
outfile = OUTPUT_DIRECTORY + '/KNAPSACK/{}/KNAPSACK_{}_{}_LOG.csv'

# The number of copies each
COPIES_EACH = 6
# The maximum weight for a single element
MAX_WEIGHT = 50
# The maximum volume for a single element
MAX_VOLUME = 50

# MIMIC
sample_list = [100, 200, 300, 400]
keepRate_list = [0.2, 0.3, 0.4, 0.5]
for t in range(numTrials):
    for samples, keepRate, m, N in product([200], [0.2], [0.9], N_list):
        fname = outfile.format('MIMIC', 'MIMIC_{}_{}'.format("problemSizes", N), str(t + 1))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals\n')

        keep = int(samples*keepRate)

        # The volume of the knapsack
        KNAPSACK_VOLUME = MAX_VOLUME * N * COPIES_EACH * .4

        # create copies
        fill = [COPIES_EACH] * N
        copies = array('i', fill)

        # create weights and volumes
        fill = [0] * N
        weights = array('d', fill)
        volumes = array('d', fill)
        for i in range(0, N):
            weights[i] = random.nextDouble() * MAX_WEIGHT
            volumes[i] = random.nextDouble() * MAX_VOLUME

        # create range
        fill = [COPIES_EACH + 1] * N
        ranges = array('i', fill)

        ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
        odd = DiscreteUniformDistribution(ranges)
        df = DiscreteDependencyTree(m, ranges)
        pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
        mimic = MIMIC(samples, keep, pop)
        fit = ConvergenceTrainer(mimic)

        times = [0]
        for i in range(0, maxIters[0]):
            start = clock()
            fit.train()
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            score = ef.value(mimic.getOptimal())
            fevals = ef.fEvals
            ef.fEvals -= 1
            st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
            print(st)
            with open(fname, 'a') as f:
                f.write(st)

# RHC
restart_list = [20 ,40, 60, 80]
for t in range(numTrials):
    for restart, N in product([20], N_list):
        fname = outfile.format('RHC', 'RHC_{}_{}'.format("problemSize", N), str(t + 1))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals\n')

        # The volume of the knapsack
        KNAPSACK_VOLUME = MAX_VOLUME * N * COPIES_EACH * .4

        # create copies
        fill = [COPIES_EACH] * N
        copies = array('i', fill)

        # create weights and volumes
        fill = [0] * N
        weights = array('d', fill)
        volumes = array('d', fill)
        for i in range(0, N):
            weights[i] = random.nextDouble() * MAX_WEIGHT
            volumes[i] = random.nextDouble() * MAX_VOLUME

        # create range
        fill = [COPIES_EACH + 1] * N
        ranges = array('i', fill)

        ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        rhc = RandomizedHillClimbing(hcp, restart)
        fit = ConvergenceTrainer(rhc)

        times = [0]
        for i in range(0, maxIters[0]):
            start = clock()
            fit.train()
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            score = ef.value(rhc.getOptimal())
            fevals = ef.fEvals
            ef.fEvals -= 1
            st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
            print(st)
            with open(fname, 'a') as f:
                f.write(st)

# SA
temperature_list = [1E1, 1E3, 1E5, 1E7, 1E9, 1E11]
CE_list = [0.35, 0.55, 0.75, 0.95]
for t in range(numTrials):
    for temperature, CE, N in product([1E1], [0.95], N_list):
        fname = outfile.format('SA', 'SA_{}_{}'.format("problemSizes", N), str(t + 1))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals\n')

        # The volume of the knapsack
        KNAPSACK_VOLUME = MAX_VOLUME * N * COPIES_EACH * .4

        # create copies
        fill = [COPIES_EACH] * N
        copies = array('i', fill)

        # create weights and volumes
        fill = [0] * N
        weights = array('d', fill)
        volumes = array('d', fill)
        for i in range(0, N):
            weights[i] = random.nextDouble() * MAX_WEIGHT
            volumes[i] = random.nextDouble() * MAX_VOLUME

        # create range
        fill = [COPIES_EACH + 1] * N
        ranges = array('i', fill)

        ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
        odd = DiscreteUniformDistribution(ranges)
        nf = DiscreteChangeOneNeighbor(ranges)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        sa = SimulatedAnnealing(temperature, CE, hcp)
        fit = ConvergenceTrainer(sa)

        times = [0]
        for i in range(0, maxIters[0]):

            start = clock()
            fit.train()
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            score = ef.value(sa.getOptimal())
            fevals = ef.fEvals
            ef.fEvals -= 1
            st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
            print(st)
            with open(fname, 'a') as f:
                f.write(st)

# GA
pop_list = [100, 200, 300, 400]
mateRate_list = [0.2, 0.4, 0.6, 0.8]
mutateRate_list = [0.2, 0.4, 0.6, 0.8]
for t in range(numTrials):
    for pop, mateRate, mutateRate, N in product([200], [0.8], [0.2], N_list):
        fname = outfile.format('GA', 'GA_{}_{}'.format("problemSizes", N), str(t + 1))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals\n')
        mate = int(pop*mateRate)
        mutate = int(pop*mutateRate)

        # The volume of the knapsack
        KNAPSACK_VOLUME = MAX_VOLUME * N * COPIES_EACH * .4

        # create copies
        fill = [COPIES_EACH] * N
        copies = array('i', fill)

        # create weights and volumes
        fill = [0] * N
        weights = array('d', fill)
        volumes = array('d', fill)
        for i in range(0, N):
            weights[i] = random.nextDouble() * MAX_WEIGHT
            volumes[i] = random.nextDouble() * MAX_VOLUME

        # create range
        fill = [COPIES_EACH + 1] * N
        ranges = array('i', fill)

        ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
        odd = DiscreteUniformDistribution(ranges)
        mf = DiscreteChangeOneMutation(ranges)
        cf = SingleCrossOver()
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
        ga = StandardGeneticAlgorithm(pop, mate, mutate, gap)
        fit = ConvergenceTrainer(ga)

        times = [0]
        for i in range(0, maxIters[0]):

            start = clock()
            fit.train()
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            score = ef.value(ga.getOptimal())
            fevals = ef.fEvals
            ef.fEvals -= 1
            st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
            print(st)
            with open(fname, 'a') as f:
                f.write(st)
