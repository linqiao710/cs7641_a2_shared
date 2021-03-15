"""
RHC NN training on HTRU2 data
"""
# Adapted from https://github.com/JonathanTay/CS-7641-assignment-2/blob/master/NN1.py
import sys
from itertools import product
import csv
import os
import time

import random as rand
from shared import Instance

sys.path.append("./ABAGAIL.jar")
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem
from func.nn.backprop import RPROPUpdateRule
import opt.RandomizedHillClimbing as RandomizedHillClimbing
from func.nn.activation import RELU

# Network parameters found "optimal" in Assignment 1
INPUT_LAYER = 10
HIDDEN_LAYER1 = 10
HIDDEN_LAYER2 = 10
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 2
OUTPUT_DIRECTORY = "output"
OUTFILE = OUTPUT_DIRECTORY + '/NN_OUTPUT/{}/NN_{}_{}_LOG.csv'
DS_NAME = 'ABALONEData'
TEST_DATA_FILE = 'data/{}_test.csv'.format(DS_NAME)
TRAIN_DATA_FILE = 'data/{}_train.csv'.format(DS_NAME)
VALIDATE_DATA_FILE = 'data/{}_validate.csv'.format(DS_NAME)


def initialize_instances(infile):
    """Read the given CSV data into a list of instances."""
    instances = []

    # Read in the CSV file
    with open(infile, "r") as dat:
        reader = csv.reader(dat)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            # TODO: Set to <= 0 to handle 0/1 labels and not just -1/1?
            instance.setLabel(Instance(0 if float(row[-1]) < 0 else 1))
            instances.append(instance)

    return instances


def f1_score(labels, predicted):
    get_count = lambda x: sum([1 for i in x if i is True])

    tp = get_count([predicted[i] == x and x == 1.0 for i, x in enumerate(labels)])
    tn = get_count([predicted[i] == x and x == 0.0 for i, x in enumerate(labels)])
    fp = get_count([predicted[i] == 1.0 and x == 0.0 for i, x in enumerate(labels)])
    fn = get_count([predicted[i] == 0.0 and x == 1.0 for i, x in enumerate(labels)])

    if tp == 0:
        return 0, 0, 0

    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return precision, recall, 0.0
    return precision, recall, f1


def error_on_data_set(network, ds, measure, ugh=False):
    N = len(ds)
    error = 0.
    correct = 0
    incorrect = 0
    actuals = []
    predicteds = []
    for instance in ds:
        network.setInputValues(instance.getData())
        network.run()
        actual = instance.getLabel().getContinuous()
        predicted = network.getOutputValues().get(0)
        predicted = max(min(predicted, 1), 0)
        if ugh:
            print("label: {}".format(instance.getLabel()))
            print("actual: {}, predicted: {}".format(actual, predicted))

        predicteds.append(round(predicted))
        actuals.append(max(min(actual, 1), 0))
        if abs(predicted - actual) < 0.5:
            correct += 1
            if ugh:
                print("CORRECT")
        else:
            incorrect += 1
            if ugh:
                print("INCORRECT")
        output = instance.getLabel()
        output_values = network.getOutputValues()
        example = Instance(output_values, Instance(output_values.get(0)))
        error += measure.value(output, example)
        if ugh:
            print("error: {}".format(measure.value(output, example)))

    MSE = error / float(N)
    acc = correct / float(correct + incorrect)
    precision, recall, f1 = f1_score(actuals, predicteds)
    if ugh:
        print("MSE: {}, acc: {}, f1: {} (precision: {}, recall: {})".format(MSE, acc, f1, precision, recall))
        import sys
        sys.exit(0)

    return MSE, acc, f1


def train(oa, network, oaName, training_ints, measure, training_iterations, outfile):
    """Train a given network on a set of instances.
    """
    print('========' + str(oaName))
    times = [0]
    for iteration in range(training_iterations):
        start = time.clock()
        oa.train()
        elapsed = time.clock() - start
        times.append(times[-1] + elapsed)
        if iteration % 10 == 0:
            MSE_trg, acc_trg, f1_trg = error_on_data_set(network, training_ints, measure)
            txt = '{},{},{},{},{}\n'.format(iteration, MSE_trg, acc_trg, f1_trg, times[-1])
            print(txt)
            with open(outfile, 'a+') as f:
                f.write(txt)


def main(learningRate, fname, restart):
    """Run this experiment"""
    training_ints = initialize_instances(TRAIN_DATA_FILE)

    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(training_ints)
    relu = RELU()
    # 50 and 0.000001 are the defaults from RPROPUpdateRule.java
    rule = RPROPUpdateRule(learningRate, learningRate, learningRate)
    classification_network = factory.createClassificationNetwork(
        [INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, OUTPUT_LAYER], relu)
    nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
    oa = RandomizedHillClimbing(nnop, restart)
    train(oa, classification_network, 'RHC', training_ints, measure, TRAINING_ITERATIONS, fname)


if __name__ == "__main__":
    seed = 42
    print("Using seed {}".format(seed))
    rand.seed(seed)
    learningRate_list = [0.00064, 0.0064, 0.064, 0.64]
    restart_list = [10, 60, 110]
    for learningRate, restart in product([0.00064], restart_list):
        fname = OUTFILE.format('RHC', 'RHC_{}_{}'.format("restart", restart), str(1))
        with open(fname, 'w') as f:
            f.write(
                '{},{},{},{},{}\n'.format('iterations', 'loss', 'acc_trg', 'f1_trg', 'time'))

        main(learningRate, fname, restart)
