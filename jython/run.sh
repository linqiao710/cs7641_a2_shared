#!/bin/bash
# edit the classpath to to the location of your ABAGAIL jar file
#

export CLASSPATH=ABAGAIL.jar:$CLASSPATH

# four peaks
echo "Running four peaks test"
jython peaks4.py

# flipflop
echo "Running flipflop test"
jython flipflop.py

# knapsack
echo "Running knapsack test"
jython knapsack.py

 # NN-Backprop
echo "Running nn_bp.py"
jython nn_bp.py

## NN-RHC
echo "Running nn_rhc.py"
jython nn_rhc.py

## NN-SA
echo "Running nn_sa.py"
jython nn_sa.py

# NN-GA
echo "Running nn_ga.py"
jython nn_ga.py

## graphs
#echo "Creating Sample Graphs"
#python plot_data.py
