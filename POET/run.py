#
# Import all the dependencies
#
import os
import numpy as np
import pandas as pd
from POET.datasets import poet_data
from POET.solver import poet_solver

if __name__ == '__main__':
    #
    # Pass the training and test data samples
    #
    X, y = None, None
    X_test, y_test = None, None
    poet_data.store_data(X, y, type='eccentricity')
    X_train, y_train = poet_data.load_data(type='eccentricity', threshold=1000)
    #
    # Pass the parameters for the POET IC Solver
    #
    params = {
        "type": "eccentricity",
        "epochs": 20,
        "batch_size": 5,
        "verbose": 2,
        "threshold": 1000
    }
    #
    # Train and evaluate the NN model
    #
    poet = poet_solver.POET_IC_Solver(X_train=X_train, y_train=y_train, **params)
    poet.generate_model()
    poet.fit()
    poet.evaluate(X_test=X_test)