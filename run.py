#
# Import all the dependencies
#
import numpy as np
from POET.solver import poet_solver

if __name__ == '__main__':

    X_train, y_train = None, None
    X_test, y_test = None, None
    path, version = None, None
    params = {
            "type": "eccentricity",
            "epochs": 30,
            "batch_size": 100,
            "verbose": 2,
            "retrain": False,
            "threshold": 2000,
            "path_to_store": path,
            "version": version
            }
    poet = poet_solver.POET_IC_Solver(**params)
    poet.store_data(X_train=X_train, y_train=y_train)
    poet.fit_evaluate(X_test=X_test)
