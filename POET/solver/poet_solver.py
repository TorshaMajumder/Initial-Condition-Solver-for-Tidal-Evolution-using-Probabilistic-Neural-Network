#
# Import all the dependencies
#
import os
import random
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

#
# Set all the parameters
#
random.seed(0)
tfpl = tfp.layers
tfd = tfp.distributions
#
# Create '/poet_output' folder if it does not exists already
#
if not os.path.exists('../poet_output'):
    os.makedirs('../poet_output')


class POET_IC_Solver(object):
    """
    Parameters
    ----------
    X_train: numpy ndarray
        training data set
    y_train: numpy ndarray
        training labels
    type: string
        type of initial condition (orbital period or eccentricity)
    epochs: int (default = 500)
        number of epochs to train the model
    batch_size: int (default = 100)
        number of samples per gradient update
    threshold: int (default = 1000)
        minimum number of training data samples
    verbose: int (default =2)
        'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

    """

    def __init__(self, X_train=None, y_train=None, type=None,
                 epochs=500, batch_size=100, verbose=2, threshold=1000):
        self.X_train = X_train
        self.y_train = y_train
        self.epochs = epochs
        self.verbose = verbose
        self.type = type
        self.batch_size = batch_size
        self.threshold = threshold
        self.model = None
        self.y_hat = None
        self.y_sd = None

        try:
            if self.type not in ["orbital_period", "eccentricity"]:
                raise ValueError(f"\n'{self.type}' is not a valid type!"
                                 f"\nPlease provide the type as - 'orbital_period' or 'eccentricity'")
        except Exception as e:
            print(e)
            exit()

        #
        # Create '/poet_output/{type}' folder if it does not exists already
        #
        if not os.path.exists(f'../poet_output/{self.type}'):
            os.makedirs(f'../poet_output/{self.type}')

    def log_loss(self, y_true, y_pred):
        return -y_pred.log_prob(y_true)

    def generate_model(self):
        #
        #
        # event shape: integer vector Tensor representing the shape
        # of single draw from this distribution
        event_shape = 1
        #
        # features: number of features from the training sample
        #
        features = self.X_train.shape[1]
        #
        # build the model using a independent normal distribution
        #
        self.model = tf.keras.Sequential([
                      tf.keras.layers.Dense(units=tfpl.IndependentNormal.params_size(event_shape),
                                            input_shape=(features,)),
                      tfpl.IndependentNormal(event_shape=event_shape)])
        return

    def fit(self):
        #
        # Fit the model using the X_train and y_train
        # Custom loss function used - log loss
        # Optimizer - Adam
        #
        try:
            if len(self.y_train) < self.threshold:
                raise ValueError(f"\nValueError: the training data size should be greater "
                                 f"than equals to the threshold value--{self.threshold} to begin training!")
        except Exception as e:
            print(f"\n{e}\n")
            print(f"\nLower bound of the estimate: None"
                  f"\nUpper bound of the estimate: None\n")
            exit()

        try:
            #
            self.model.compile(loss=self.log_loss, optimizer='adam')
            self.model.fit(self.X_train, self.y_train,
                           epochs=self.epochs,
                           batch_size=self.batch_size,
                           verbose=self.verbose)
            #
            # Save the model (model.h5) under the folder - /poet_output/{type}/
            #
            self.model.save(f"../poet_output/{self.type}/model.h5")
            print(f"\nThe POET_IC_Solver model is fitted!\n"
                  f"\nThe results are stored in --/poet_output/{self.type}/ folder!")
        except Exception as e:
            print(f"\nUnknown Exception: {e}\n")
        return

    def evaluate(self, X_test=None, y_test=None):
        """

        Parameters
        ----------
        X_test: numpy nd-array
            test data sample
        y_test: numpy nd-array (ignored)
            test labels

        Returns
        -------
        y_hat_lower: numpy nd-array
            lower bound of the actual estimate
        y_hat_upper: numpy nd-array
            upper bound of the actual estimate

        Notes: the results are stored as dictionary in folder - /poet_output/{type}/ - as results.pickle
        """
        #
        # Check if the data is a numpy nd-array
        #
        try:
            if isinstance(X_test, (np.ndarray, np.generic)):
                X_test = np.array(X_test)
            elif isinstance(X_test, pd.DataFrame):
                X_test = X_test.to_numpy()
            else:
                raise TypeError(f"\nTypeError: Unable to load the data. Expected a format as "
                            f"a numpy array or a pandas dataframe.\n")
            if y_test:
                if isinstance(y_test, (np.ndarray, np.generic)):
                    y_test = np.array(y_test)
                elif isinstance(y_test, pd.DataFrame):
                    y_test = y_test.to_numpy()
                else:
                    raise TypeError(f"\nTypeError: Unable to load the data. Expected a format as "
                                    f"a numpy array or a pandas dataframe.\n")
        except Exception as e:
            print(e)
            exit()
        try:
            if X_test.ndim < 2:
                X_test = X_test.reshape((1, X_test.shape[0]))
            if y_test:
                if y_test.ndim < 2:
                    y_test = y_test.reshape((1, y_test.shape[0]))
        except Exception as e:
            print(f"Exception Raised: {e}")
        #
        try:
            #
            # Declare all the variables
            #
            count = 0
            #
            # Load the stored model from the folder - /poet_output/{type}/
            #
            model = tf.keras.models.load_model(f"../poet_output/{self.type}/model.h5",
                                               custom_objects={'log_loss': self.log_loss})
            #
            # Calculate the mean and the std. deviation
            #
            self.y_hat = model(X_test).mean()
            self.y_sd = model(X_test).stddev()
            #
            # Calculate the lower and the upper bound of the original estimate
            #
            y_hat_lower = self.y_hat - 2 * self.y_sd
            y_hat_upper = self.y_hat + 2 * self.y_sd
            #
            #
            #
            if y_test:
                for i in range(len(y_test)):
                    if (y_test[i] >= y_hat_lower[i]) and (y_test[i] <= y_hat_upper[i]):
                        count += 1
                #
                # Calculate the accuracy of the model if y_test is provided
                #
                accuracy = (count/len(y_test))*100
            else:
                accuracy = None
            #
            # Calculate the log value of the ratio - (y_hat_upper/y_hat_lower)
            #
            with np.errstate(invalid='ignore'):
                log_dist = np.nanmean(np.log(y_hat_upper/y_hat_lower))
            #
            # Store the results as a dictionary
            #
            data = {"y_hat_lower": y_hat_lower, "y_hat_upper": y_hat_upper,
                    "accuracy": accuracy, "log_dist": log_dist,
                    "y_hat": self.y_hat, "y_sd": self.y_sd}
            #
            #
            #
            print(f"\nThe POET_IC_Solver model is evaluated!\n")
            #
            # Store the result in -- '/poet_output/{type}/' folder
            #
            with open(f"../poet_output/{self.type}/results.pickle", 'wb') as file:
                pickle.dump(data, file)
            #
            #
            #
            print(f"\nThe results are stored in --/poet_output/{self.type}/ folder!\n")
            print(f"\nLower bound of the estimate: {y_hat_lower}"
                  f"\nUpper bound of the estimate: {y_hat_upper}")

            return y_hat_lower, y_hat_upper

        except Exception as e:
            print(f"\nUnknown Exception: {e}\n")


if __name__ == '__main__':

    X_train, y_train = None, None
    X_test, y_test = None, None

    params = {
            "type": "eccentricity",
            "epochs": 20,
            "batch_size": 5,
            "verbose": 2,
            "threshold": 1000
            }

    poet = POET_IC_Solver(X_train=X_train, y_train=y_train, **params)
    poet.generate_model()
    poet.fit()
    poet.evaluate(X_test=X_test, y_test=y_test)




