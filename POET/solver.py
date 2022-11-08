#
# Import all the dependencies
#
import os
import random
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
#
# Set all the parameters
#
random.seed(49)
tfpl = tfp.layers
tfd = tfp.distributions
#
# Create '/results' folder if it does not exists already
#
if not os.path.exists('../results'):
    os.makedirs('../results')
#
#
#
class POET_IC_Solver(object):

    def __init__(self, X_train=None, y_train=None,
                 epochs=300, batch_size=100, verbose=0):
        self.X_train = X_train
        self.y_train = y_train
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.model = None
        self.y_hat = None
        self.y_sd = None

    def log_loss(self, y_true, y_pred):

        return -y_pred.log_prob(y_true)

    def generate_model(self):

        event_shape = 1
        features = self.X_train.shape[1]

        self. model = tf.keras.Sequential([
                      tf.keras.layers.Dense(units=tfpl.IndependentNormal.params_size(event_shape),
                                            input_shape=(features,)),
                      tfpl.IndependentNormal(event_shape=event_shape)])
        return

    def fit(self):

        self.model.compile(loss=self.log_loss, optimizer='adam')
        self.model.fit(self.X_train, self.y_train,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=self.verbose)
        #
        #
        #
        with open(f"../results/model.pickle", 'wb') as file:
            pickle.dump(self.model, file)

        return

    def evaluate(self, X_test, y_test):

        #
        # Declare all the variables
        #
        count = 0
        #
        #
        #
        self.y_hat = self.model(X_test).mean()
        self.y_sd = self.model(X_test).stddev()
        y_hat_lower = self.y_hat - 2 * self.y_sd
        y_hat_upper = self.y_hat + 2 * self.y_sd

        for i in range(len(y_test)):
            if (y_test[i] >= y_hat_lower[i]) and (y_test[i] <= y_hat_upper[i]):
                count += 1

        accuracy = (count/len(y_test))*100

        with np.errstate(invalid='ignore'):
            log_dist = np.nanmean(np.log(y_hat_upper/y_hat_lower))
        #
        # Store the metadata
        #
        data = {"y_hat_lower": y_hat_lower, "y_hat_upper": y_hat_upper,
                "accuracy": accuracy, "log_dist": log_dist,
                "y_hat": self.y_hat, "y_sd": self.y_sd}

        #
        #
        #
        print(f"\nThe POET_IC_Solver model is evaluated!\n")
        #
        # Store the metadata in -- '/results/' folder
        #
        with open(f"../results/metadata.pickle", 'wb') as file:
            pickle.dump(data, file)
        #
        #
        #
        print(f"\nThe metadata of the model is stored in --/results folder!\n")

        return y_hat_lower, y_hat_upper


if __name__ == '__main__':

    params = {
            "X_train": X_train,
            "y_train": y_train,
            "epochs": 500,
            "batch_size": 100,
            "verbose": 0
            }

    poet = POET_IC_Solver(**params)
    poet.generate_model()
    poet.fit()
    poet.evaluate(X_test=X_test, y_test=y_test)




