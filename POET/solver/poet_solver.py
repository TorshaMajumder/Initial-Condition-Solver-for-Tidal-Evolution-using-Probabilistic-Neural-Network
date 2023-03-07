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


class POET_IC_Solver(object):
    """
    path_to_store: string
        path or directory to store the output
    retrain: boolean
        retraining of the NN model
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

    def __init__(self, type=None, path_to_store=None, epochs=500, batch_size=100,
                 verbose=2, threshold=1000, version=None,retrain=False):
        self.epochs = epochs
        self.verbose = verbose
        self.type = type
        self.path = path_to_store
        self.version = version
        self.batch_size = batch_size
        self.threshold = threshold
        self.model = None
        self.y_hat = None
        self.y_sd = None
        self.retrain = retrain
        #
        # Check if '/self.path/' file or directory exist
        #
        try:
            if not os.path.exists(f'/{self.path}/'):
                raise FileNotFoundError(f"\nFileNotFoundError: '{self.path}/' file or directory "
                                        f"doesn't exist.\n")
        except BaseException as e:
            print(f"{e}")
            exit()
        #
        # Create '/self.path/poet_output/{type}' folder if it does not exist already
        #
        try:
            if not os.path.exists(f'/{self.path}/poet_output/{self.type}_{self.version}'):
                os.makedirs(f'/{self.path}/poet_output/{self.type}_{self.version}')
        except FileNotFoundError as e:
            print(f"\nFileNotFoundError: {self.path}/poet_output/{self.type}_{self.version} file or directory "
                  f"doesn't exist.\n")
            exit()

    def store_data(self, X_train=None, y_train=None):
        """
        Parameters
        ----------
        X_train: numpy ndarray
            training data set
        y_train: numpy ndarray
            training labels

        Returns
        -------
        the results are stored as CSV files in folder - /{self.path}/poet_output/{self.type}_{self.version}/datasets
        """
        #
        # Declare all variables
        #
        file_list = list()
        X_train, y_train = np.array(X_train), np.array(y_train)
        #
        # Create '{self.path}/poet_output/{self.type}_{self.version}/datasets/' folder if it does not exist already
        # Else store all the files from the /datasets/ folder
        #
        if not os.path.exists(f'/{self.path}/poet_output/{self.type}_{self.version}/datasets/'):
            os.makedirs(f'/{self.path}/poet_output/{self.type}_{self.version}/datasets/')
        else:
            for file in os.listdir(f'/{self.path}/poet_output/{self.type}_{self.version}/datasets/'):
                file_list.append(file)
        #
        # Check if the data is a numpy nd-array
        #
        try:

            if isinstance(X_train, (np.ndarray, np.generic, list)):
                X_train = np.array(X_train)
            elif isinstance(X_train, pd.DataFrame):
                X_train = X_train.to_numpy()
            else:
                raise TypeError(f"\nTypeError: Unable to load the data for X_train. Expected a format as "
                                f"a numpy array or a pandas dataframe.\n")

            if isinstance(y_train, (np.ndarray, np.generic, list)):
                y_train = np.array(y_train)
            elif isinstance(y_train, pd.DataFrame):
                y_train = y_train.to_numpy()
            else:
                raise TypeError(f"\nTypeError: Unable to load the data for y_train. Expected a format as "
                                f"a numpy array or a pandas dataframe.\n")
        except BaseException as e:
            print(f"{e}")
            exit()
        #
        # Change the dimension of the data
        #
        try:
            if X_train.ndim < 2:
                X_train = X_train.reshape((1, X_train.shape[0]))
            if y_train.ndim < 2:
                if y_train.ndim == 0:
                    y_train = np.array([y_train])
                y_train = y_train.reshape((1, y_train.shape[0]))
        except Exception as e:
            print(f"\nException Raised: {e}\n")
        #
        # Store the data in a CSV file
        #
        new_data_df = pd.DataFrame(X_train)
        new_data_df = new_data_df.astype('float64')
        new_labels_df = pd.DataFrame(y_train)
        new_labels_df = new_labels_df.astype('float64')
        #
        # Append data to the dataframe or create a new dataframe
        #
        try:
            if len(file_list) > 0:
                if "data.csv.gz" in file_list:
                    data_df = pd.read_csv(f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/data.csv.gz",
                                          compression='gzip')
                    new_data_df.columns = data_df.columns
                    data_df = pd.concat([data_df, new_data_df], ignore_index=True)
                    data_df.to_csv(path_or_buf=f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/data.csv.gz",
                                   index=False, compression='gzip')

                if "label.csv.gz" in file_list:
                    labels_df = pd.read_csv(f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/label.csv.gz",
                                            compression='gzip')
                    new_labels_df.columns = labels_df.columns
                    labels_df = pd.concat([labels_df, new_labels_df], ignore_index=True)
                    labels_df.to_csv(path_or_buf=f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/label.csv.gz",
                                     index=False, compression='gzip')
            else:
                new_data_df.to_csv(path_or_buf=f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/data.csv.gz",
                                   index=False, compression='gzip')
                new_labels_df.to_csv(path_or_buf=f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/label.csv.gz",
                                     index=False, compression='gzip')

        except Exception as e:
            print(f"\nException Raised: {e}\n")
            exit()

        print(f"\nThe data is stored in --{self.path}/poet_output/{self.type}_{self.version}/datasets/ folder!\n")

    def load_data(self):
        #
        # Declare all variables
        #
        file_list = list()
        #
        # Check if '{self.path}/poet_output/{self.type}_{self.version}/datasets/' folder exist already
        # Else store all the files from the /datasets/ folder
        #
        try:
            if os.path.exists(f'/{self.path}/poet_output/{self.type}_{self.version}/datasets/'):
                for file in os.listdir(f'/{self.path}/poet_output/{self.type}_{self.version}/datasets/'):
                    file_list.append(file)
                try:
                    if "data.csv.gz" not in file_list:
                        raise FileNotFoundError(f"\nFileNotFoundError: 'data.csv.gz' file "
                                                f"doesn't exists in the folder - "
                                                f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/.\n")
                    if "label.csv.gz" not in file_list:
                        raise FileNotFoundError(f"\nFileNotFoundError: 'label.csv.gz' file "
                                                f"doesn't exists in the folder - "
                                                f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/.\n")
                except BaseException as e:
                    print(f"{e}")
                    exit()
        except Exception as e:
            print(f"\nException Raised: {e}\n")
            exit()
        #
        # Load the data
        #
        try:
            data_df = pd.read_csv(f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/data.csv.gz",
                                  compression='gzip')
            labels_df = pd.read_csv(f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/label.csv.gz",
                                    compression='gzip')
            #
            # Convert pandas dataframe to numpy ndarray
            #
            data_df = data_df.to_numpy()
            labels_df = labels_df.to_numpy()

            return data_df, labels_df

        except Exception as e:
            print(f"\nException Raised: {e}\n")
            exit()

    def log_loss(self, y_true, y_pred):
        return -y_pred.log_prob(y_true)

    def fit_evaluate(self, X_test=None, y_test=None):
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

        Notes: the results are stored as dictionary in folder - /{self.path}/poet_output/{self.type}_{self.version}/
                - as results.pickle
        """
        #
        # Declare all variables
        #
        file_list = list()
        count = 0
        #
        # Load the training data set
        #
        X_train, y_train = self.load_data()
        #
        # Check if the training data set reached the threshold value
        #
        try:
            if len(y_train) < self.threshold:
                raise ValueError(f"\nValueError: the training data size (current size - {len(y_train)}) should be greater "
                                 f"than equals to the threshold value--{self.threshold} to begin training!\n")
        except BaseException as e:
            print(f"{e}")
            print(f"\nLower bound of the estimate: None"
                  f"\nUpper bound of the estimate: None\n")
            exit()
        #
        # Verify if the NN model already exist or retrain is True
        #
        try:
            if os.path.exists(f'/{self.path}/poet_output/{self.type}_{self.version}'):
                for file in os.listdir(f'/{self.path}/poet_output/{self.type}_{self.version}'):
                    file_list.append(file)
            #
            # Create new NN model if "model.h5" doesn't exist or retrain = True
            #
            if "model.h5" not in file_list or self.retrain:

                try:
                    #
                    # Fit the model using the X_train and y_train
                    # Custom loss function used - log loss
                    # Optimizer - Adam
                    #
                    #
                    # event shape: integer vector Tensor representing the shape
                    # of single draw from this distribution
                    event_shape = 1
                    #
                    # features: number of features from the training sample
                    #
                    features = X_train.shape[1]
                    #
                    # Initialize the Adam optimizer
                    #
                    opt = tf.keras.optimizers.Adam(learning_rate=0.0003)
                    #
                    # build the model using a independent normal distribution
                    #
                    self.model = tf.keras.Sequential([
                        tf.keras.layers.Dense(units=tfpl.IndependentNormal.params_size(event_shape),
                                              input_shape=(features,)),
                        tfpl.IndependentNormal(event_shape=event_shape)])
                    self.model.compile(loss=self.log_loss, optimizer=opt)
                    self.model.fit(X_train, y_train,
                                   epochs=self.epochs,
                                   batch_size=self.batch_size,
                                   verbose=self.verbose)
                    #
                    # Save the model (model.h5) under the folder - {self.path}/poet_output/{self.type}_{self.version}/
                    #
                    self.model.save(f"/{self.path}/poet_output/{self.type}_{self.version}/model.h5")
                    print(f"\nThe POET_IC_Solver model is fitted!\n"
                          f"\nThe model is stored in -- {self.path}/poet_output/{self.type}_{self.version}/model.h5 "
                          f"directory!\n")
                except Exception as e:
                    print(f"\nException Raised: {e}\n")
            #
            # If "model.h5" exists and retrain = False then load the existing NN model
            #
            else:
                try:
                    #
                    # Load the stored model from the folder - /{self.path}/poet_output/{self.type}_{self.version}/
                    #
                    self.model = tf.keras.models.load_model(f"/{self.path}/poet_output/{self.type}_{self.version}/model.h5",
                                                            custom_objects={'log_loss': self.log_loss})
                except FileNotFoundError as e:
                    print(f"\nFileNotFoundError: Unable to load the NN model!\n")

        except Exception as e:
            print(f"\nException Raised: {e}\n")

        #
        # Check if the data is a numpy nd-array
        #
        try:
            if isinstance(X_test, (np.ndarray, np.generic)):
                X_test = np.array(X_test)
            elif isinstance(X_test, pd.DataFrame):
                X_test = X_test.to_numpy()
            else:
                raise TypeError(f"\nTypeError: Unable to load the data for X_test. Expected a format as "
                                f"a numpy array or a pandas dataframe.\n")

            if y_test:
                if isinstance(y_test, (np.ndarray, np.generic)):
                    y_test = np.array(y_test)
                elif isinstance(y_test, pd.DataFrame):
                    y_test = y_test.to_numpy()
                else:
                    raise TypeError(f"\nTypeError: Unable to load the data for y_test. Expected a format as "
                                    f"a numpy array or a pandas dataframe.\n")
        except BaseException as e:
            print(f"{e}")
            exit()
        #
        # Change the dimension of the data
        #
        try:
            if X_test.ndim < 2:
                X_test = X_test.reshape((1, X_test.shape[0]))
            if y_test:
                if y_test.ndim < 2:
                    y_test = y_test.reshape((1, y_test.shape[0]))
        except Exception as e:
            print(f"\nException Raised: {e}\n")

        try:
            #
            # Calculate the mean and the std. deviation
            #
            self.y_hat = self.model(X_test).mean()
            self.y_sd = self.model(X_test).stddev()
            #
            # Calculate the lower and the upper bound of the original estimate
            #
            y_hat_lower = self.y_hat - 2 * self.y_sd
            y_hat_upper = self.y_hat + 2 * self.y_sd
            #
            # Calculate the accuracy of the model if y_test is provided
            #
            if y_test:
                for i in range(len(y_test)):
                    if (y_test[i] >= y_hat_lower[i]) and (y_test[i] <= y_hat_upper[i]):
                        count += 1
                #
                #
                #
                accuracy = (count/len(y_test))*100
            else:
                accuracy = None
            #
            # Calculate the log ratio - (y_hat_upper/y_hat_lower)
            #
            with np.errstate(invalid='ignore'):
                log_ratio = np.nanmean(np.log(y_hat_upper/y_hat_lower))
            #
            # Store the results as a dictionary
            #
            data = {"y_hat_lower": y_hat_lower, "y_hat_upper": y_hat_upper,
                    "accuracy": accuracy, "log_ratio": log_ratio,
                    "y_hat": self.y_hat, "y_sd": self.y_sd}
            #
            #
            #
            print(f"\nThe POET_IC_Solver model is evaluated!\n")
            #
            # Store the result in -- '/{self.path}/poet_output/{self.type}/' folder
            #
            with open(f"/{self.path}/poet_output/{self.type}_{self.version}/results.pickle", 'wb') as file:
                pickle.dump(data, file)
            #
            #
            #
            print(f"The results are stored in --{self.path}/poet_output/{self.type}_{self.version}/results.pickle "
                  f"directory!\n")
            print(f"\nLower bound of the estimate: {y_hat_lower}"
                  f"\nUpper bound of the estimate: {y_hat_upper}")

            return y_hat_lower, y_hat_upper

        except Exception as e:
            print(f"\nException Raised: {e}\n")


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
    poet = POET_IC_Solver(**params)
    poet.store_data(X_train=X_train, y_train=y_train)
    poet.fit_evaluate(X_test=X_test)






