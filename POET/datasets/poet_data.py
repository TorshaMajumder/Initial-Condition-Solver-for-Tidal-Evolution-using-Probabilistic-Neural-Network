#
# Import all the dependencies
#
import os
import numpy as np
import pandas as pd


def store_data(X=None, y=None, type=None):

    file_list = list()
    try:
        if type not in ["eccentricity", "orbital_period"]:
            raise ValueError(f"\n'{type}' is not a valid type!"
                             f"\nPlease provide the type as - 'orbital_period' or 'eccentricity'")
    except Exception as e:
        print(e)
        exit()
    #
    # Create '/datasets/{type}' folder if it does not exists already
    #
    if not os.path.exists(f'./datasets/{type}'):
        os.makedirs(f'./datasets/{type}')
    else:
        for file in os.listdir(f'./datasets/{type}'):
            file_list.append(file)
    #
    # Check if the data is a numpy nd-array
    #
    try:
        if isinstance(X, (np.ndarray, np.generic)):
            X = np.array(X)
        elif isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        else:
            raise TypeError(f"\nTypeError: Unable to load the data. Expected a format as "
                            f"a numpy array or a pandas dataframe.\n")

        if isinstance(y, (np.ndarray, np.generic)):
            y = np.array(y)
        elif isinstance(y, pd.DataFrame):
            y = y.to_numpy()
        else:
            raise TypeError(f"\nTypeError: Unable to load the data. Expected a format as "
                            f"a numpy array or a pandas dataframe.\n")
    except Exception as e:
        print(e)
        exit()
    try:
        if X.ndim < 2:
            X = X.reshape((1, X.shape[0]))
        if y.ndim < 2:
            y = y.reshape((1, y.shape[0]))
    except Exception as e:
        print(f"Exception Raised: {e}")
    #
    # Store the data in a CSV file
    #
    new_data_df = pd.DataFrame(X)
    new_data_df = new_data_df.astype('float64')
    new_labels_df = pd.DataFrame(y)
    new_labels_df = new_labels_df.astype('float64')
    try:
        if len(file_list) > 0:
            if "data.csv.gz" in file_list:
                data_df = pd.read_csv(f"./datasets/{type}/data.csv.gz", compression='gzip')
                new_data_df.columns = data_df.columns
                data_df = pd.concat([data_df, new_data_df], ignore_index=True)
                data_df.to_csv(path_or_buf=f"./datasets/{type}/data.csv.gz", index=False, compression='gzip')

            if "label.csv.gz" in file_list:
                labels_df = pd.read_csv(f"./datasets/{type}/label.csv.gz", compression='gzip')
                new_labels_df.columns = labels_df.columns
                labels_df = pd.concat([labels_df, new_labels_df], ignore_index=True)
                labels_df.to_csv(path_or_buf=f"./datasets/{type}/label.csv.gz", index=False, compression='gzip')
        else:
            new_data_df.to_csv(path_or_buf=f"./datasets/{type}/data.csv.gz", index=False, compression='gzip')
            new_labels_df.to_csv(path_or_buf=f"./datasets/{type}/label.csv.gz", index=False, compression='gzip')

    except Exception as e:
        print(e)
        exit()

    print(f"\nThe data is stored in --/datasets/{type}/ folder!\n")


def load_data(type=None, threshold=None):

    file_list = list()
    try:
        if os.path.exists(f'./datasets/{type}'):
            for file in os.listdir(f'./datasets/{type}'):
                file_list.append(file)
            try:
                if "data.csv.gz" not in file_list:
                    raise FileNotFoundError(f"\nFileNotFoundError: 'data.csv.gz' file "
                                                f"doesn't exists in the folder - /{type}.\n")
                if "label.csv.gz" not in file_list:
                    raise FileNotFoundError(f"\nFileNotFoundError: 'label.csv.gz' file "
                                            f"doesn't exists in the folder - /{type}.\n")
            except Exception as e:
                print(e)
                exit()
    except Exception as e:
        print(e)
        exit()

    try:
        data_df = pd.read_csv(f"./datasets/{type}/data.csv.gz", compression='gzip')
        labels_df = pd.read_csv(f"./datasets/{type}/label.csv.gz", compression='gzip')
        #
        # convert pandas dataframe to numpy ndarray
        #
        data_df = data_df.to_numpy()
        labels_df = labels_df.to_numpy()
        #
        # Check if the data samples reached the threshold value
        #
        if len(labels_df) < threshold:
            raise ValueError(f"\nValueError: the training data size should be greater "
                             f"than equals to the threshold value -- {threshold} to begin training!\n")
        else:
            return data_df, labels_df

    except Exception as e:
        print(f"\n{e}\n")
        exit()


if __name__ == '__main__':
    X, y= None, None
    store_data(X, y, type='eccentricity')
    X_train, y_train = load_data(type='eccentricity', threshold=1000)
    print(X_train.shape, y_train.shape)