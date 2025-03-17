import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Function to read data from CSV files and create X, Y arrays
def load_data(root_folder):
    X = []
    Y = []
    ignored_files = 0
    problematic_files = []

    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            net_folder = os.path.join(folder_path, "net")
            rim_folder = os.path.join(folder_path, "rim")
            if os.path.exists(net_folder) and os.path.exists(rim_folder):
                for category_folder, label in [(net_folder, 1), (rim_folder, 0)]:
                    for filename in os.listdir(category_folder):
                        file_path = os.path.join(category_folder, filename)

                        try:
                            df = pd.read_csv(file_path, header=None)

                            # Check for unexpected strings in the DataFrame
                            if not df.applymap(lambda x: isinstance(x, (int, float))).all().all():
                                problematic_files.append(file_path)
                                print(f"Problem detected in file: {file_path}")
                                print(df)
                                continue

                            # Discard CSV files with less than 210 lines
                            if df.shape[0] < 210:
                                ignored_files += 1
                                continue

                            # Truncate CSV files with more than 210 lines
                            if df.shape[0] > 210:
                                df = df.head(210)

                            X.append(df.iloc[:, 1:].values)
                            Y.append(label)

                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}")
                            problematic_files.append(file_path)

    print("Ignored number of files due to less than 210 datapoints:", ignored_files)
    print("Total problematic files:", len(problematic_files))

    return np.array(X), np.array(Y)


def split_data(X, Y):
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.1, random_state=42)
    X_test, X_valid, Y_test, Y_valid = train_test_split(X_temp, Y_temp, test_size=0.05, random_state=42)
    return X_train, Y_train, X_test, Y_test, X_valid, Y_valid

def remove_non_numeric_rows(data):
    """
    Removes rows with any non-numeric data from a numpy array.

    Parameters:
        data (numpy.ndarray): The array to process.

    Returns:
        numpy.ndarray: A new array with only numeric rows.
    """
    return np.array([row for row in data if np.all([isinstance(x, (int, float, np.number)) for x in row])])