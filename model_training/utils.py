import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Function to read data from CSV files and create X, Y arrays
import os
import pandas as pd
import numpy as np

def load_data(root_folder):
    X = []
    Y = []
    ignored_files = 0
    problematic_files = []

    # Process both NET and RIM categories directly in root folder
    for category_folder, label in [("net", 1), ("rim", 0)]:
        full_path = os.path.join(root_folder, category_folder)
        
        if not os.path.exists(full_path):
            print(f"Warning: Category folder {category_folder} not found in {root_folder}")
            continue

        for filename in os.listdir(full_path):
            file_path = os.path.join(full_path, filename)
            
            try:
                df = pd.read_csv(file_path, header=None)

                # Check for non-numeric values
                if not df.applymap(lambda x: isinstance(x, (int, float))).all().all():
                    problematic_files.append(file_path)
                    print(f"Non-numeric values in file: {file_path}")
                    continue

                # Handle row count requirements
                if len(df) < 210:
                    ignored_files += 1
                    continue
                
                # Truncate if too long
                df = df.head(210)
                
                X.append(df.iloc[:, 1:].values)
                Y.append(label)

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                problematic_files.append(file_path)

    print(f"Ignored {ignored_files} files with insufficient rows")
    print(f"Found {len(problematic_files)} problematic files")
    
    return np.array(X), np.array(Y)


def split_data(X, Y):
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_temp, Y_temp, test_size=0.4, random_state=42)
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

def remove_non_numeric_rows(data):
    """
    Removes rows with any non-numeric data from a numpy array.

    Parameters:
        data (numpy.ndarray): The array to process.

    Returns:
        numpy.ndarray: A new array with only numeric rows.
    """
    return np.array([row for row in data if np.all([isinstance(x, (int, float, np.number)) for x in row])])