
from utils import load_data, split_data
from model import train_model
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from sklearn.model_selection import train_test_split  # Add this import
import numpy as np
import tensorflow as tf

def main():
    X,Y = load_data("../collected_data")
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = split_data(X, Y)
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_valid shape:", X_valid.shape)
    print("Y_valid shape:", Y_valid.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)
    print("X_train dtype:", X_train.dtype)
    print("X_test dtype:", X_test.dtype)
    print("X_valid dtype:", X_valid.dtype)
    train_model(X_train, Y_train, X_test, Y_test, X_valid, Y_valid)
    

if __name__ == "__main__":
    main()
