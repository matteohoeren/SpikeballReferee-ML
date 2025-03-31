import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
#import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Input, Reshape

import plotly.graph_objects as go
from plotly.subplots import make_subplots


VERBOSE_OUTPUT = False
SAVE_AS_TFLITE = False
USE_END_EARLY_REDUCE_LR = True


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2, 
    patience=10, 
    min_lr=1e-6,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True, 
    verbose=1
)

# Conv 1D Variant Model
def create_1dconv_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(210, 3)),  # Input shape matches your data
        tf.keras.layers.Conv1D(8, 4, padding="same", activation="relu"),  # Apply 1D convolution
        tf.keras.layers.MaxPooling1D(3),  # Pool along the time axis
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv1D(16, 4, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling1D(3),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(2, activation="softmax")  # Final output layer
    ])
    return model

def create_complex_cnn_model(l1=0.001, l2=0.001, dropout_rate=0.2):
    """Creates a more complex CNN model with regularization, Batch Normalization, and increased depth."""
    model = tf.keras.Sequential([
        Input(shape=(210, 3)),
        Reshape((210, 3, 1)),

        Conv2D(32, (3, 3), padding="same", activation="relu", kernel_regularizer=l1_l2(l1=l1, l2=l2)),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding="same"),
        Dropout(dropout_rate),

        Conv2D(64, (3, 3), padding="same", activation="relu", kernel_regularizer=l1_l2(l1=l1, l2=l2)),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding="same"),
        Dropout(dropout_rate),

        Flatten(),
        Dense(128, activation="relu", kernel_regularizer=l1_l2(l1=l1, l2=l2)),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(2, activation="softmax")  # Final output layer
    ])
    return model

# Conv 2D Variant Model
def create_2dconv_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(210, 3)),
        tf.keras.layers.Reshape((210, 3, 1)),  # Add a channel dimension
        tf.keras.layers.Conv2D(
            8, (4, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((3, 1)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(16, (4, 1), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((3, 1)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    return model

def normalize_data(X_train, X_test, X_valid=None):
    # Calculate min and max values from training data only
    min_vals = X_train.min(axis=(0, 1))
    max_vals = X_train.max(axis=(0, 1))
    
    # Scale to [-1, 1] range
    X_train_normalized = 2 * (X_train - min_vals) / (max_vals - min_vals) - 1
    X_test_normalized = 2 * (X_test - min_vals) / (max_vals - min_vals) - 1
    
    if X_valid is not None:
        X_valid_normalized = 2 * (X_valid - min_vals) / (max_vals - min_vals) - 1
        return X_train_normalized, X_test_normalized, X_valid_normalized
    
    return X_train_normalized, X_test_normalized

def plot_confusion_matrix(model, X_eval, y_eval, class_names, title='Confusion Matrix', cmap=plt.cm.Blues, normalize=None):
    """
    Generates and plots a Confusion Matrix for a given Keras model.

    Args:
        model (tf.keras.Model): The TRAINED and BUILT Keras model.
                                It's assumed model.build() has been implicitly
                                or explicitly called before this function.
        X_eval (np.ndarray or tf.Tensor): Evaluation features. Expected shape is
                                         (num_samples, 210, 3) for batch prediction.
        y_eval (np.ndarray or tf.Tensor): True labels for evaluation. Expected shape
                                          is (num_samples,) with integer labels (0, 1, ...).
        class_names (list): List of class names corresponding to labels [0, 1, ...].
                            Example: ['Rand', 'Netz'] for labels 0 and 1.
        title (str): The title for the plot.
        cmap (matplotlib.colors.Colormap): Colormap for the heatmap.
        normalize ({'true', 'pred', 'all'}, optional): Normalization strategy.
            - 'true': Normalizes over true labels (rows) -> Recall.
            - 'pred': Normalizes over predicted labels (cols) -> Precision.
            - 'all': Normalizes over all predictions.
            - None (default): Shows absolute counts.
    """
    print("--- Preparing for Confusion Matrix ---")
    print(f"Input X_eval type: {type(X_eval)}")
    print(f"Input y_eval type: {type(y_eval)}")

    # 1. Validate and potentially convert inputs
    if not isinstance(X_eval, (np.ndarray, tf.Tensor)):
        try:
            print("Attempting to convert X_eval to NumPy array...")
            X_eval_processed = np.array(X_eval)
        except Exception as e:
            print(f"ERROR: Could not convert X_eval to NumPy array: {e}")
            return
    else:
        X_eval_processed = X_eval

    if not isinstance(y_eval, (np.ndarray, tf.Tensor)):
         try:
            print("Attempting to convert y_eval to NumPy array...")
            y_eval_processed = np.array(y_eval)
         except Exception as e:
            print(f"ERROR: Could not convert y_eval to NumPy array: {e}")
            return
    else:
        y_eval_processed = y_eval

    # Ensure y_eval is 1D array/tensor
    if len(y_eval_processed.shape) != 1:
         print(f"Warning: y_eval shape is {y_eval_processed.shape}. Attempting to flatten.")
         # Handle potential TensorFlow tensor flatten/reshape if needed
         if hasattr(y_eval_processed, 'numpy'): # Check if it's a tf.Tensor
             y_eval_processed = tf.reshape(y_eval_processed, [-1]).numpy()
         else:
             y_eval_processed = y_eval_processed.flatten()
         print(f"Shape after flatten: {y_eval_processed.shape}")
         if len(y_eval_processed.shape) != 1:
              print("ERROR: Could not flatten y_eval to 1D array.")
              return


    print(f"Processed X_eval shape: {X_eval_processed.shape}")
    print(f"Processed y_eval shape: {y_eval_processed.shape}")

    # Check if X_eval shape looks reasonable for batch prediction
    if len(X_eval_processed.shape) != 3 or X_eval_processed.shape[1] != 210 or X_eval_processed.shape[2] != 3:
        print(f"WARNING: Unexpected shape for X_eval: {X_eval_processed.shape}. "
              "Expected (num_samples, 210, 3) for batch prediction.")
        # Depending on the error source, you might decide to stop here or proceed cautiously.

    # 2. Make Predictions (Batch Prediction is Standard)
    try:
        print("Running model.predict on the evaluation batch...")
        # Ensure the model receives the data in the expected format (usually NumPy or TF Tensor)
        y_pred_proba = model.predict(X_eval_processed)
        print(f"Prediction output shape: {y_pred_proba.shape}")

    except Exception as e:
        print(f"\nERROR during model.predict: {e}")
        print("This often means the model wasn't properly built before calling predict.")
        print("Try calling 'model.build(input_shape=(None, 210, 3))' after creating/loading the model.")
        print("Also check model definition for inconsistencies (e.g., Input vs Reshape vs Conv2D layers).")
        # Optionally print model summary if possible
        try:
             model.summary()
        except:
             print("Could not print model summary.")
        return # Stop execution if prediction fails

    # 3. Convert probabilities to class labels
    #    Assumes output layer is softmax/sigmoid giving probabilities per class
    y_pred = np.argmax(y_pred_proba, axis=1) # Get index of max probability
    print(f"Predicted labels shape: {y_pred.shape}")

    # 4. Calculate Confusion Matrix using sklearn
    try:
        print("Calculating confusion matrix...")
        cm = confusion_matrix(y_eval_processed, y_pred, normalize=normalize)
    except Exception as e:
        print(f"ERROR calculating confusion matrix: {e}")
        print("Check if y_eval and y_pred contain valid integer labels and have the same length.")
        return

    # 5. Plot the Confusion Matrix
    print("Plotting confusion matrix...")
    try:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(8, 6)) # Adjust figure size if needed
        disp.plot(cmap=cmap, ax=ax, values_format='.2f' if normalize else 'd')
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
        print("--- Confusion Matrix Plotting Complete ---")
    except Exception as e:
        print(f"ERROR plotting confusion matrix: {e}")

def plot_history(history):
    """Plots training and validation accuracy and loss on a single plot using Plotly."""

    acc = history.history.get('accuracy')
    val_acc = history.history.get('val_accuracy')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')

    if acc is None or val_acc is None or loss is None or val_loss is None:
        print("Warning: Accuracy or loss data not found in history. Check your model training setup.")
        return

    epochs = list(range(1, len(acc) + 1))

    fig = go.Figure()

    # Add accuracy traces
    fig.add_trace(go.Scatter(x=epochs, y=acc, mode='lines', name='Training Accuracy', marker=dict(color='blue', line=dict(width=4))))
    fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines', name='Test Accuracy', marker=dict(color='darkgreen', line=dict(width=4))))

    # Add loss traces
    fig.add_trace(go.Scatter(x=epochs, y=loss, mode='lines', name='Training Loss', marker=dict(color='red', line=dict(width=2))))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', name='Test Loss', marker=dict(color='purple', line=dict(width=2))))

    # Update layout
    fig.update_layout(
        xaxis_title="Epochs",
        yaxis_title="Value (Accuracy/Loss)",
        margin=dict(l=20, r=20, t=20, b=20),
        title_x=0.5,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2, 
            xanchor="center",
            x=0.5
        )
    )

    fig.show()

def evaluate_model(model, X_test, Y_test):
    # Evaluate the model on the validation set
    loss, accuracy = model.evaluate( X_test, Y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

def plot_cnn_model(model):
   plot_model(
    model,
    to_file="model_architecture.png",
    show_shapes=True,
    show_layer_names=True,
    dpi=96,
    rankdir="LR", 
    expand_nested=False,
    layer_range=None,
    show_layer_activations=True

    
)

def get_class_weights(Y_train):
    classes = np.unique(Y_train)
    if len(classes) != 2:
            print(f"Warnung: Erwartet wurden 2 Klassen, gefunden wurden {len(classes)}: {classes}")
            class_weight_dict = None
    else:
        weights = compute_class_weight(class_weight='balanced',
                                        classes=classes,
                                        y=Y_train)

        class_weight_dict = {classes[0]: weights[1], classes[1]: weights[0]}
        print(f"Verwendete Klassengewichte: {class_weight_dict}")
    return class_weight_dict

def train_model(X_train, Y_train, X_test, Y_test, X_valid, Y_valid):
    """
    Trains the 2D CNN model, evaluates it, and saves it.
    Optionally plots confusion matrices and detailes from the training process.

    Args:
        X_train, Y_train: Training data and labels.
        X_test, Y_test: Test data and labels (for final evaluation).
        X_valid, Y_valid: Validation data and labels (for monitoring during training).
    """
    print("--- Creating Model ---")
    model = create_2dconv_cnn_model()

    print()
    print("--- Building Model ---")
    # Explicitly build the model with the expected input shape
    try:
        model.build(input_shape=(None, 210, 3))
        print("Model built successfully.")
        model.summary() 
    except Exception as build_e:
        print(f"Warning: Error during explicit model.build: {build_e}. Model might be built implicitly later.")
        model.summary() 

    # plot_cnn_model(model) # Optional visualization
    print()
    print("--- Calculating Class Weights ---")
    class_weight_dict = get_class_weights(Y_train)

    print()
    print("--- Compiling Model ---")
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print()
    print("--- Normalizing Data ---")
    X_train_norm, X_test_norm, X_valid_norm = normalize_data(X_train, X_test, X_valid)
    print(f"Shape X_train_norm: {X_train_norm.shape}")
    print(f"Shape X_valid_norm: {X_valid_norm.shape}")
    print(f"Shape X_test_norm: {X_test_norm.shape}")
    print(f"Normalized train range: [{X_train_norm.min():.4f}, {X_train_norm.max():.4f}]")

    print()
    print("--- Starting Model Training ---")


    history = model.fit(X_train_norm, Y_train,
                        epochs=200, 
                        batch_size=20,
                        validation_data=(X_valid_norm, Y_valid),
                        class_weight=class_weight_dict,
                        callbacks=[reduce_lr, early_stopping],
                        verbose=1) 
    
    print("--- Training Finished ---")

    print()
    print("--- Evaluating Model on TEST Set ---")
    try:
        print("Running batch prediction on test set...")
        y_pred_proba_test = model.predict(X_test_norm)
        y_pred_test = np.argmax(y_pred_proba_test, axis=1)
    except Exception as e:
         print(f"\nERROR during batch prediction on test set: {e}")
         print("Check if model is built and X_test_norm has correct shape.")
         return 

    class_labels = ['Rand', 'Netz']
    plot_confusion_matrix(model, X_test_norm, Y_test, class_labels, title='Confusion Matrix (Test Set)', normalize=None)
    plot_confusion_matrix(model, X_test_norm, Y_test, class_labels, title='CM Normalisiert (Recall) (Test Set)', normalize='true')


    print("Running model.evaluate on test set...")
    evaluate_model(model, X_test_norm, Y_test) 


    correct_pred = np.sum(y_pred_test == Y_test)
    total_samples = len(Y_test)
    print(f"\nOverall Test Accuracy: {correct_pred / total_samples * 100:.2f}% ({correct_pred}/{total_samples})")

    if VERBOSE_OUTPUT:
        print("\n--- Detailed Test Set Prediction Output ---")
        np.set_printoptions(precision=2, suppress=True)
        for i in range(total_samples):
            true_label = Y_test[i]
            predicted_label = y_pred_test[i]
            confidence = np.max(y_pred_proba_test[i]) * 100
            status = "correctly" if predicted_label == true_label else "incorrectly"
            pred_class_name = class_labels[predicted_label]
            print(f"[{true_label}] Sample {i} is {status} predicted as {pred_class_name} with {confidence:.1f}% confidence")
        print("--- End Detailed Output ---")


    if SAVE_AS_TFLITE:
        print()
        print("--- Converting Model to TFLite ---")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            file_path = 'spikeball.tflite'
            with open(file_path, 'wb') as f:
                f.write(tflite_model)
            print(f"TFLite model saved successfully to {file_path}")
        except Exception as e:
            print(f"ERROR converting model to TFLite: {e}")

    print("--- train_model function finished ---")