import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Input, Reshape
from tensorflow.keras.regularizers import l1_l2

import plotly.graph_objects as go
from plotly.subplots import make_subplots


VERBOSE_OUTPUT = False
SAVE_AS_TFLITE = False
USE_END_EARLY_REDUCE_LR = True


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
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
            y=-0.2,  # Adjust this value to fine-tune the position
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

def train_model(X_train, Y_train, X_test, Y_test, X_valid, Y_valid):
    print(X_train.shape)
    model = create_2dconv_cnn_model()
    model.summary()
    plot_cnn_model(model)
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


    X_train_norm, X_test_norm, X_valid_norm = normalize_data(X_train, X_test, X_valid)
    print(f"Normalized train range: [{X_train_norm.min():.4f}, {X_train_norm.max():.4f}]")
    history = model.fit(X_train_norm, Y_train, epochs=200, batch_size=20, validation_data=(X_test_norm, Y_test),callbacks=[reduce_lr, early_stopping])   #,callbacks=[reduce_lr, early_stopping]
    np.set_printoptions(precision=2, suppress=True)

    correct_pred = 0
    incorrect_pred = 0
    for sample_num in range(len(X_valid_norm)):
        single_sample = np.expand_dims(X_valid_norm[sample_num], axis=0)  # Add batch dimension
        single_sample = np.expand_dims(single_sample, axis=-1)  # Add channel dimension
        #print(single_sample)
        result = model.predict(single_sample, verbose=0)
        predicted_label = np.argmax(result[0])
        true_label = Y_valid[sample_num]
        confidence = np.max(result[0]) * 100
        
        if predicted_label == true_label:
            correct_pred += 1
            if predicted_label == 0:
               if(VERBOSE_OUTPUT):
                print(f"[{true_label}]Sample {sample_num} is correctly predicted as RIM with {confidence}% confidence")
            else:
               if(VERBOSE_OUTPUT):
                print(f"[{true_label}]Sample {sample_num} is correctly predicted as NET with {confidence}% confidence")
        else:
            incorrect_pred += 1
            if predicted_label == 0:
               if(VERBOSE_OUTPUT):
                  print(f"[{true_label}]Sample {sample_num} is incorrectly predicted as RIM with {confidence}% confidence")
            else:
               if(VERBOSE_OUTPUT):
                print(f"[{true_label}]Sample {sample_num} is incorrectly predicted as NET with {confidence}% confidence")
    print(f"Correctly predicted samples: {correct_pred/len(X_valid_norm)*100}% of sample size: {len(X_valid_norm)}")       

    plot_history(history)
    
    evaluate_model(model, X_valid_norm, Y_valid)

    if SAVE_AS_TFLITE:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open('spikeball.tflite', 'wb') as f:
            f.write(tflite_model)

