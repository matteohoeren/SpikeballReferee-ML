import tensorflow as tf
import numpy as np

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


VERBOSE_OUTPUT = False
SAVE_AS_TFLITE = False


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


def create_cnn_model():
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

# def create_cnn_model():
#     model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(210, 3)),
#     tf.keras.layers.Conv2D(
#         8, (4, 3), padding="same", activation="relu"),
#     tf.keras.layers.MaxPooling2D((3, 1)),  # (batch, 70, 3, 8)
#     tf.keras.layers.Dropout(0.1),
#     tf.keras.layers.Conv2D(16, (4, 1), padding="same", activation="relu"),  # (batch, 70, 3, 16)
#     tf.keras.layers.MaxPooling2D((3, 1)),  # (batch, 23, 1, 16)
#     tf.keras.layers.Dropout(0.1),
#     tf.keras.layers.Flatten(),  # (batch, 23*16)
#     tf.keras.layers.Dense(16, activation="relu"),  # (batch, 16)
#     tf.keras.layers.Dropout(0.1),
#     tf.keras.layers.Dense(2, activation="softmax")  # (batch, 2)
# ])

#     return model

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




def evaluate_model(model, X_test, Y_test):
    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, Y_test)
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
    model = create_cnn_model()
    model.summary()
    plot_cnn_model(model)
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


    X_train_norm, X_test_norm, X_valid_norm = normalize_data(X_train, X_test, X_valid)
    print(f"Normalized train range: [{X_train_norm.min():.4f}, {X_train_norm.max():.4f}]")
    #model.fit(X_train, Y_train, epochs=100, batch_size=20, validation_data=(X_test, Y_test))   
    model.fit(X_train_norm, Y_train, epochs=100, batch_size=20, validation_data=(X_test_norm, Y_test),callbacks=[reduce_lr, early_stopping])   
    np.set_printoptions(precision=2, suppress=True)

    correct_pred = 0
    incorrect_pred = 0
    for sample_num in range(len(X_test)):
        single_sample = np.expand_dims(X_test[sample_num], axis=0)  # Add batch dimension
        single_sample = np.expand_dims(single_sample, axis=-1)  # Add channel dimension
        #print(single_sample)
        result = model.predict(single_sample, verbose=0)
        predicted_label = np.argmax(result[0])
        true_label = Y_test[sample_num]
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
    print(f"Correctly predicted samples: {correct_pred/len(X_test)*100}% of sample size: {len(X_test)}")       

    
    evaluate_model(model, X_test_norm, Y_test)

    if SAVE_AS_TFLITE:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open('spikeball.tflite', 'wb') as f:
            f.write(tflite_model)

