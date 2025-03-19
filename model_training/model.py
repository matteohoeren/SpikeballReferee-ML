import tensorflow as tf
import numpy as np

from tensorflow.keras.utils import plot_model



VERBOSE_OUTPUT = False
SAVE_AS_TFLITE = False


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

def train_model(X_train, Y_train, X_test, Y_test):
    model = create_cnn_model()
    model.summary()
    plot_cnn_model(model)
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    # somewhere between 16 and 32 batch size was best
    # print(type(X_train))
    # print(type(Y_train))
    # print(type(X_test))
    # print(type(Y_test))

    # X_train = np.array(X_train, dtype=np.float32)
    # Y_train = np.array(Y_train, dtype=np.float32)
    # X_test = np.array(X_test, dtype=np.float32)
    # Y_test = np.array(Y_test, dtype=np.float32)

    model.fit(X_train, Y_train, epochs=100, batch_size=20, validation_data=(X_test, Y_test))   
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

    
    evaluate_model(model, X_test, Y_test)

    if SAVE_AS_TFLITE:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open('spikeball.tflite', 'wb') as f:
            f.write(tflite_model)

