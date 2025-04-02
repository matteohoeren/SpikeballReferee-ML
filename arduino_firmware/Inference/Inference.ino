/*
  Tensorflow Lite Micro SpikeballReferee Inference
  Written for the Arduino 33 BLE (Sense) rev1
  (rev2 won't work because of another IMU sensor)

  This sketch has 3 main functions:
  - Record acceleration from a LSM9DS1 Sensor
  - Run inference with the collected data and the provided model
  - Display the result with the onboard LED / send the results via BLE

  Modified by Matteo Hoeren 
*/

#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <ArduinoBLE.h>
#include <math.h> // Keep for fabs() and sqrt()
#include "model.h" // Contains the TFLite model byte array

// --- Configuration ---
// Set DEBUG_MODE to 1 to enable Serial output, 0 to disable
#define DEBUG_MODE 0

// --- Constants ---
const int ledPin = LED_BUILTIN; // Built-in LED (often green or yellow)
const int buttonPin = 4; // Digital pin 4 for potential future use
const int buzzerPin = D6;

const float accelerationThreshold = 1000.0; // Trigger threshold in milli-g
const int numSamples = 210;                 // Number of samples expected by the model
const int numChannels = 3;                  // X, Y, Z acceleration channels

// Normalization constants derived from Python training script
const float train_min_vals[3] = { -8049.000000f, -8083.000000f, -8946.000000f };
const float train_max_vals[3] = { 8059.000000f, 9005.000000f, 9017.000000f };

// --- BLE UUIDs ---
const char* deviceServiceUuid = "19B10010-E8F2-537E-4F6C-D104768A1214";
const char* deviceCharacteristicUuid = "19B10011-E8F2-537E-4F6C-D104768A1214";
const char* deviceDescriptorUuid = "19B10012-E8F2-537E-4F6C-D104768A1214";

// --- BLE Objects ---
BLEService spikeballService(deviceServiceUuid);
// Characteristic to send the rim probability (0-255)
BLEByteCharacteristic spikeballCharacteristic(deviceCharacteristicUuid, BLERead | BLENotify);
BLEDescriptor spikeballServiceDescriptor(deviceDescriptorUuid, "Rim Probability (0-255)");

// --- Global Variables ---
int samplesRead = numSamples; // Start in idle state, waiting for trigger
float CALIBRATION_X, CALIBRATION_Y, CALIBRATION_Z; // Stores resting 'g' values

// --- TensorFlow Lite Variables ---
tflite::AllOpsResolver tflOpsResolver; // Pulls in all standard TF Micro Ops
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Memory arena for TensorFlow Micro
constexpr int tensorArenaSize = 32 * 1024; // 32KB, adjust based on model needs
// Align arena to 16 bytes for performance
alignas(16) uint8_t tensorArena[tensorArenaSize];

// --- Helper Macro for Conditional Serial Printing ---
#if DEBUG_MODE
  #define DEBUG_PRINT(x) Serial.print(x)
  #define DEBUG_PRINTLN(x) Serial.println(x)
  // Define DEBUG_PRINT_FLOAT for specific float formatting
  #define DEBUG_PRINT_FLOAT(val, prec) Serial.print(val, prec)
#else
  #define DEBUG_PRINT(x)    // Compiles to nothing
  #define DEBUG_PRINTLN(x)  // Compiles to nothing
  #define DEBUG_PRINT_FLOAT(val, prec) // Compiles to nothing
#endif

// --- Function Declarations (for BLE callbacks) ---
void onBLEConnected(BLEDevice central);
void onBLEDisconnected(BLEDevice central);
void signalErrorLoop();


// ==============================================================
// ===                  Setup Function                        ===
// ==============================================================
void setup() {
#if DEBUG_MODE
  Serial.begin(115200);
  // Wait for Serial connection only in DEBUG mode
  while (!Serial);
  DEBUG_PRINTLN("\n[DEBUG] Serial initialized");
#endif

  // Initialize onboard RGB LED pins
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  digitalWrite(LEDR, LOW);
  digitalWrite(LEDG, LOW);
  digitalWrite(LEDB, LOW);
  // Initialize standard built-in LED
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  pinMode(buzzerPin, OUTPUT);

  // Initialize IMU
  if (!IMU.begin()) {
    DEBUG_PRINTLN("[ERROR] Failed to initialize IMU!");
    signalErrorLoop(); // Indicate error permanently
  }
  DEBUG_PRINTLN("[DEBUG] IMU initialized");

  // Configure IMU settings
  IMU.setAccelFS(3);           // Set accelerometer full scale range (+/- 3g suitable for impacts)
  IMU.setAccelODR(5);          // Set accelerometer output data rate (104 Hz = 5)
  IMU.setAccelOffset(0, 0, 0); // Using manual calibration below
  IMU.setAccelSlope(1, 1, 1);
  IMU.accelUnit = GRAVITY;     // Read calibration in 'g'
  DEBUG_PRINTLN("[DEBUG] IMU configured");

  // Perform calibration
  DEBUG_PRINTLN("[INFO] Starting calibration countdown (keep board still)...");
  digitalWrite(LEDB, HIGH); // Blue LED during countdown
#if DEBUG_MODE
  delay(1000); DEBUG_PRINTLN("[INFO] 3...");
  delay(1000); DEBUG_PRINTLN("[INFO] 2...");
  delay(1000); DEBUG_PRINTLN("[INFO] 1...");
  delay(1000);
#else
  delay(4000); // Wait 4 seconds if no debug output
#endif
  digitalWrite(LEDB, LOW);
  // Read resting acceleration to establish baseline 'g' vector
  IMU.readAcceleration(CALIBRATION_X, CALIBRATION_Y, CALIBRATION_Z);
  DEBUG_PRINTLN("[DEBUG] Calibration values (in g):");
  // Manual formatting for calibration values
  DEBUG_PRINT("  X: "); DEBUG_PRINT_FLOAT(CALIBRATION_X, 4);
  DEBUG_PRINT(", Y: "); DEBUG_PRINT_FLOAT(CALIBRATION_Y, 4);
  DEBUG_PRINT(", Z: "); DEBUG_PRINT_FLOAT(CALIBRATION_Z, 4);
  DEBUG_PRINTLN("");


  // Initialize TensorFlow
  tflModel = tflite::GetModel(spikeball_tflite); // Get model from model.h
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    DEBUG_PRINTLN("[ERROR] Model schema mismatch!");
    signalErrorLoop();
  }
  DEBUG_PRINTLN("[DEBUG] TF Model loaded");

  // Create Interpreter
  static tflite::MicroInterpreter static_interpreter(
    tflModel, tflOpsResolver, tensorArena, tensorArenaSize, nullptr
  );
  tflInterpreter = &static_interpreter;

  // Allocate Tensors
  if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
    DEBUG_PRINTLN("[ERROR] Failed to allocate tensors!");
    signalErrorLoop();
  }
  DEBUG_PRINTLN("[DEBUG] Tensors allocated");

  // Get pointers to Input/Output Tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  // Verify Input Tensor properties (optional debug check)
#if DEBUG_MODE
  DEBUG_PRINTLN("[DEBUG] Input Tensor Details:");
  DEBUG_PRINT("  Dims: "); DEBUG_PRINTLN(tflInputTensor->dims->size);
  if (tflInputTensor->dims->size >= 3) { // Basic check for expected dimensions
     DEBUG_PRINT("  Shape: (");
     DEBUG_PRINT(tflInputTensor->dims->data[0]); DEBUG_PRINT(", "); // Batch
     DEBUG_PRINT(tflInputTensor->dims->data[1]); DEBUG_PRINT(", "); // Samples
     DEBUG_PRINT(tflInputTensor->dims->data[2]); DEBUG_PRINTLN(")"); // Channels
     if (tflInputTensor->dims->data[1] != numSamples || tflInputTensor->dims->data[2] != numChannels) {
         DEBUG_PRINTLN("[WARNING] Input tensor shape mismatch expected!");
     }
  }
  DEBUG_PRINT("  Type: "); DEBUG_PRINTLN(tflInputTensor->type == kTfLiteFloat32 ? "Float32" : "Other");
#endif


  // Initialize BLE
  if (!BLE.begin()) {
    DEBUG_PRINTLN("[ERROR] starting Bluetooth® Low Energy module failed!");
    signalErrorLoop();
  }

  // Set BLE event handlers for connection management
  BLE.setEventHandler(BLEConnected, onBLEConnected);
  BLE.setEventHandler(BLEDisconnected, onBLEDisconnected);

  // Set BLE appearance and name
  BLE.setLocalName("SpikeballReferee");
  BLE.setDeviceName("SpikeballReferee"); // Can set both
  // Set advertised service
  BLE.setAdvertisedService(spikeballService);

  // Add descriptor and characteristic to the service
  spikeballCharacteristic.addDescriptor(spikeballServiceDescriptor);
  spikeballService.addCharacteristic(spikeballCharacteristic);

  // Add the service
  BLE.addService(spikeballService);

  // Set initial value for the characteristic (e.g., 0 probability)
  spikeballCharacteristic.writeValue((uint8_t)0);

  // Start advertising
  if (BLE.advertise()) {
    DEBUG_PRINTLN("[INFO] Bluetooth® advertising started...");
  } else {
    DEBUG_PRINTLN("[ERROR] Failed to start BLE advertising!");
    signalErrorLoop();
  }

  DEBUG_PRINTLN("[INFO] Setup complete. Waiting for motion or connection...");
  tone(buzzerPin,4000,250);
}


// ==============================================================
// ===              LED Indicator Functions                   ===
// ==============================================================
void lightUpGreen(){
  digitalWrite(LEDR, LOW);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, LOW);
}

void lightUpRed(){
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, LOW);
  digitalWrite(LEDB, LOW);
}

void lightUpBlue(){
  digitalWrite(LEDR, LOW);
  digitalWrite(LEDG, LOW);
  digitalWrite(LEDB, HIGH);
}

// Turn off indicator LEDs
void lightsOff(){
  digitalWrite(LEDR, LOW);
  digitalWrite(LEDG, LOW);
  digitalWrite(LEDB, LOW);
}

// Indicate a permanent error state (e.g., blinking red)
void signalErrorLoop() {
  while (true) {
    digitalWrite(LEDR, HIGH); delay(250);
    digitalWrite(LEDR, LOW); delay(250);
  }
}



// ==============================================================
// ===             Normalization Function                     ===
// ==============================================================
/**
 * Normalizes the input data buffer in place to the range [-1, 1].
 */
void normalizeInputData(TfLiteTensor* inputTensor) {
  if (!inputTensor || inputTensor->type != kTfLiteFloat32) {
    DEBUG_PRINTLN("[ERROR] Invalid input tensor for normalization!");
    return;
  }

  float* inputData = inputTensor->data.f;
  // Calculate total elements based on tensor dimensions for robustness
  int numElements = 1;
  for (int i = 0; i < inputTensor->dims->size; ++i) {
     numElements *= inputTensor->dims->data[i];
  }

  // Basic sanity check
  if (numElements != numSamples * numChannels) {
     DEBUG_PRINT("[WARNING] Unexpected number of elements in input tensor: ");
     DEBUG_PRINT(numElements); DEBUG_PRINT(" Expected: "); DEBUG_PRINTLN(numSamples * numChannels);
  }

  DEBUG_PRINTLN("[DEBUG] Normalizing input data...");

  for (int i = 0; i < numSamples; ++i) {
    for (int j = 0; j < numChannels; ++j) {
      int index = i * numChannels + j;

      // Safety check for bounds
      if (index >= numElements) {
          DEBUG_PRINT("[ERROR] Normalization index out of bounds: "); DEBUG_PRINTLN(index);
          continue; // Skip this element
      }

      float minVal = train_min_vals[j];
      float maxVal = train_max_vals[j];
      float range = maxVal - minVal;
      float currentValue = inputData[index];

      // Avoid division by zero
      if (fabs(range) < 1e-6f) {
          inputData[index] = 0.0f; // Assign neutral value if range is zero
      } else {
          // Apply normalization formula: 2 * (X - min) / (max - min) - 1
          inputData[index] = 2.0f * (currentValue - minVal) / range - 1.0f;
      }
    }
  }
  DEBUG_PRINTLN("[DEBUG] Normalization complete.");
}


// ==============================================================
// ===               BLE Event Handlers                       ===
// ==============================================================

void onBLEConnected(BLEDevice central) {
  DEBUG_PRINT("[INFO] Device connected: ");
  DEBUG_PRINTLN(central.address());
  // Optional: Stop advertising when connected
  // BLE.stopAdvertise();
  digitalWrite(LED_BUILTIN, HIGH); // Turn on built-in LED to indicate connection
}

void onBLEDisconnected(BLEDevice central) {
  DEBUG_PRINT("[INFO] Device disconnected: ");
  DEBUG_PRINTLN(central.address());
  digitalWrite(LED_BUILTIN, LOW); // Turn off built-in LED
  lightsOff(); // Turn off result LEDs

  // IMPORTANT: Re-start advertising so a new device can connect
  if (BLE.advertise()) {
      DEBUG_PRINTLN("[INFO] Restarted advertising.");
  } else {
      DEBUG_PRINTLN("[ERROR] Failed to restart advertising!");
      // Consider a less fatal error signal here if needed
  }
}


// ==============================================================
// ===                  Main Loop                             ===
// ==============================================================
void loop() {
  // Poll BLE frequently to handle connections, disconnections, and data requests
  BLE.poll();

  float aX, aY, aZ;
  float deltaX, deltaY, deltaZ;

  // --- Wait for Significant Motion (Idle State) ---
  while (samplesRead == numSamples) {
    // Poll BLE even while waiting
    BLE.poll();

    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(aX, aY, aZ); // Read in 'g' units

      // Calculate change from calibration, convert g to milli-g
      deltaX = (aX - CALIBRATION_X) * 1000.0f;
      deltaY = (aY - CALIBRATION_Y) * 1000.0f;
      deltaZ = (aZ - CALIBRATION_Z) * 1000.0f;

      // Calculate magnitude of the change vector
      float accelerationMagnitude = sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ);

      // Check if threshold is exceeded
      if (accelerationMagnitude > accelerationThreshold) {
        DEBUG_PRINTLN(""); // Add a newline for better log readability
        DEBUG_PRINT("[DEBUG] Motion detected! Accel: "); DEBUG_PRINT_FLOAT(accelerationMagnitude, 2);
        DEBUG_PRINT(" > "); DEBUG_PRINT_FLOAT(accelerationThreshold, 2); DEBUG_PRINTLN("");
        DEBUG_PRINTLN("[DEBUG] Starting data collection...");
        lightUpBlue(); // Blue LED indicates data collection phase
        samplesRead = 0; // Reset sample counter to start collection
        break; // Exit waiting loop
      }
    }
     delay(5); // Small delay to prevent busy-waiting and allow BLE poll
  }

  // --- Record Sample Data ---
  while (samplesRead < numSamples) {
    // Don't forget to poll BLE during data collection too
    BLE.poll();

    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(aX, aY, aZ); // Read in 'g'

      // Calculate delta from calibration, scale to milli-g
      deltaX = (aX - CALIBRATION_X) * 1000.0f;
      deltaY = (aY - CALIBRATION_Y) * 1000.0f;
      deltaZ = (aZ - CALIBRATION_Z) * 1000.0f;

      // Store the delta values directly into the input tensor's float buffer
      int idx = samplesRead * numChannels;
      if ((idx + 2) < (numSamples * numChannels)) { // Bounds check before writing
          tflInputTensor->data.f[idx + 0] = deltaX;
          tflInputTensor->data.f[idx + 1] = deltaY;
          tflInputTensor->data.f[idx + 2] = deltaZ;
      } else {
          DEBUG_PRINT("[ERROR] Tensor write index out of bounds: "); DEBUG_PRINTLN(idx);
          // Handle error? Maybe reset samplesRead = numSamples to abort?
      }

      samplesRead++;

      // --- If Buffer Full -> Normalize & Infer ---
      if (samplesRead == numSamples) {
        lightsOff(); // Turn off blue LED (collection done)
        DEBUG_PRINTLN("[DEBUG] Data collection complete.");

        // 1. Normalize the collected data IN PLACE
        normalizeInputData(tflInputTensor);

        // 2. Run Inference
        DEBUG_PRINTLN("[DEBUG] Running inference...");
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          DEBUG_PRINTLN("[ERROR] Invoke failed!");
          lightUpRed(); // Indicate error with Red LED
          // Consider resetting samplesRead = numSamples to allow retrying
          return; // Exit this loop iteration on failure
        }
        DEBUG_PRINTLN("[DEBUG] Inference complete.");

        // 3. Process Output Tensor
        float rimProb = tflOutputTensor->data.f[0]; // Probability for class 0 (Rand/Rim)
        float netProb = tflOutputTensor->data.f[1]; // Probability for class 1 (Netz/Net)

        // Clamp probabilities just in case of numerical instability
        rimProb = max(0.0f, min(1.0f, rimProb));
        netProb = max(0.0f, min(1.0f, netProb));

        DEBUG_PRINTLN("\n[RESULT]");
        DEBUG_PRINT("  Net Probability: "); DEBUG_PRINT_FLOAT(netProb * 100.0f, 4); DEBUG_PRINTLN("%");
        DEBUG_PRINT("  Rim Probability: "); DEBUG_PRINT_FLOAT(rimProb * 100.0f, 4); DEBUG_PRINTLN("%");


        // 4. Update BLE Characteristic if connected
        if (BLE.connected()) {
          // Convert Rim probability [0.0, 1.0] to a byte [0, 255]
          uint8_t bleValue = (uint8_t)(rimProb * 255.0f);
          if (spikeballCharacteristic.writeValue(bleValue)) {
              DEBUG_PRINT("[DEBUG] BLE Characteristic updated with value: "); DEBUG_PRINT(bleValue);
              DEBUG_PRINT(" (from "); DEBUG_PRINT_FLOAT(rimProb, 4); DEBUG_PRINTLN(")");
          } else {
              DEBUG_PRINTLN("[WARNING] Failed to write BLE characteristic.");
          }
        } else {
          DEBUG_PRINTLN("[INFO] No BLE device connected, skipping characteristic update.");
        }

        // 5. Indicate Result with LED
        if (netProb > rimProb) {
          DEBUG_PRINTLN("  Prediction: NET");
          lightUpGreen(); // Green for Net
          tone(buzzerPin, 2000, 100);
          delay(150);
          tone(buzzerPin, 2000, 100);
        } else {
          DEBUG_PRINTLN("  Prediction: RIM");
          lightUpRed(); // Red for Rim
          tone(buzzerPin, 250, 500);
        }

        // samplesRead is already numSamples, so the outer loop condition
        // will be met, and it will go back to waiting for motion.
         DEBUG_PRINTLN("\n[INFO] Ready for next detection.");

      } // End of if (samplesRead == numSamples)
    } // End of if (IMU.accelerationAvailable())
     delay(1); // Small delay to yield processor slightly if needed
  } // End of while (samplesRead < numSamples) - Data collection loop
} // End of loop()