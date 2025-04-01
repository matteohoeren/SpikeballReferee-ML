/*
  Tensorflow Lite Micro SpikeballReferee Inference
  Written for the Arduino 33 BLE (Sense) rev1 
  (rev2 won't work because of another IMU sensor)

  This sketch has 3 main functions:
  - Record acceleration from a LSM9DS1 Sensor
  - Run inference with the collected data and the provided model
  - Display the result with the onboard LED / send the results via BLE
  - Future: Add acoustic signal

  Written by Matteo Hoeren 10 may 2024
*/

#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <ArduinoBLE.h>
#include <avr/dtostrf.h>
#include <math.h>
#include "model.h"

const int ledPin = LED_BUILTIN; // set ledPin to on-board LED
const int buttonPin = 4; // set buttonPin to digital pin 4

const float accelerationThreshold = 1000.0;
const int numSamples = 210;
const int numChannels = 3;

const float train_min_vals[3] = { -8049.000000f, -8083.000000f, -8946.000000f };
const float train_max_vals[3] = { 8059.000000f, 9005.000000f, 9017.000000f };

const char* deviceServiceUuid = "19B10010-E8F2-537E-4F6C-D104768A1214";
const char* deviceCharacteristicUuid = "19B10011-E8F2-537E-4F6C-D104768A1214";
const char* deviceDescriptorUuid = "19B10012-E8F2-537E-4F6C-D104768A1214";


BLEService spikeballService(deviceServiceUuid); 
BLEByteCharacteristic spikeballCharacteristic(deviceCharacteristicUuid, BLERead | BLENotify);
BLEDescriptor spikeballServiceDescriptor(deviceDescriptorUuid, "rim-detection");


int samplesRead = numSamples;
float CALIBRATION_X, CALIBRATION_Y, CALIBRATION_Z;

// global variables used for TensorFlow Lite (Micro)
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

constexpr int tensorArenaSize = 32 * 1024;
uint8_t tensorArena[tensorArenaSize] __attribute__((aligned(16)));

void setup() {
  Serial.begin(115200);
  while (!Serial);
  Serial.println("[DEBUG] Serial initialized");

  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  digitalWrite(LEDB, LOW);
  pinMode(LED_BUILTIN, OUTPUT);
  

  // Initialize IMU
  if (!IMU.begin()) {
    Serial.println("[ERROR] Failed to initialize IMU!");
    while (1);
  }
  Serial.println("[DEBUG] IMU initialized");

  // Configure IMU settings
  IMU.setAccelFS(3);           
  IMU.setAccelODR(5);           
  IMU.setAccelOffset(0, 0, 0);  
  IMU.setAccelSlope(1, 1, 1);  
  IMU.accelUnit = GRAVITY;
  Serial.println("[DEBUG] IMU configured");

  // Perform calibration
  Serial.println("[INFO] Starting calibration countdown...");
  delay(1000);
  Serial.println("[INFO] 3");
  delay(1000);
  Serial.println("[INFO] 2");
  delay(1000);
  Serial.println("[INFO] 1");
  delay(1000);
  IMU.readAcceleration(CALIBRATION_X, CALIBRATION_Y, CALIBRATION_Z);
  Serial.println("[DEBUG] Calibration values:");
  Serial.print("X: "); Serial.print(CALIBRATION_X);
  Serial.print(" Y: "); Serial.print(CALIBRATION_Y);
  Serial.print(" Z: "); Serial.println(CALIBRATION_Z);

  // Initialize TensorFlow: Pass the name of the char array from the model.h file
  tflModel = tflite::GetModel(spikeball_tflite);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[ERROR] Model schema mismatch!");
    while (1);
  }
  Serial.println("[DEBUG] TF Model loaded");

  static tflite::MicroInterpreter static_interpreter(
    tflModel,
    tflOpsResolver,
    tensorArena,
    tensorArenaSize,
    nullptr
  );
  tflInterpreter = &static_interpreter;

  if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[ERROR] Failed to allocate tensors!");
    while (1);
  }
  Serial.println("[DEBUG] Tensors allocated");

  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  // Print tensor details
  // Serial.println("[DEBUG] Input tensor details:");
  // Serial.print("Dimensions: "); Serial.println(tflInputTensor->dims->size);
  // Serial.print("Size of first dim: "); Serial.println(tflInputTensor->dims->data[0]);
  // Serial.print("Type: "); Serial.println(tflInputTensor->type);

  if (!BLE.begin()) {
    Serial.println("starting Bluetooth® Low Energy module failed!");

    while (1);
  }

  // set the local name peripheral advertises
  BLE.setLocalName("SpikeballReferee");
  // set the UUID for the service this peripheral advertises:
  BLE.setAdvertisedService(spikeballService);
  spikeballCharacteristic.addDescriptor(spikeballServiceDescriptor);
  spikeballService.addCharacteristic(spikeballCharacteristic);

  // add the service
  BLE.addService(spikeballService);

  spikeballCharacteristic.writeValue(0);

  // start advertising
  BLE.advertise();

  Serial.println("Bluetooth® device active, waiting for connections...");
}

void lightUpGreen(){
  digitalWrite(LEDR, LOW);
  digitalWrite(LEDG, HIGH);
}

void lightUpRed(){
  digitalWrite(LEDG, LOW);
  digitalWrite(LEDR, HIGH);
}

void normalizeInputData(TfLiteTensor* inputTensor) {
  if (!inputTensor || inputTensor->type != kTfLiteFloat32) {
    Serial.println("[ERROR] Invalid input tensor for normalization!");
    return;
  }

  float* inputData = inputTensor->data.f;
  // Calculate total elements based on tensor dimensions for robustness
  // Assumes input shape is [1, numSamples, numChannels] or similar flattened representation
  int numElements = 1;
  for (int i = 0; i < inputTensor->dims->size; ++i) {
     numElements *= inputTensor->dims->data[i];
  }

  if (numElements != numSamples * numChannels) {
     Serial.print("[WARNING] Unexpected number of elements in input tensor: ");
     Serial.print(numElements);
     Serial.print(" Expected: ");
     Serial.println(numSamples * numChannels);
     // Might still work if flattening is consistent, but indicates potential config mismatch
  }

  Serial.println("[DEBUG] Normalizing input data...");

  for (int i = 0; i < numSamples; ++i) {
    for (int j = 0; j < numChannels; ++j) {
      int index = i * numChannels + j;

      // Ensure index is within bounds (safety check)
      if (index >= numElements) {
          Serial.print("[ERROR] Normalization index out of bounds: "); Serial.println(index);
          continue; // Skip this element
      }

      float minVal = train_min_vals[j]; // Get min for current channel (X, Y, or Z)
      float maxVal = train_max_vals[j]; // Get max for current channel
      float range = maxVal - minVal;
      float currentValue = inputData[index];

      // Avoid division by zero if min == max for a channel
      if (fabs(range) < 1e-6f) { // Use float epsilon comparison
          inputData[index] = 0.0f; // Assign a neutral value
      } else {
          // Apply the formula: 2 * (X - min) / (max - min) - 1
          inputData[index] = 2.0f * (currentValue - minVal) / range - 1.0f;
      }
    }
  }
  Serial.println("[DEBUG] Normalization complete.");
}

void loop() {
  float aX, aY, aZ;
  float deltaX, deltaY, deltaZ;

  while (samplesRead == numSamples) {
    BLE.poll();
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      
      deltaX = (aX - CALIBRATION_X) * 1000;
      deltaY = (aY - CALIBRATION_Y) * 1000;
      deltaZ = (aZ - CALIBRATION_Z) * 1000;

      float acceleration = sqrt(deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ);

      if (acceleration > accelerationThreshold) {
        //Serial.println("\n[DEBUG] Motion detected! Starting data collection");
        //Serial.print("Acceleration trigger value: "); Serial.println(acceleration);
        samplesRead = 0;
        break;
      }
    }
  }

  // Record sample
  while (samplesRead < numSamples) {
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      
      deltaX = (aX - CALIBRATION_X) * 1000;
      deltaY = (aY - CALIBRATION_Y) * 1000;
      deltaZ = (aZ - CALIBRATION_Z) * 1000;

      int idx = samplesRead * numChannels;
      tflInputTensor->data.f[idx + 0] = deltaX;
      tflInputTensor->data.f[idx + 1] = deltaY;
      tflInputTensor->data.f[idx + 2] = deltaZ;


      samplesRead++;

      if (samplesRead == numSamples) {
        normalizeInputData(tflInputTensor); 


        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          Serial.println("[ERROR] Invoke failed!");
          return;
        }

        float rimProb = tflOutputTensor->data.f[0];
        float netProb = tflOutputTensor->data.f[1];
        
        Serial.println("\n[DEBUG] Raw output tensor values:");
        Serial.print("Output[0] (Rim): "); Serial.println(rimProb, 6);
        Serial.print("Output[1] (Net): "); Serial.println(netProb, 6);
        
        Serial.println("\n[RESULT]");
        Serial.print("Net: ");
        Serial.println(netProb*100, 6);
        Serial.print("Rim: ");
        Serial.println(rimProb*100, 6);
        BLE.poll();

        // Convert float probability (0-1) to a uint8_t (0-255)
        uint8_t byteValue = (uint8_t)(rimProb * 255);
        spikeballCharacteristic.writeValue(byteValue);



        if(netProb > rimProb){
          lightUpGreen();
        } else {
          lightUpRed();
        }
        
        // Print first and last few samples of collected data
        // Serial.println("\n[DEBUG] First 3 samples of collected data:");
        // for(int i = 0; i < 3; i++) {
        //   int idx = i * numChannels;
        //   Serial.print(tflInputTensor->data.f[idx + 0]); Serial.print(",");
        //   Serial.print(tflInputTensor->data.f[idx + 1]); Serial.print(",");
        //   Serial.println(tflInputTensor->data.f[idx + 2]);
        // }
        
        // Serial.println("[DEBUG] Last 3 samples of collected data:");
        // for(int i = numSamples-3; i < numSamples; i++) {
        //   int idx = i * numChannels;
        //   Serial.print(tflInputTensor->data.f[idx + 0]); Serial.print(",");
        //   Serial.print(tflInputTensor->data.f[idx + 1]); Serial.print(",");
        //   Serial.println(tflInputTensor->data.f[idx + 2]);
        // }
      }
    } 
  }
}