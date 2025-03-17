# SendAcceleration (Arduino Sketch)

This Arduino sketch reads acceleration values from the LSM9DS1 IMU sensor and continuously prints them to the Serial Monitor. This data is intended to be collected by the `data_collector.py` script to later train an ML-Model.

## Hardware Requirements

- Arduino Nano 33 BLE (Sense) or
- Any MCU compatible with the IDE connected to an LSM9DS1 sensor

## Dependencies

- Arduino IDE
- Arduino_LSM9DS1 Library

## Functionality

This sketch performs the following main tasks:

1.  **Initializes the IMU:** Sets up the LSM9DS1 sensor for acceleration readings.
2.  **Calibrates the Accelerometer:** Takes a short calibration reading to establish a baseline.
3.  **Reads Acceleration Data:** Continuously reads acceleration values (x, y, z) from the LSM9DS1 sensor.
4.  **Prints Data to Serial:** Prints the acceleration data, along with a timestamp, to the Serial Monitor in CSV format.

## Setup

1.  **Install Dependencies:** Ensure that you have the `Arduino_LSM9DS1` library installed via the Arduino IDE's Library Manager.
2.  **Connect Hardware:**
    - If using the Arduino Nano 33 BLE (Sense), no additional hardware is needed (the IMU is built-in).
    - If using any other MCU, connect the LSM9DS1 sensor to the MCU according and configure it correctly in code.
3.  **Upload the Sketch:** Upload the `SendAcceleration.ino` sketch to your Arduino board using the Arduino IDE.
4.  **Configure Serial Monitor:** Open the Serial Monitor in the Arduino IDE and set the baud rate to **115200**.

## Data Format

The sketch outputs data to the Serial Monitor in the following CSV format:

```
timestamp,x_accel,y_accel,z_accel
```

Where:

- `timestamp` is the time in milliseconds since the Arduino started.
- `x_accel`, `y_accel`, and `z_accel` are the acceleration values in the x, y, and z axes, respectively (scaled by 1000 and converted to integers for data collection efficiency).

## Calibration

The sketch includes a short calibration phase at the beginning to establish a baseline reading for each axis. Keep the sensor still during this calibration period for best results.

## Usage

1.  Upload the sketch to your Arduino.
2.  Open the Serial Monitor in the Arduino IDE (make sure the baud rate is set to 115200).
3.  The sketch will print the accelerometer data to the Serial Monitor.
4.  Use the `data_collector.py` Python script (from the `data_collector` directory) to capture this data and save it to CSV files for model training.

## Notes

- **Data Collection:** This sketch is designed to be used with a host PC running the data collection script (`data_collector/__main__.py`).
- **Calibration:** The calibration at startup is basic. For more accurate results, consider running a more thorough calibration routine (as mentioned in the code comments) and updating the `IMU.setAccelOffset` and `IMU.setAccelSlope` values in the `setup()` function.

## DIY Calibration

For improved accuracy, consider running a DIY Calibration sketch first and replace the lines below to the code output of the program:

```
IMU.setAccelOffset(0, 0, 0); // uncalibrated
IMU.setAccelSlope (1, 1, 1); // uncalibrated
```

## Further Exploration

- Examine the `data_collector.py` script to understand how to capture the serial data and save it to files.
- Explore the Arduino_LSM9DS1 library documentation for more advanced IMU configuration options.
