# SpikeballReferee-ML

This repository contains all the software artifacts created within the scope of my bachelor's thesis, which focuses on using machine learning to differentiate between rim and net hits in Spikeball. The goal is to create a digital referee that can automatically detect faulty hits, resolving common disputes during gameplay.

## Project Overview

This project is divided into three main components:

1.  **Data Collection:** Acquiring accelerometer data from an Arduino sensor during Spikeball gameplay.
2.  **Model Training:** Training a machine learning model using the collected data.
3.  **Arduino Firmware:** Utilizing the trained model on an Arduino for real-time hit detection.

## Repository Structure

The repository is organized as follows:

```
SpikeballReferee-ML/
├── data_collector/ # Code for data collection (Python)
├── model_training/ # Code for model training (Python)
├── arduino_firmware/ # Code for data collection & inference on the Arduino
├── LICENSE
└── README.md
```

## Getting Started

Refer to the specific directories for detailed instructions on setting up and running each component. A typical workflow is as follows:

1.  **Data Collection:** Use the `data_collector` program to gather accelerometer data from real Spikeball games, labeling each hit as either "rim" or "net". There is an arduino program for sending accelerometer data to a host PC.
2.  **Model Training:** Use the `model_training` programm to process the collected data and train the machine learning model.
3.  **Arduino Firmware:** Upload the `arduino_firmware/Inference` code to your Arduino, using your freshly trained model.

If you are not planning to collect your own data, use the model provided and start by uploading the inference sketch.

## Dependencies

The specific dependencies for each component are detailed within the respective directories. Generally, you will need:

- Arduino IDE
- Python 3.x
- Libraries like: `pandas`, `numpy`, `scikit-learn`, `tensorflow-lite-micro`

## Future development

This project is provided as-is and will probably not be further developed in the future.
Here are some things that could be done to make this project more useful:

- Migrate Arduino sketches to PlatformIO projects
- Support for other (less expensive) MCU's and IMU sensors
- Making use of the new [LiteRT library](https://github.com/google-ai-edge/litert) which seems to replace TFLite-Micro

## Contributing

If you would like to contribute to this project, please feel free to submit a pull request.

## License

This project is licensed under the **GNU General Public License v3.0**. See the `LICENSE` file for details. In summary, this license allows you to use, share, and modify this software, even for commercial purposes, as long as you provide appropriate credit, disclose source code changes, and license derived works under GPL v3.
