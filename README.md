# Computer Vision Parkinson's Tremor Amplitude Measurement
A project investigating the measurement of the amplitude of Parkinson's hand tremors using computer vision (namely hand tracking) methods. Written for my dissertation at the University of Manchester.

---

Requires Mediapipe (```pip install mediapipe```), and OpenCV (```pip install opencv-python```).

The configuration file, ```hta_config.yaml```, contains various settings which should be set before running. These are each explained in the config file. The simplest way to run the program is to set ```AUTO_MODE``` and ```USE_CUSTOM_LANDMARKS``` to ```False``` in the configuration file, then run ```python3 hand_tremor_amplitude.py``` with no command line arguments. You will then be prompted to choose a video file to measure tremor amplitude from, input the depth measurement and input any any other required information. Tremor will then be measured and a plot, alongside the tremor measurement and an error analysis, will be presented.

The program can also be used without any user prompting, saving results and plots to a CSV file. Switch ```AUTO_MODE``` to ```True``` to do this, and call the program using ```python3 hand_tremor_amplitude.py [video filepath] [depth in cm] ['resting' or 'postural'] [tracking landmarks]```. This may be useful for automated batch processing of videos.

---

All of the code within this repository is a part of my dissertation "Development of a Computer Vision Method to Measure Parkinson's Tremor Amplitude", supervised by Dr. David Wong of the University of Manchester.
