README of DES3 FYP Project, developed by Edmund Lee and Khang Minsoo of HKUST CSE 2019~2020

This project consists of two major segments
1. System set up – object detection, voice recognition and path planning
2. Guiding vehicle set-up

Part 1
The system has been tested on a personal laptop with GPU config of NVIDIA GTX 1050Ti. 
To run the system, the laptop has to be connected to a remote microphone along with ZED 1
Stereo Vision camera and HC05 bluetooth module.

After connecting the necessary periperhals, you can run the system by executing final_run.py


Part 2
For Guiding vehicle set-up the STM board has to be flashed with the STM code found in 
STM32_code_for_the_guiding_car.rar. When the hardware of the vehicle is ready with the flahsed
code, ensure that it can establish Bluetooth communication with the personal laptop. Afterwards,
the guiding car and the system of part 1 can be run together as one combined guiding system.

Team members for contact with regards to system information:
mkhang@connect.ust.hk, lyleeaf@connect.ust.hk
