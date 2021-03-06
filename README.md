# Intro
This is a small project for trying out simple finger gesture recognition using two AXDL345 accelerometer on Raspberry Pi.

### Setup:
To do finger gesture recognition, we need at least two of ADXL345. Since  ADXL345 uses I2C to communicate, we need two different addresses for the purpose. Connect one of the ADXL345 's SDO to VCC to change the address to 0x1d (default 0x53). Then connect them to Raspberry Pi 3B as below. Then attach the two ADXL345 as below to a hand glove.

<center><img alt="node" src="images/fritzing ADXL345.jpg" width="500"/></center>

<center><img alt="node" src="images/ADXL result open.png" width="300"/></center>

<center><img alt="node" src="images/ADXL result close.png" width="300"/></center>

### How it works:

ADXL345 accelerometer gives x,y,z acceleration depending on the chip's orientation. See https://www.youtube.com/watch?v=T_iXLNkkjFo for more details.

The program captures the accelerometer data corresponding to the simple finger gesture of open and close in different orientation. Then Bayesian optimization tunes a xgboost classifier hyperparameters (tree depth and learning rate) for better result.

