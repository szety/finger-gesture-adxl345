# finger-gesture-adxl345
This is a small project for trying out simple finger gesture recognition using two AXDL345 accelerometer on Raspberry Pi.

### Setup:
To do finger gesture recongniton, we need at least two of ADXL345.Since  ADXL345 uses I2C to communicate, we need two different addresses for the purpose. Connect one of the ADXL345 's SDO to VCC to change address to 0x1d (default 0x53). Then connect them to Raspberry Pi 3B as below. Then attach the two ADXL345 as below to a hand gloves.

<center><img src="images/fritzing ADXL345.jpg" width="500"/></center>

### How it works:
ADXL345 accelerometer gives x,y,z acceraleration depending on the chip's orientation. See https://www.youtube.com/watch?v=T_iXLNkkjFo for more details.

Use the program to capture data of corresponding finger gesture of open or close. Then use xgb 
 
