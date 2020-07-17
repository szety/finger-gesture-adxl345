"""
A Qt application with two AXDL345 accelerometers for
finger gesture recognition
"""

# Author: Tom Sze <sze.takyu@gmail.com>

import sys
import numpy as np
import pyqtgraph as pg
import csv

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, \
    QVBoxLayout, QLineEdit, QPushButton, QSizePolicy, \
    QSpacerItem, QApplication
from joblib import load

is_test = True

if not is_test:
    import board
    import busio
    import adafruit_adxl34x

    i2c_address1 = 0x1d
    i2c_address2 = 0x53
    i2c = busio.I2C(board.SCL, board.SDA)
    accelerometer = adafruit_adxl34x.ADXL345(i2c, address=i2c_address1)
    accelerometer2 = adafruit_adxl34x.ADXL345(i2c, address=i2c_address2)


class MyWidget(QWidget):

    def __init__(self):
        super(MyWidget, self).__init__()

        self.is_start_record = False

        num_data = 100  # graph limit
        self.x = list(range(num_data))
        self.v_1_x = [0 for _ in range(num_data)]
        self.v_1_y = [0 for _ in range(num_data)]
        self.v_1_z = [0 for _ in range(num_data)]

        self.v_2_x = [0 for _ in range(num_data)]
        self.v_2_y = [0 for _ in range(num_data)]
        self.v_2_z = [0 for _ in range(num_data)]

        self.record = []

        self.model = load('ADXL345_xgb_gesture.joblib')

        """ controls """
        # layout
        self.hbox_main = QHBoxLayout()
        self.vbox_btns = QVBoxLayout()
        self.vbox_graph = QVBoxLayout()

        # label
        self.lb_1_x_value = QLabel('0')
        self.lb_1_y_value = QLabel('0')
        self.lb_1_z_value = QLabel('0')

        self.lb_2_x_value = QLabel('0')
        self.lb_2_y_value = QLabel('0')
        self.lb_2_z_value = QLabel('0')

        self.lb_gesture = QLabel('Gesture')

        newfont = QFont('Times', 15)
        self.lb_1_x_value.setFont(newfont)
        self.lb_1_y_value.setFont(newfont)
        self.lb_1_z_value.setFont(newfont)
        self.lb_2_x_value.setFont(newfont)
        self.lb_2_y_value.setFont(newfont)
        self.lb_2_z_value.setFont(newfont)
        self.lb_gesture.setFont(newfont)

        # graph
        self.graph_1_x = pg.PlotWidget()
        self.graph_1_y = pg.PlotWidget()
        self.graph_1_z = pg.PlotWidget()

        self.graph_2_x = pg.PlotWidget()
        self.graph_2_y = pg.PlotWidget()
        self.graph_2_z = pg.PlotWidget()

        self.graph_1_x.setLabel('left', '1 x acc')
        self.graph_1_y.setLabel('left', '1 y acc')
        self.graph_1_z.setLabel('left', '1 z acc')
        self.graph_1_x.setLabel('bottom', 'time')
        self.graph_1_y.setLabel('bottom', 'time')
        self.graph_1_z.setLabel('bottom', 'time')

        self.graph_2_x.setLabel('left', '2 x acc')
        self.graph_2_y.setLabel('left', '2 y acc')
        self.graph_2_z.setLabel('left', '2 z acc')
        self.graph_2_x.setLabel('bottom', 'time')
        self.graph_2_y.setLabel('bottom', 'time')
        self.graph_2_z.setLabel('bottom', 'time')

        self.graph_1_x.setYRange(-15, 15)
        self.graph_1_y.setYRange(-15, 15)
        self.graph_1_z.setYRange(-15, 15)

        self.graph_2_x.setYRange(-15, 15)
        self.graph_2_y.setYRange(-15, 15)
        self.graph_2_z.setYRange(-15, 15)

        pen = pg.mkPen(color=(255, 0, 0))

        self.data_line_1_x = self.graph_1_x.plot(self.x, self.v_1_x, pen=pen)
        self.data_line_1_y = self.graph_1_y.plot(self.x, self.v_1_x, pen=pen)
        self.data_line_1_z = self.graph_1_z.plot(self.x, self.v_1_x, pen=pen)

        self.data_line_2_x = self.graph_2_x.plot(self.x, self.v_1_x, pen=pen)
        self.data_line_2_y = self.graph_2_y.plot(self.x, self.v_1_x, pen=pen)
        self.data_line_2_z = self.graph_2_z.plot(self.x, self.v_1_x, pen=pen)

        # line edit
        self.txb_gesture = QLineEdit()

        # button
        self.btn_start_record = QPushButton('Start record', self)
        self.btn_stop_record_and_save_csv = QPushButton('Stop Record and Save CSV', self)

        self.btn_start_record.setFont(QFont('Times', 15))
        self.btn_stop_record_and_save_csv.setFont(QFont('Times', 15))

        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.btn_start_record.setSizePolicy(size_policy)
        self.btn_stop_record_and_save_csv.setSizePolicy(size_policy)

        # timer
        self.timer = QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.timer_function)
        if not is_test:
            self.timer.start()

        """ place controls to layout """
        vertical_spacer = QSpacerItem(0, 5)

        self.vbox_graph.addWidget(self.graph_1_x, alignment=Qt.AlignTop)
        self.vbox_graph.addWidget(self.lb_1_x_value, alignment=Qt.AlignCenter)
        self.vbox_graph.addItem(vertical_spacer)

        self.vbox_graph.addWidget(self.graph_1_y, alignment=Qt.AlignTop)
        self.vbox_graph.addWidget(self.lb_1_y_value, alignment=Qt.AlignCenter)
        self.vbox_graph.addItem(vertical_spacer)

        self.vbox_graph.addWidget(self.graph_1_z, alignment=Qt.AlignTop)
        self.vbox_graph.addWidget(self.lb_1_z_value, alignment=Qt.AlignCenter)
        self.vbox_graph.addItem(vertical_spacer)

        self.vbox_graph.addWidget(self.graph_2_x, alignment=Qt.AlignTop)
        self.vbox_graph.addWidget(self.lb_2_x_value, alignment=Qt.AlignCenter)
        self.vbox_graph.addItem(vertical_spacer)

        self.vbox_graph.addWidget(self.graph_2_y, alignment=Qt.AlignTop)
        self.vbox_graph.addWidget(self.lb_2_y_value, alignment=Qt.AlignCenter)
        self.vbox_graph.addItem(vertical_spacer)

        self.vbox_graph.addWidget(self.graph_2_z, alignment=Qt.AlignTop)
        self.vbox_graph.addWidget(self.lb_2_z_value, alignment=Qt.AlignCenter)
        self.vbox_graph.addItem(vertical_spacer)

        self.vbox_btns.addWidget(self.lb_gesture)
        self.vbox_btns.addItem(QSpacerItem(0, 400))
        self.vbox_btns.addWidget(self.txb_gesture)
        self.vbox_btns.addWidget(self.btn_start_record)
        self.vbox_btns.addWidget(self.btn_stop_record_and_save_csv)

        self.hbox_main.addLayout(self.vbox_graph)
        self.hbox_main.addLayout(self.vbox_btns)

        self.setLayout(self.hbox_main)

        """ signal and slot """
        self.btn_start_record.clicked.connect(self.on_btn_start_record)
        self.btn_stop_record_and_save_csv.clicked.connect(self.on_btn_stop_record_and_save_csv)

        self.move(300, 200)
        self.setWindowTitle('ADXL345 Finger Gesture')
        self.show()

    def on_btn_stop_record_and_save_csv(self):
        self.is_start_record = False

        csv_filename = self.txb_gesture.text() + '_record.csv'
        f = open(csv_filename, 'w')

        with f:
            writer = csv.writer(f)
            writer.writerows(self.record)

    def on_btn_start_record(self):
        self.is_start_record = True

    def timer_function(self):
        
        # draw data
        self.x = self.x[1:]
        self.x.append(self.x[-1] + 1)

        self.v_1_x = self.v_1_x[1:]
        self.v_1_x.append(accelerometer.acceleration[0])

        self.v_1_y = self.v_1_y[1:]
        self.v_1_y.append(accelerometer.acceleration[1])

        self.v_1_z = self.v_1_z[1:]
        self.v_1_z.append(accelerometer.acceleration[2])

        self.lb_1_x_value.setText('{:.2f}'.format(accelerometer.acceleration[0]))
        self.lb_1_y_value.setText('{:.2f}'.format(accelerometer.acceleration[1]))
        self.lb_1_z_value.setText('{:.2f}'.format(accelerometer.acceleration[2]))

        self.v_2_x = self.v_2_x[1:]
        self.v_2_x.append(accelerometer2.acceleration[0])

        self.v_2_y = self.v_2_y[1:]
        self.v_2_y.append(accelerometer2.acceleration[0])

        self.v_2_z = self.v_2_z[1:]
        self.v_2_z.append(accelerometer2.acceleration[0])

        self.data_line_1_x.setData(self.x, self.v_1_x)
        self.data_line_1_y.setData(self.x, self.v_1_y)
        self.data_line_1_z.setData(self.x, self.v_1_z)

        self.data_line_2_x.setData(self.x, self.v_2_x)
        self.data_line_2_y.setData(self.x, self.v_2_y)
        self.data_line_2_z.setData(self.x, self.v_2_z)

        self.lb_2_x_value.setText('{:.2f}'.format(accelerometer2.acceleration[0]))
        self.lb_2_y_value.setText('{:.2f}'.format(accelerometer2.acceleration[1]))
        self.lb_2_z_value.setText('{:.2f}'.format(accelerometer2.acceleration[2]))

        # classifiy gesture
        y_pred = self.model.predict(np.asarray([[accelerometer.acceleration[0],
                                                 accelerometer.acceleration[1],
                                                 accelerometer.acceleration[2],
                                                 accelerometer2.acceleration[0],
                                                 accelerometer2.acceleration[1],
                                                 accelerometer2.acceleration[2]]]))

        if y_pred[0] == 1:
            self.lb_gesture.setText('open')
        else:
            self.lb_gesture.setText('close')

        # record data
        if self.is_start_record:
            self.record.append([accelerometer.acceleration[0],
                                accelerometer.acceleration[1],
                                accelerometer.acceleration[2],
                                accelerometer2.acceleration[0],
                                accelerometer2.acceleration[1],
                                accelerometer2.acceleration[2],
                                self.txb_gesture.text()])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyWidget()
    sys.exit(app.exec_())
