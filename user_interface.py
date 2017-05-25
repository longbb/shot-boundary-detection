import sys
import os
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from video import Video
from threading import Thread
from multiprocessing import Process
import time
import numpy as np
import cv2

class UserInterface(object):
    application = None
    widget = None
    sub_widget = None
    scroll = None
    select_file_button = None
    file_name = None
    video = None
    labels_array = []
    pixmap_array = []
    open_button_array = []

    def __init__(self):
        self.application = QApplication(sys.argv)

        self.widget = QWidget()
        self.widget.setFixedSize(600, 500)
        self.widget.setWindowTitle('Shot boundary ditection')

        self.select_file_button = QPushButton(self.widget)
        self.select_file_button.setText('Select video')
        self.select_file_button.resize(150, 50)
        self.select_file_button.move(175,225)
        self.select_file_button.clicked.connect(self.select_file)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.widget)
        # self.scroll.setWidgetResizable(True)
        self.scroll.move(250, 0)
        self.scroll.setFixedHeight(500)
        self.scroll.setFixedWidth(600)


        self.widget.show()
        self.scroll.show()

        sys.exit(self.application.exec_())

    def select_file(self):
        new_widget = QWidget()
        self.file_name = QFileDialog.getOpenFileName(new_widget, 'Open File', '/')
        self.select_file_button.setVisible(False)

        time.sleep(2)
        # worker = Thread(target=self.shot_boundary_detection, args=(str(self.file_name),))
        # worker.start()
        # worker.join()
        self.shot_boundary_detection(str(self.file_name))

        max_column = 100 / 3
        max_length = 325 * (max_column + 2)
        self.widget.setFixedSize(600, max_length)

        for i in range(0, len(self.video.key_frame_array)):
            image_path = './test_frame/frame{}.jpg'.format(self.video.key_frame_array[i])
            position_x = 25
            position_y = 50 + i * 175
            self.add_image_to_widget(image_path, 300, 300, position_x, position_y)

    def shot_boundary_detection(self, file_name):
        self.video = Video(file_name)
        self.video.detect_shot_boundary_2(0.5, 0.2, 3)

    def add_image_to_widget(self, image_path, width, height, position_x, position_y):
        label = QLabel(self.widget)
        label.resize(width, height)
        label.move(position_x, position_y)
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(width, height, Qt.KeepAspectRatio)
        label.setPixmap(pixmap)
        label.setVisible(True)
        self.labels_array.append(label)
        open_shot_button = QPushButton(self.widget)
        open_shot_button.setText('Show Shot')
        open_shot_button.resize(150, 50)
        open_shot_button.move(position_x + 350, position_y + 125)
        shot_position = len(self.open_button_array)
        open_shot_button.clicked.connect(lambda: self.show_shot(shot_position))
        open_shot_button.setVisible(True)
        self.open_button_array.append(open_shot_button)

    def show_shot(self, shot_position):
        worker = Process(target=self.cv2_show_shot, args=(shot_position,))
        worker.start()

    def cv2_show_shot(self, shot_position):
        shot_path = './shot/shot' + str(shot_position) + '.avi'
        os.system('python play_shot.py ' + shot_path)

if __name__ == '__main__':
    gui = UserInterface()
