from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QComboBox, QPushButton, QTextBrowser, QLineEdit, QSpinBox, QCheckBox
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QRect, QCoreApplication
import numpy as np

cameraNo = 0 #전역변수 cameraNo, 기본값=0 으로 선언
modelName = '55QNED80' #전역변수 모델명, 기본값=55QNED80 으로 선언
okNG = 'NA' #전역변수 양불판정, 기본값=NA 으로 선언

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        global cameraNo
        # capture from web cam
        cap = cv2.VideoCapture(cameraNo)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        #self.wait()

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('UI_ver2')
        self.resize(900,500)
        self.disply_width = 640
        self.display_height = 480
        grid = QGridLayout()
        self.setLayout(grid)
        global cameraNo
        global okNG

        grid.addWidget(QLabel('모델 선택'), 0, 0)
        grid.addWidget(QLabel(''), 0, 1)
        grid.addWidget(QLabel('카메라 번호 선택'), 0, 2)
        grid.addWidget(QLabel('판정 시작/중지'), 0, 3)
        grid.addWidget(QLabel(''), 2, 0)
        grid.addWidget(QLabel('라인 모니터링'), 3, 0)
        grid.addWidget(QLabel('양불판정 결과'), 3, 3)

        # create spinBox that select camera No.
        self.spinBox = QSpinBox()
        self.spinBox.setMinimum(0)
        self.spinBox.setSingleStep(1)
        grid.addWidget(self.spinBox, 1, 2)
        self.a = self.spinBox.value()
        self.spinBox.valueChanged.connect(self.value_changed)

        # create start/stop buttons
        start_button = QPushButton('Start')
        stop_button = QPushButton('Stop')
        start_button.resize(75,23)
        stop_button.resize(75,23)
        grid.addWidget(start_button, 1, 3)
        grid.addWidget(stop_button, 2, 3)
        start_button.clicked.connect(self.startButton)
        stop_button.clicked.connect(self.stopButton)

        # create comboBox that select model name
        comboBox = QComboBox(self)
        comboBox.addItem('55QNED80')
        comboBox.addItem('75QNED90')
        comboBox.addItem('32LR65')
        comboBox.resize(111,22)
        grid.addWidget(comboBox, 1, 0)
        comboBox.activated[str].connect(self.onActivated)

        # create textBrowser that shows OK/NG
        self.textBrowser = QTextBrowser()
        self.textBrowser.setAcceptRichText(True)
        self.textBrowser.resize(150,120)
        self.textBrowser.setHtml(QCoreApplication.translate("mainWindow","<p align=\"center\">""<b>""<span style=\" font-size:30pt; color:red;\">NA</span></b></p>",None))
        grid.addWidget(self.textBrowser, 4, 3)

        # create the label that holds the image
        self.video_label = QLabel(self)
        self.video_label.resize(self.disply_width, self.display_height)
        grid.addWidget(self.video_label, 4, 0)
        self.video_thread()

        # create CheckBox for NG - 추후 연동후 삭제 예정
        cb = QCheckBox('NG', self)
        cb.stateChanged.connect(self.startButton)
        grid.addWidget(cb, 1, 5)

    def video_thread(self):
        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def value_changed(self):
        global cameraNo
        cameraNo = self.spinBox.value()
        print('cameraNo :', cameraNo)
        self.thread.stop()
        self.video_thread()

    def onActivated(self, text):
        global modelName
        modelName = text
        print('model name : ', modelName)

    def startButton(self, state):
        global okNG
        okNG = 'OK'

        if state == Qt.Checked: # 체크박스 부분  - 연동후 수정 예정.
            okNG = 'NG'
        else:
            okNG = 'OK'

        if okNG == 'OK':
            self.textBrowser.setHtml(QCoreApplication.translate("mainWindow","<p align=\"center\">""<b>""<span style=\" font-size:30pt; color:red;\">OK</span></b></p>",None))
        elif okNG == 'NG':
            self.textBrowser.setHtml(QCoreApplication.translate("mainWindow","<p align=\"center\">""<b>""<span style=\" font-size:30pt; color:red;\">NG</span></b></p>",None))
        print(okNG,'After start...')

    def stopButton(self):
        global  okNG
        okNG = 'NA'
        self.textBrowser.setHtml(QCoreApplication.translate("mainWindow","<p align=\"center\">""<b>""<span style=\" font-size:30pt; color:red;\">NA</span></b></p>",None))
        print('stopped...')

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())