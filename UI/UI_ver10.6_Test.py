from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QComboBox, QPushButton, QTextBrowser, QSpinBox
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QRect, QCoreApplication, QTimer
import numpy as np
import torch
import os


cameraNo = 0  # 전역변수 cameraNo, 기본값=0 으로 선언
okNG = 'NA'  # 전역변수 양불판정, 기본값=NA 으로 선언


class ObjectDetection(QThread):
    change_pixmap_signal_YOLO = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        #self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='./Data_lake/75QNED90.pt')  ##수정
        #model = torch.hub.load('ultralytics/yolov5', 'custom', path='./Data_lake/75QNED90.pt') ##after
        # self.model = model ##after
        self.model = torch.hub.load('./yolov5', 'custom', path='./Data_lake/75QNED90.pt', source='local')

        self.model.conf = 0.25  # confidence threshold 설정
        self.classes = self.model.names

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:", self.device)
        self._run_flag_YOLO = True

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):  # For a given label value, return corresponding string label.(numeric>string)
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        global okNG
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)

                # set box colors
                OKNG = self.class_to_label(labels[i])[0:2]
                if OKNG == 'NG':
                    bgr = (0, 0, 255)  # NG는 빨간색 박스로 설정
                    okNG = 'NG'
                elif OKNG == 'OK':
                    bgr = (0, 255, 0)  # OK는 초록색 박스로 설정
                    okNG = 'OK'
                else:  # 효과가 없는듯
                    okNG = 'NA'

                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

                # print class label
                print(self.class_to_label(labels[i]))

        return frame

    def run(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        global cameraNo
        global okNG
        cap = cv2.VideoCapture(cameraNo)
        while self._run_flag_YOLO:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            if ret:
                self.change_pixmap_signal_YOLO.emit(frame)
        # shut down capture system
        cap.release()

    def stop_YOLO(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag_YOLO = False
        # self.wait()


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MOBASU_BETA')
        self.resize(680, 600)
        self.disply_width = 640
        self.display_height = 480
        global cameraNo
        global okNG

        label1 = QLabel('모델 선택',self)
        label1.move(20, 20)
        label2 = QLabel('카메라 번호', self)
        label2.move(215, 20)
        label3 = QLabel('판정 시작/중지', self)
        label3.move(310, 20)
        label4 = QLabel('라인 모니터링', self)
        label4.move(20, 90)
        label4 = QLabel('양불판정 결과', self)
        label4.move(550, 20)


        # create spinBox that select camera No.
        self.spinBox = QSpinBox(self)
        self.spinBox.setMinimum(0)
        self.spinBox.setSingleStep(1)
        self.spinBox.resize(75, 23)
        self.spinBox.move(215,35)
        self.a = self.spinBox.value()
        self.spinBox.valueChanged.connect(self.value_changed)

        # create start/stop buttons
        start_button = QPushButton('Start',self)
        stop_button = QPushButton('Stop',self)
        start_button.resize(75, 23)
        stop_button.resize(75, 23)
        start_button.move(310,35)
        stop_button.move(390,35)
        start_button.clicked.connect(self.startButton)
        stop_button.clicked.connect(self.stopButton)

        # create comboBox that select model name
        self.comboBox = QComboBox(self)
        # search models automatically
        self.comboBox.clear()
        self.pt_list = os.listdir('./Data_lake')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./Data_lake/' + x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)
        # update models
        self.comboBox.currentTextChanged.connect(self.change_model)
        self.comboBox.resize(180, 22)
        self.comboBox.move(20,35)

        # create textBrowser that shows OK/NG
        self.textBrowser = QTextBrowser(self)
        self.textBrowser.setAcceptRichText(True)  # 서식있는 텍스트 가능
        self.textBrowser.resize(110, 56)
        self.textBrowser.move(550,35)

        # create the label that holds the image_YOLO
        self.video_label_YOLO = QLabel(self)
        self.video_label_YOLO.resize(self.disply_width, self.display_height)
        self.video_label_YOLO.move(20,105)
        self.video_thread_YOLO()

    def video_thread_YOLO(self):
        # create the video capture thread
        self.thread_YOLO = ObjectDetection()
        # connect its signal to the update_image slot
        self.thread_YOLO.change_pixmap_signal_YOLO.connect(self.update_image)
        # start the thread
        self.thread_YOLO.start()

    def value_changed(self):
        global cameraNo
        cameraNo = self.spinBox.value()
        print('cameraNo :', cameraNo)
        self.thread_YOLO.stop_YOLO()
        self.video_thread_YOLO()

    def startButton(self, state):
        global okNG
        self.video_thread_YOLO()

        print('start...')

    def showOKNG(self):
        global okNG
        print('showOKNG:', okNG)
        if okNG == 'NG':
            self.textBrowser.setHtml(QCoreApplication.translate("mainWindow",
                                                                "<p align=\"center\">""<b>""<span style=\" font-size:30pt; color:red;\">NG</span></b></p>",
                                                                None))
        elif okNG == 'OK':
            self.textBrowser.setHtml(QCoreApplication.translate("mainWindow",
                                                                "<p align=\"center\">""<b>""<span style=\" font-size:30pt; color:red;\">OK</span></b></p>",
                                                                None))
        elif okNG == 'NA':
            self.textBrowser.setHtml(QCoreApplication.translate("mainWindow",
                                                                "<p align=\"center\">""<b>""<span style=\" font-size:30pt; color:red;\">NA</span></b></p>",
                                                                None))

    def stopButton(self):
        global okNG
        okNG = 'NA'
        self.textBrowser.setHtml(QCoreApplication.translate("mainWindow",
                                                            "<p align=\"center\">""<b>""<span style=\" font-size:30pt; color:red;\">NA</span></b></p>",
                                                            None))
        self.thread_YOLO.stop_YOLO()
        print('stopped...')

    def closeEvent(self, event):
        self.thread_YOLO.stop_YOLO()
        event.accept()

    def search_pt(self):
        pt_list = os.listdir('./Data_lake')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./Data_lake/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        #self.thread_YOLO.model = torch.hub.load('ultralytics/yolov5', 'custom', path="./Data_lake/%s" % self.model_type) ##수정
        #model_path = "./Data_lake/%s" % self.model_type   ##after
        #model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)   ##after
        #self.thread_YOLO.model = model    ##after
        self.model = torch.hub.load('./yolov5', 'custom', path="./Data_lake/%s" % self.model_type)
        self.thread_YOLO.classes = self.thread_YOLO.model.names
        print('Change model to %s' % x)
        # self.statistic_msg('Change model to %s' % x)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label_YOLO.setPixmap(qt_img)  # 여기서 Label과 실시간 YOLO영상 연결
        self.showOKNG()  # 여기서 실시간 OKNG디스플레이 실행

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