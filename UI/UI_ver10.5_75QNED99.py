from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QComboBox, QPushButton, QTextBrowser, QSpinBox
from PyQt5.QtGui import QPixmap, QIcon
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

    #def __init__(self):
    #    super().__init__()
    #    self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='./Data_lake/75QNED90.pt')
    def __init__(self):
        super().__init__()
        model_path = './Data_lake/75QNED90_NGOK_update.pt'  # 로컬 모델 파일 경로
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

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

    def plot_boxes(self, results, frame):  ## 결과물 표시
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

                # set box colors : ##여기부터 UI
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

    def run(self):   ##비디오 프레임 읽기, 결과 출력
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        global cameraNo
        global okNG
        cap = cv2.VideoCapture(cameraNo)  ##비디오 프레임읽기
        while self._run_flag_YOLO:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.score_frame(frame)  ##결과 출력
            frame = self.plot_boxes(results, frame)
            if ret:
                self.change_pixmap_signal_YOLO.emit(frame)
        # shut down capture system  ##캡쳐 시스템 종료
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
        self.setWindowIcon(QIcon('symbol.ico'))
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


        # create spinBox that select camera No.  ##카메라 번호 선택창 수정
        self.spinBox = QSpinBox(self)
        self.spinBox.setMinimum(0)
        self.spinBox.setSingleStep(1)
        self.spinBox.resize(75, 23)
        self.spinBox.move(215,35)
        self.a = self.spinBox.value()
        self.spinBox.valueChanged.connect(self.value_changed)

        # create start/stop buttons  ##Start / Stop 버튼 수정
        start_button = QPushButton('Start',self)
        stop_button = QPushButton('Stop',self)
        start_button.resize(75, 23)
        stop_button.resize(75, 23)
        start_button.move(310,35)
        stop_button.move(390,35)
        start_button.clicked.connect(self.startButton)
        stop_button.clicked.connect(self.stopButton)

        # create comboBox that select model name  ##모델명선택 박스 ***
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

        # create textBrowser that shows OK/NG   ##OK/NG 창 표시 text browser
        self.textBrowser = QTextBrowser(self)
        self.textBrowser.setAcceptRichText(True)  # 서식있는 텍스트 가능
        self.textBrowser.resize(110, 56)
        self.textBrowser.move(550,35)

        # create the label that holds the image_YOLO  ##Yolo로 검출된 이미지 표시 label 생성 ***
        self.video_label_YOLO = QLabel(self)
        self.video_label_YOLO.resize(self.disply_width, self.display_height)
        self.video_label_YOLO.move(20,105)
        self.video_thread_YOLO()

    def video_thread_YOLO(self):   #video_thread_YOLO 메서드 정의 -> YOLO로 검출된 이미지를 실시간으로 업데이트 하기 위한 스레드를 생성하고 시작
        # create the video capture thread
        self.thread_YOLO = ObjectDetection()
        # connect its signal to the update_image slot
        self.thread_YOLO.change_pixmap_signal_YOLO.connect(self.update_image)
        # start the thread
        self.thread_YOLO.start()

    def value_changed(self): ##value_changed 메서드 정의 -> camera변경
        global cameraNo
        cameraNo = self.spinBox.value()
        print('cameraNo :', cameraNo)
        self.thread_YOLO.stop_YOLO()
        self.video_thread_YOLO()

    def startButton(self, state):   ##start 버튼 정의
        global okNG
        self.video_thread_YOLO()  ##start를 누르면 객체감지 및 추적 시작

        print('start...')

    def showOKNG(self):   ##NG OK NG 값 출력 *** -> 수정 필요(NG가 나올때 알람음 등 필요)
        global okNG
        print('showOKNG:', okNG)
        if okNG == 'NG':  ##NG 출력 조건
            self.textBrowser.setHtml(QCoreApplication.translate("mainWindow",
                                                                "<p align=\"center\">""<b>""<span style=\" font-size:30pt; color:red;\">NG</span></b></p>",
                                                                None))
        elif okNG == 'OK':  ##OK 출력 조건
            self.textBrowser.setHtml(QCoreApplication.translate("mainWindow",
                                                                "<p align=\"center\">""<b>""<span style=\" font-size:30pt; color:blue;\">OK</span></b></p>",
                                                                None))
        elif okNG == 'NA':  ##OK 출력 조건
            self.textBrowser.setHtml(QCoreApplication.translate("mainWindow",
                                                                "<p align=\"center\">""<b>""<span style=\" font-size:30pt; color:black;\">NA</span></b></p>",
                                                                None))

    def stopButton(self):   ##stop 버튼 정의
        global okNG
        okNG = 'NA'
        self.textBrowser.setHtml(QCoreApplication.translate("mainWindow",
                                                            "<p align=\"center\">""<b>""<span style=\" font-size:30pt; color:black;\">NA</span></b></p>",
                                                            None))
        self.thread_YOLO.stop_YOLO()
        print('stopped...')

    def closeEvent(self, event):  ##closeEvent -> 윈도우가 닫힐때
        self.thread_YOLO.stop_YOLO()
        event.accept()

    def search_pt(self):   ##pt 확장자 파일 검색  -> .pt가 안보이도록 수정 필요
        pt_list = os.listdir('./Data_lake')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./Data_lake/' + x))   ##x는 pt_list.sort 메서드의 key 매개변수에 전달되는 람다 함수에서 사용되는 변수, 파일크기순 정렬
        #pt_list.sort() 오름차순 정렬, 작동여부 확인 필요 ***

        if pt_list != self.pt_list:  #pt_list와 selt.pt_list가 다를때 self.pt_list를 업데이트하고 combobox 초기화 후 pt_list 추가  무슨의미지? ***
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def change_model(self, x):   # *** 오프라인 작동을 위한 수정 필요
        self.model_type = self.comboBox.currentText()
        #model_path = os.path.join('./Data_lake', self.model_type) ##GPT
        #self.thread_YOLO.model = torch.load(model_path) ##GPT
        #self.thread_YOLO.classes = self.thread_YOLO.model.names  ##GPT
        self.thread_YOLO.model = torch.hub.load('ultralytics/yolov5', 'custom', path="./Data_lake/%s" % self.model_type)
        self.thread_YOLO.classes = self.thread_YOLO.model.names
        print('Change model to %s' % x)
        # self.statistic_msg('Change model to %s' % x)

    @pyqtSlot(np.ndarray)   ## 실시간 이미지 업데이트, OKNG 디스플레이
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label_YOLO.setPixmap(qt_img)  # 여기서 Label과 실시간 YOLO영상 연결
        self.showOKNG()  # 여기서 실시간 OKNG디스플레이 실행

    def convert_cv_qt(self, cv_img):   ##OpenCV 이미지를 QPixmap으로 변환하는 convert_cv_qt 메서드
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == '__main__':   ##PyQt 애플리케이션 실행
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())