from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QComboBox, QPushButton, QTextBrowser, QSpinBox, QDoubleSpinBox
from PyQt5.QtGui import QPixmap, QFont, QIcon
import sys
import cv2
import io
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QRect, QCoreApplication, QTimer
import numpy as np
import torch
import os
# >>ADDITIONAL CODE
from PyQt5.QtMultimedia import QSound
import ultralytics
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath #pt파일 읽기 오류 해결
import datetime # For logging purposes

cameraNo = 0  # 전역변수 cameraNo, 기본값=0 으로 선언
okNG = 'NA'  # 전역변수 양불판정, 기본값=NA 으로 선언

default_encoding = 'utf-8'  # 기본 인코딩을 설정합니다.

sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding=default_encoding, errors='replace')


class ObjectDetection(QThread):
    change_pixmap_signal_YOLO = pyqtSignal(np.ndarray)
    update_OKNG_count = pyqtSignal(tuple) 

    def __init__(self, model_type):
        super().__init__()
        # self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='./Data_lake/75QNED90.pt')  ##수정
        # model = torch.hub.load('ultralytics/yolov5', 'custom', path='./Data_lake/75QNED90.pt') ##after
        # self.model = model ##after
        self.model = torch.hub.load('./yolov5', 'custom', path="./Data_lake/%s" % model_type, source='local')

        self.model.conf = 0.5  # confidence threshold 설정 || Now set same value as the spinbox value
        self.classes = self.model.names
        # >>ADDED CODE: ZOOM SCALE AND
        self.zoom_scale = 1  # Set default
        #self.ng_threshold = 1 # Set default

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:", self.device)
        self._run_flag_YOLO = True

        #>>ADDED CODE-TENTATIVE: NG log "time-out"
        #self.donotlog = False # Start from allowing logging
        #self.qtimer_NGtimeout = QTimer()
        #self.qtimer_NGtimeout.moveToThread(self)
        #self.qtimer_NGtimeout.timeout.connect(self.resetlogtimeout)
        #self.timeout_time = 5000 #Set default: 5 seconds; can be adjusted in the Main thread
        #The Flag solution (based on an estimation of time, rather than real times)
        self.timeout_time = 5 #5 seconds; can be altered from the main thread
        self.timeout_remaining = 0
        self.modelname = model_type[0:len(model_type)-3] #For logging purpose

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
        ok_count=0 #>>ADDED CODE: count OK instances
        ng_count=0 #>>ADDED CODE: count NG instances
        #ng_hidden_remain=self.ng_threshold-1 #>>ADDED CODE; e.g. NG Hidden 

        n = len(labels)
        if n<1: okNG = 'NA' #>>ADDED CIDE: default to NA if no boxes
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)

                # set box colors
                #ADDITIONAL CODE: trigger NA when no boxes
                OKNG = self.class_to_label(labels[i])[0:2]
                if OKNG == 'NG':
                    bgr = (0, 0, 255)  # NG는 빨간색 박스로 설정
                    okNG = 'NG'
                    ng_count+=1 #ADDED CODE: NG count adds by one each NG box
                elif OKNG == 'OK':
                    bgr = (0, 255, 0)  # OK는 초록색 박스로 설정
                    okNG = 'OK'
                    ok_count +=1
                else:  # 효과가 없는듯
                    okNG = 'NA'

                #ADDITIONAL CODE: ensure that if NG exists, status is NG
                if ng_count > 0: okNG = 'NG'
                #elif (ng_count <= 0) & n>=1: okNG = 'OK'
                #else: okNG = 'NA'
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

                # print class label
                print(self.class_to_label(labels[i]))

        #>>ADDED CODE: return ng_count
        return frame,ok_count,ng_count


    def run(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        global cameraNo
        global okNG
        #cap = cv2.VideoCapture(cameraNo)
        # ADDITIONAL CODE: cv2 access resolutions oever 640x480
        cap = cv2.VideoCapture(cameraNo, cv2.CAP_DSHOW)
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000) #Arbitrarily large number for max value
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 4000) #Arbitrarily large number for max value
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #Largest 4:3 ratio 
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) #I cann think of
        while self._run_flag_YOLO:
            ret, frame = cap.read()
            if not ret:
                break

            # >>ADDITIONAL CODE: CALCULATIONS OF FRAME SCALE BASED ON self.zoom_scale
            if self.zoom_scale != 1: frame = self.zoom_frame(frame)

            results = self.score_frame(frame)
            frame,ok_count,ng_count = self.plot_boxes(results, frame)  #ADDITIONAL CODE: ok_count and ng_count for log system
            if ret:
                self.change_pixmap_signal_YOLO.emit(frame)
                #>> ADDITIONAL CODE: NG box count and "timer" move
                self.update_OKNG_count.emit((ok_count,ng_count))
                if self.timeout_remaining > 0: self.timeout_remaining-=1 #subtract if not zero yet

            #>>ADDED CODE: generate log when NG occurs
            if okNG == 'NG':
                if self.timeout_remaining == 0: #If timeout not ongoing           old: not self.donotlog:
                    #Start no-log timer ("Activate")
                    #self.qtimer_NGtimeout.start(self.timeout_time)  #again
                    #self.donotlog = True # do not log until timer below says so
                    self.timeout_remaining = (12 * self.timeout_time) #For each frame, reduce the remaining time based on FPS*time 
                    self.generate_log(frame, results, ok_count, ng_count)

        # shut down capture system
        cap.release()

    def stop_YOLO(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag_YOLO = False
        # self.wait()

    # >>ADDED CODE - ZOOM IN AND ZOOM OUT FUNCTIONS
    # >>Slot for main window zoom buttom
    def zoomin(self):
        # function for zooming in; called by main window button, affects directly QThread
        # Affects a thread variable, tentatively named "self.zoom", valued by default at 1.0 (normal)
        # This function subtracts that value by -0.2, which is to be responded to by the next frame generation
        if self.zoom_scale > 0.2:
            self.zoom_scale -= 0.1
        else:pass

    def zoomout(self):
        # function for zooming in; called by main window button, affects directly QThread
        # Affects a thread variable, tentatively named "self.zoom", valued by default at 1.0 (normal)
        # This function adds that value by +0.2, which is to be responded to by the next frame generations
        if self.zoom_scale < 1:
            self.zoom_scale += 0.1
        else:pass

    def generate_log(self, frame, results, okc, ngc):
        #Output: generation of file {YYMMDD}_{HHMMSS}.log and {YYMMDD}_{HHMMSS}.png, containing box coordinate dumps, and current frame dump
        #Input: box-drawn frame (for JPG, and for calculations) and results, for box details
        #Refer to the text format I made earlier
        
        #Time stamp
        dt = datetime.datetime.now()
        filename_basic = dt.strftime("%y%m%d_%H%M%S") #e.g. 250406_145130; use for log file name
        log_timestamp = dt.strftime("%x %X") #Standard date and time format
        

        #Model name, OK and NG count; use
        mod_tp = f"Model type:\t{self.modelname}" 
        confth = f"Confidence:\t{round(self.model.conf,2)}"
        okcstr = f"Number of OK:\t{okc}"
        ngcstr = f"Number of NG:\t{ngc}"

        #Detection results table:
        tableheader = "Name\tXmin\tYmin\tXmax\tYmax"
        tableborder = "====================================="
        tablecontents = "" #initiate table write
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)

                # print class label
                label = self.class_to_label(labels[i])
                tablecontents += f"{label}\t{x1}\t{y1}\t{x2}\t{y2}\n"

        #Print all into ".\log\{filename_basic}.log"
        with open(f'.\\log\\{filename_basic}.log', mode='w') as f:
            print(log_timestamp, tableborder, mod_tp, confth, okcstr, ngcstr, '', 'Detections:', tableheader, tableborder, tablecontents, sep="\n", file=f)

        #Save frame into ".\log\{filename_basic}.jpg" as JPG file for space efficiency
        cv2.imwrite(f'.\\log\\{filename_basic}.jpg', frame)

    def zoom_frame(self, frame):
        h, w, ch = frame.shape
        centerx, centery = int(h / 2), int(w / 2)
        radx, rady = int(self.zoom_scale * centerx), int(self.zoom_scale * centery)
        minx, maxx = centerx - radx, centerx + radx
        miny, maxy = centery - rady, centery + rady

        frame_cropped = frame[minx:maxx, miny:maxy]
        frame = cv2.resize(frame_cropped, (w, h))
        return frame
        
    def setconfvalue(self, conf):
        #Set model conf by input value
        self.model.conf = conf

    #def resetlogtimeout(self):
    #    self.zoom_scale=1 #Testing to indicate timer has timed
    #    self.donotlog = False
    #    self.qtimer_NGtimeout.stop()

    # >>END OF ADDED CODE


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.disply_width = 1280
        self.display_height = 720
        self.setWindowTitle('Detect_4.4')
        self.setWindowIcon(QIcon('simbol.ico'))
        self.resize(self.disply_width+20, self.display_height+156)
        
        global cameraNo
        global okNG

        # LG 로고 Size 수정
        # create a QLabel widget
        self.image_label = QLabel(self)
        # load the image using QPixmap
        pixmap = QPixmap('logo.png')
        #scaled_pixmap = pixmap.scaled(150, 30)
        # set the pixmap as the image for the label
        self.image_label.setPixmap(pixmap)
        # resize the label to fit the image
        self.image_label.resize(pixmap.width(), pixmap.height())
        # move the label to a desired position
        self.image_label.move(self.disply_width-150, self.display_height+64)

        # Hide labels we can hide
        label1 = QLabel('Select Model', self)
        label1.move(160, 15)
        label2 = QLabel('Camera change', self)
        #label2.move(330, 10)
        label2.move(160, 70)
        # label3 = QLabel('판정 시작/중지', self)
        # label3.move(310, 20)
        # label4 = QLabel('라인 모니터링', self)
        # label4.move(20, 90)
        # label4 = QLabel('양불판정 결과', self)
        # label4.move(550, 20)
        # label6 = QLabel('Camera Control', self)
        # label6.move(685, 90)
        #label7 = QLabel('Made by Jongkwon', self)
        #label7.move(555, 546)
        #label7 = QLabel('*모델명 선택 불가한', self)
        #label7.move(10, 546)
        #label7 = QLabel('BETA 버전입니다', self)
        #label7.move(138, 546)

        # create spinBox that select camera No.
        self.spinBox = QSpinBox(self)
        self.spinBox.setMinimum(0)
        self.spinBox.setSingleStep(1)
        self.spinBox.resize(95, 23)
        #self.spinBox.move(330, 25)
        self.spinBox.move(160, 85)
        self.a = self.spinBox.value()
        self.spinBox.valueChanged.connect(self.value_changed)

        # create start/stop buttons
        start_button = QPushButton('► Start', self)
        stop_button = QPushButton('■ Stop', self)
        start_button.resize(60, 40)
        stop_button.resize(60, 40)
        start_button.move(10, 15)
        stop_button.move(70, 15)
        start_button.clicked.connect(self.startButton)
        stop_button.clicked.connect(self.stopButton)

        # create camera control buttons
        zoom_in_button = QPushButton('+', self)
        zoom_out_button = QPushButton('-', self)
        zoom_in_button.resize(40, 40)
        zoom_out_button.resize(40, 40)
        #zoom_in_button.move(455, 10)
        zoom_in_button.move(90, 70)
        #zoom_out_button.move(495, 10)
        zoom_out_button.move(50, 70)
        zoom_in_button.clicked.connect(self.zoomin)
        zoom_out_button.clicked.connect(self.zoomout)
        # >>CREATE ZOOM PERCENTAGE LABEL
        self.zoom_label = QLabel('100%', self)
        self.zoom_label.move(15, 82)

        # create NG sound object
        # >>ADDITIONAL CODE
        self.NGsound = QSound("NG.wav")

        # >>ADDITIONAL CODE: determine time in seconds for timeout between log writing when NG occur
        timeoutlabel = QLabel('Approximate time between parts',self)
        timeoutlabel.move(360, 15)
        self.timeoutbox = QSpinBox(self)
        self.timeoutbox.setMinimum(1)
        self.timeoutbox.setValue(5) #Default is approximately 5 seconds
        self.timeoutbox.setSingleStep(1)
        self.timeoutbox.resize(95, 23)
        self.timeoutbox.move(360, 30)
        self.timeoutbox.valueChanged.connect(self.setNGtimeout)
        self.timeoutunitlabel = QLabel('Seconds',self)
        self.timeoutunitlabel.move(470,35)

        # >>ADDITIONAL CODE: confidence threshold manual adjust
        conflabel = QLabel('Confidence Threshold',self)
        conflabel.move(360, 70)
        self.confbox = QDoubleSpinBox(self)
        self.confbox.setMinimum(0.1)
        self.confbox.setMaximum(1.0)
        self.confbox.setValue(0.5) #Default conf = 0.5
        self.confbox.setSingleStep(0.05)
        self.confbox.resize(95, 23)
        self.confbox.move(360, 85)
        self.confbox.valueChanged.connect(self.setconf)
        
        # create OKNG count label
        self.ng_count_label = QLabel(f'NG: {0}   ',self)
        #self.ng_count_label.move(10, self.display_height+70)
        self.ng_count_label.move(self.disply_width-230, 62)
        self.ok_count_label = QLabel(f'OK: {0}   ',self)
        self.ok_count_label.move(self.disply_width-230, 42)

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
        self.comboBox.move(160, 30)

        # create textBrowser that shows OK/NG
        self.textBrowser = QTextBrowser(self)
        self.textBrowser.setAcceptRichText(True)  # 서식있는 텍스트 가능
        #self.textBrowser.setStyleSheet("background: transparent;")
        #self.textBrowser.resize(110, 56)
        self.textBrowser.resize(180, 116)
        self.textBrowser.move(self.disply_width-170, 62)
        
        # create the label that holds the image_YOLO
        self.video_label_YOLO = QLabel(self)
        self.video_label_YOLO.resize(self.disply_width, self.display_height)
        self.video_label_YOLO.move(10, 122)
        self.video_label_YOLO.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.video_thread_YOLO()
        self.textBrowser.raise_() # Ensure OK/NG indicator stays on top

    def video_thread_YOLO(self):
        # create the video capture thread
        self.model_type = self.comboBox.currentText()
        self.thread_YOLO = ObjectDetection(self.model_type)
        # connect its signal to the update_image slot
        self.thread_YOLO.change_pixmap_signal_YOLO.connect(self.update_image)
        self.thread_YOLO.update_OKNG_count.connect(self.updateOKNGcount)
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
                                                                "<p align=\"center\">""<b>""<span style=\" font-size:60pt; color:red;\">NG</span></b></p>",
                                                                None)) ##font-size: 30pt --original
            # >>ADDITIONAL CODE
            self.makeNGsound()
        elif okNG == 'OK':
            self.textBrowser.setHtml(QCoreApplication.translate("mainWindow",
                                                                "<p align=\"center\">""<b>""<span style=\" font-size:60pt; color:lime;\">OK</span></b></p>",
                                                                None))
        elif okNG == 'NA':
            self.textBrowser.setHtml(QCoreApplication.translate("mainWindow",
                                                                "<p align=\"center\">""<b>""<span style=\" font-size:60pt; color:gray;\">NA</span></b></p>",
                                                                None))

    def stopButton(self):
        global okNG
        okNG = 'NA'
        self.textBrowser.setHtml(QCoreApplication.translate("mainWindow",
                                                            "<p align=\"center\">""<b>""<span style=\" font-size:60pt; color:gray;\">NA</span></b></p>",
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
        # self.thread_YOLO.model = torch.hub.load('ultralytics/yolov5', 'custom', path="./Data_lake/%s" % self.model_type) ##수정
        # model_path = "./Data_lake/%s" % self.model_type   ##after
        # model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)   ##after
        # self.thread_YOLO.model = model    ##after
        self.thread_YOLO.model = torch.hub.load('./yolov5', 'custom', path="./Data_lake/%s" % self.model_type, source='local')
        self.thread_YOLO.classes = self.thread_YOLO.model.names
        print('Change model to %s' % x)
        # self.statistic_msg('Change model to %s' % x)

        #ADDITIONAL CODE: UPDATE MODEL NAME FOR LOG
        self.thread_YOLO.modelname = self.model_type[0:len(self.model_type)-3]

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

    # >>ADDITIONAL CODE
    def makeNGsound(self):
        # To ensure smooth looping
        if self.NGsound.isFinished():
            self.NGsound.play()

    def zoomin(self):
        self.thread_YOLO.zoomin()
        self.updatezoomlabel()

    def zoomout(self):
        # call thread function
        self.thread_YOLO.zoomout()
        self.updatezoomlabel()

    def updatezoomlabel(self):
        p = f"{int(1 / self.thread_YOLO.zoom_scale * 100)}%"
        self.zoom_label.setText(p)

    def setNGtimeout(self):
        self.thread_YOLO.timeout_time = self.timeoutbox.value()
        
    def setconf(self):
        self.thread_YOLO.setconfvalue(self.confbox.value())

    @pyqtSlot(tuple)
    def updateOKNGcount(self, okng_count):
        # Receives a signal from the Object Detection Thread, number of NG boxes
        self.ok_count_label.setText(f"OK: {okng_count[0]}")
        self.ng_count_label.setText(f"NG: {okng_count[1]}")

    def resizeEvent(self, newSize):
        #Resizes the display and repositions certain parts
        h=self.height()
        w=self.width()
        self.display_height = h-156
        self.disply_width = w-20
        self.video_label_YOLO.resize(self.disply_width, self.display_height)
        self.image_label.move(self.disply_width-150, self.display_height+124)
        #self.ng_count_label.move(10, self.display_height+130)
        self.ng_count_label.move(self.disply_width-230, 62) 
        self.ok_count_label.move(self.disply_width-230, 42) 
        self.textBrowser.move(self.disply_width-170, 4)
        #if self.disply_width >= self.display_height*(16/9):
        #    self.textBrowser.move((int((w+self.display_height*(16/9))/2)-self.textBrowser.width()), 62)
        #else:
        #    self.textBrowser.move(w-self.textBrowser.width()-10, (62+int((self.display_height-self.disply_width*(9/16))/2)))
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    with open('./theme.qss') as f:
        _style = f.read()
        app.setStyleSheet(_style)
    sys.exit(app.exec_())

