from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QComboBox, \
    QPushButton, QTextBrowser, QLineEdit, QSpinBox, QCheckBox, QFileDialog
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QRect, QCoreApplication
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

modelName = '배면 빛누출'  # 전역변수 모델명, 기본값=55QNED80 으로 선언
okNG = 'NA'  # 전역변수 양불판정, 기본값=NA 으로 선언

# CNN 모델 로드
@tf.autograph.experimental.do_not_convert
def load_cnn_model():
    try:
        model = load_model('ox_class_cnn.h5')
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_cnn_model()

@tf.autograph.experimental.do_not_convert
def predict_image(image):
    try:
        # 모델 입력 크기에 맞게 이미지 크기 조정
        image = cv2.resize(image, (400, 256))  # (width, height) 순서로 크기 조정
        print(f"Resized image shape: {image.shape}")
        image = image / 255.0  # 정규화
        print(f"Normalized image: {image}")
        image = np.expand_dims(image, axis=0)  # 배치 차원 추가
        print(f"Image with batch dimension: {image.shape}")
        predictions = model.predict(image)
        print(f"Predictions: {predictions}")  # 예측 결과 출력
        return np.argmax(predictions)
    except Exception as e:
        print(f"Error in predict_image: {e}")
        return -1

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Processing')
        self.resize(900, 500)
        self.disply_width = 640
        self.display_height = 480
        grid = QGridLayout()
        self.setLayout(grid)
        global okNG
        grid.addWidget(QLabel('판정 종류 선택'), 0, 0)
        grid.addWidget(QLabel(''), 0, 1)
        grid.addWidget(QLabel(' '), 0, 2)
        #
        grid.addWidget(QLabel('사진 합치기'), 0, 3)
        grid.addWidget(QLabel(''), 2, 0)
        grid.addWidget(QLabel('이미지 확인'), 3, 0)
        grid.addWidget(QLabel('판정 결과'), 3, 3)

        # Create comboBox that selects model name
        comboBox = QComboBox(self)
        comboBox.addItem('배면 빛누출')
        comboBox.resize(111, 22)
        grid.addWidget(comboBox, 1, 0)
        comboBox.activated[str].connect(self.onActivated)

        # Create textBrowser that shows OK/NG
        self.textBrowser = QTextBrowser()
        self.textBrowser.setAcceptRichText(True)
        self.textBrowser.resize(150, 120)
        self.textBrowser.setHtml(QCoreApplication.translate("mainWindow",
                                                            "<p align=\"center\"><b><span style=\" font-size:30pt; color:red;\">NA</span></b></p>",
                                                            None))
        grid.addWidget(self.textBrowser, 4, 3)

        # Create the label that holds the image
        self.video_label = QLabel(self)
        self.video_label.resize(self.disply_width, self.display_height)
        grid.addWidget(self.video_label, 4, 0)

        # Create load image button
        load_image_button = QPushButton('이미지 불러오기')
        grid.addWidget(load_image_button, 4, 3)
        load_image_button.clicked.connect(self.load_image)

    def onActivated(self, text):
        global modelName
        modelName = text
        print('model name : ', modelName)

    def closeEvent(self, event):
        event.accept()

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        try:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
            return QPixmap.fromImage(p)
        except Exception as e:
            print(f"Error in convert_cv_qt: {e}")
            return None

    def load_image(self):
        """Load an image file and display it"""
        try:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                       "Image Files (*.png *.jpg *.bmp);;All Files (*)",
                                                       options=options)
            if file_name:
                cv_img = cv2.imread(file_name)
                qt_img = self.convert_cv_qt(cv_img)
                if qt_img:
                    self.video_label.setPixmap(qt_img)
                    # CNN 판정 수행
                    global okNG
                    result = predict_image(cv_img)
                    print(f"Prediction result index: {result}")  # 예측 결과 인덱스 출력
                    if result == 0:
                        okNG = 'OK'
                    elif result == 1:
                    else:
                        okNG = 'Error'
                    self.textBrowser.setHtml(QCoreApplication.translate("mainWindow",
                                                                        f"<p align=\"center\"><b><span style=\" font-size:30pt; color:red;\">{okNG}</span></b></p>",
                                                                        None))
                    print(f'Prediction result: {okNG}')
        except Exception as e:
            print(f"Error in load_image: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())