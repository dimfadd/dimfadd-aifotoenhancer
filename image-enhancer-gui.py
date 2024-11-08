from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QHBoxLayout, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image
import cv2
import numpy as np
from gfpgan import GFPGANer
import sys

# Initialize the GFPGAN model
model_path = 'E:\\college\\BIG DATA &AI\\AIimage-enhancer\\GFPGAN\\experiments\\pretrained_models\\GFPGANv1.4.pth'
enhancer = GFPGANer(
    model_path=model_path,
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

class ImageEnhancerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Image Enhancer with GFPGAN - PyQt5")
        self.setGeometry(100, 100, 900, 500)

        # Image placeholders
        self.img = None
        self.enhanced_img = None

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Image display area
        self.image_layout = QHBoxLayout()
        self.main_layout.addLayout(self.image_layout)

        # Original image label
        self.original_label = QLabel("Original Image", self)
        self.original_label.setFixedSize(400, 400)
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("border: 1px solid gray;")
        self.image_layout.addWidget(self.original_label)

        # Enhanced image label
        self.enhanced_label = QLabel("Enhanced Image", self)
        self.enhanced_label.setFixedSize(400, 400)
        self.enhanced_label.setAlignment(Qt.AlignCenter)
        self.enhanced_label.setStyleSheet("border: 1px solid gray;")
        self.image_layout.addWidget(self.enhanced_label)

        # Button layout
        self.button_layout = QHBoxLayout()
        self.main_layout.addLayout(self.button_layout)

        # Open button
        self.open_button = QPushButton("Open Image", self)
        self.open_button.setFixedSize(120, 40)
        self.open_button.clicked.connect(self.open_image)
        self.button_layout.addWidget(self.open_button)

        # Enhance button
        self.enhance_button = QPushButton("Enhance Image", self)
        self.enhance_button.setFixedSize(120, 40)
        self.enhance_button.clicked.connect(self.enhance_image)
        self.button_layout.addWidget(self.enhance_button)

        # Save button
        self.save_button = QPushButton("Save Enhanced Image", self)
        self.save_button.setFixedSize(150, 40)
        self.save_button.clicked.connect(self.save_image)
        self.button_layout.addWidget(self.save_button)

    def open_image(self):
        # Open an image file
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            self.img = cv2.imread(file_path)
            self.display_image(self.img, self.original_label)

    def display_image(self, img, label):
        # Convert and display the image on a QLabel
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        qimage = QImage(img_pil.tobytes(), img_pil.width, img_pil.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio))

    def enhance_image(self):
        if self.img is not None:
            _, _, self.enhanced_img = enhancer.enhance(self.img, has_aligned=False, only_center_face=False)
            self.display_image(self.enhanced_img, self.enhanced_label)

    def save_image(self):
        if self.enhanced_img is not None:
            options = QFileDialog.Options()
            file_save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "JPEG Files (*.jpg)", options=options)
            if file_save_path:
                cv2.imwrite(file_save_path, self.enhanced_img)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageEnhancerApp()
    window.show()
    sys.exit(app.exec_())
