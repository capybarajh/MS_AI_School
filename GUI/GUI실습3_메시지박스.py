# PySide6에서 파일 다이얼로그와 메시지 박스 소개
import sys
from PySide6.QtWidgets import QApplication,\
    QWidget, QVBoxLayout, QPushButton, QMessageBox


class MainWindow(QWidget) :
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Message Box Ex")

        layout = QVBoxLayout()

        info_button = QPushButton("info message")
        info_button.clicked.connect(self.show_info_message)
        layout.addWidget(info_button)

        warning_button = QPushButton("Waring message")
        warning_button.clicked.connect(self.show_warning_message)
        layout.addWidget(warning_button)

        question_button = QPushButton("Question message")
        question_button.clicked.connect(self.show_question_message)
        layout.addWidget(question_button)

        self.setLayout(layout)

    def show_info_message(self):
        QMessageBox.information(self, "info", "info message",
                                QMessageBox.Ok, QMessageBox.Close)

    def show_warning_message(self):
        QMessageBox.warning(self, "warning", "warning message",
                                QMessageBox.Ok, QMessageBox.Close)

    def show_question_message(self):
        result = QMessageBox.question(self, "question", "question message ?",
                                QMessageBox.Yes | QMessageBox.No)
        if result == QMessageBox.Yes :
            QMessageBox.information(self, "question", "Yes", QMessageBox.Ok)
        else :
            QMessageBox.information(self, "question", "No", QMessageBox.Ok)

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    main_windo = MainWindow()
    main_windo.show()
    app.exec()

