import sys
from PySide6.QtWidgets import QApplication, QLabel,\
    QVBoxLayout, QLineEdit,QPushButton,QCheckBox,QMessageBox, QWidget

class MainWindow(QWidget) :
    def __init__(self):
        super().__init__()
        self.setWindowTitle("기본 위젯 실습")
        # self.resize(500, 500) # 기본 윈도우 크기 조절

        self.label = QLabel("위치 정보를 입력하세요.")
        self.line_edit = QLineEdit()
        self.checkbox = QCheckBox("위치 정보 전달 동의")
        self.send_button = QPushButton("전송")

        layer = QVBoxLayout()
        layer.addWidget(self.label)
        layer.addWidget(self.line_edit)
        layer.addWidget(self.checkbox)
        layer.addWidget(self.send_button)

        self.setLayout(layer)

        self.send_button.clicked.connect(self.show_message)

    def show_message(self):
        if self.checkbox.isChecked() :
            message = self.line_edit.text()
            print(f"입력 내용 : {message}")
            self.line_edit.clear()
        else :
            error_message = "동의 체크 버튼이 클릭되지 않았습니다."
            QMessageBox.critical(self, '에러',error_message)
            self.line_edit.clear()

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())