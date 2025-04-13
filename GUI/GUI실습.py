import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, \
    QVBoxLayout, QWidget, QLabel, QLineEdit, QCheckBox, QMessageBox

# 1. 창크 조절 실습 (버튼 추가 후 버튼 클릭 함수 생성)
# class MainWindow(QMainWindow) :
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("창 크기 조절 실습")
#         self.resize(500,500)
#
#         # new button
#         self.button = QPushButton("클릭", self)
#         self.button.clicked.connect(self.buttonClicked)
#
#         # 버튼 위치 및 크기 설정
#         self.button.setGeometry(50,50, 200, 50)
#
#     def buttonClicked(self):
#         print("버튼 클릭 되었습니다.")
#
# if __name__ == "__main__" :
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#
#     sys.exit(app.exec())

# 2. 레이블 간단 실습 -> 텍스트나 이미지를 표시하는데 사용되는 위젯
# app = QApplication([])
# window = QWidget()
#
# label = QLabel("안녕하세요.")
# layout = QVBoxLayout()
# layout.addWidget(label)
#
# window.setLayout(layout)
# window.resize(500,500)
# window.show()
# app.exec()

# 3. 텍스트 상자 : 사용자로부터 입력값을 받기 위한 위젯
# app = QApplication([])
# window = QWidget()
#
# line_edit = QLineEdit()
# layout = QVBoxLayout()
# layout.addWidget(line_edit)
#
# window.setLayout(layout)
# window.resize(500,500)
# window.show()
# app.exec()

# 4. Button : 사용자의 동작을 처리하기 위해 클릭 등의 이벤트를 감지하는데 사용되는 위젯
# app = QApplication([])
# window = QWidget()
#
# button = QPushButton("저장")
# layout = QVBoxLayout()
# layout.addWidget(button)
#
# window.setLayout(layout)
# window.resize(500,500)
# window.show()
# app.exec()

# 5. CheckBox : 사용자로부터 선택 여부를 받는데 사용되는 위젯
# app = QApplication([])
# window = QWidget()
#
# checkbox = QCheckBox("동의합니다.")
# layout = QVBoxLayout()
# layout.addWidget(checkbox)
#
# window.setLayout(layout)
# window.resize(500,500)
# window.show()
# app.exec()

# 6. 메시지 박스 : 정보를 표시하거나 사용자에게 메시지를 전달하는데 사용되는 위젯
# app = QApplication([])
# message_box = QMessageBox()
# message_box.setWindowTitle("알림")
# message_box.setText("작업 완료")
# message_box.exec()