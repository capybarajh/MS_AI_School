"""
시그널과 슬롯의 연결 응용 버전 실습
1. 사용자로부터 나이, 성별, 국가 입력 받기
2. 보기버튼누르면입력정보를보여준다
3. 입력정보창에서는(나이성별국가)입력받은정보출력
- 추가 적으로 저장 버튼, 닫기 버튼 불러오기 버튼 추가
4. 저장된 정보 창에서는 리스트 박스를 이용하여
(중복 방지를 위한 ID 추가 여기서는 Time 함수를 이용)
DB 처럼 auto number 없음 저장된 정보를 읽어 오기
- ID / 나이 / 성별 / 국가
"""
import sys
import csv
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout,\
    QLabel, QLineEdit, QPushButton, QDialog, QMessageBox, QListWidget


class InputWindow(QWidget) :
    # 사용자로부터 나이, 성별, 국가 입력 받기
    # 보기버튼
    def __init__(self):
        super().__init__()

        # line_edit
        self.age_line_edit = QLineEdit()
        self.gender_line_edit = QLineEdit()
        self.country_line_edit = QLineEdit()

        # button
        self.view_button = QPushButton("view")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(QLabel("age : "))
        layout.addWidget(self.age_line_edit)

        layout.addWidget(QLabel("gender : "))
        layout.addWidget(self.gender_line_edit)

        layout.addWidget(QLabel("country : "))
        layout.addWidget(self.country_line_edit)
        layout.addWidget(self.view_button)

        self.setLayout(layout)

        self.view_button.clicked.connect(self.show_info)

    def show_info(self):
        age = self.age_line_edit.text()
        gender = self.gender_line_edit.text()
        country = self.country_line_edit.text()

        info_window = InfoWindow(age, gender, country)
        info_window.setModal(True)
        info_window.exec()



class InfoWindow(QDialog) :
    def __init__(self, age, gender, country):
        super().__init__()
        self.setWindowTitle("info")

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"age : {age}"))
        layout.addWidget(QLabel(f"gender : {gender}"))
        layout.addWidget(QLabel(f"country : {country}"))

        # button
        save_button = QPushButton("save")
        close_button = QPushButton("close")
        load_button = QPushButton("load")

        layout.addWidget(save_button)
        layout.addWidget(close_button)
        layout.addWidget(load_button)

        self.setLayout(layout)

        # connect
        save_button.clicked.connect(lambda : self.save_info(age, gender, country))
        close_button.clicked.connect(self.close)
        load_button.clicked.connect(self.load_info)

    def save_info(self, age, gender, country):
        data = [generate_id(), age, gender, country]
        try :
            with open("info.csv" , 'a' , newline="", encoding='utf-8') as f :
                writer = csv.writer(f)
                writer.writerow(data)
            QMessageBox.information(self, "save ok" , "info save ok")
        except Exception as e :
            QMessageBox.critical(self, "save no", f"{str(e)}")

    def load_info(self):
        try :
            with open("info.csv", 'r') as f:
                reader = csv.reader(f)
                lines = [line for line in reader]

            if len(lines) > 0 :
                list_winow = ListWindow(lines)
                list_winow.exec()
            else :
                QMessageBox.information(self, "load info no")

        except Exception as e :
            QMessageBox.information(self, "load no", f"{str(e)}")

class ListWindow(QDialog) :
    def __init__(self, lines):
        super().__init__()
        self.setWindowTitle("save info")

        list_widget = QListWidget()
        for line in lines :
            item = f"ID {line[0]}, age {line[1]}, gender {line[2]}, country{line[3]}"
            list_widget.addItem(item)

        layout = QVBoxLayout()
        layout.addWidget(list_widget)

        self.setLayout(layout)

def generate_id() :
    import time
    return str(int(time.time()))

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    input_window = InputWindow()
    input_window.show()
    app.exec()