import sys
import csv
from PySide6.QtWidgets import QApplication, QWidget,\
    QVBoxLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QListWidget, QMessageBox


class MainWindow(QWidget) :
    def __init__(self):
        super().__init__()

        self.setWindowTitle("UI")
        self.resize(500, 250)

        # Group Box
        group_box1 = QGroupBox("info")
        group_box2 = QGroupBox("input view")
        group_box3 = QGroupBox("save & load")

        # label setting
        self.label_id = QLabel("id : ")
        self.label_age = QLabel("age : ")
        self.label_gender = QLabel("gender : ")
        self.label_country = QLabel("country : ")

        # input label setting
        self.line_edit_id = QLineEdit()
        self.line_edit_age = QLineEdit()
        self.line_edit_gender = QLineEdit()
        self.line_edit_country = QLineEdit()

        # push box setting
        self.button_view = QPushButton("view")
        self.button_view.clicked.connect(self.show_info)
        self.button_close = QPushButton("close")
        self.button_close.clicked.connect(self.close_info)
        self.button_save = QPushButton("save")
        self.button_save.clicked.connect(self.save_info)
        self.button_load = QPushButton("load")
        self.button_load.clicked.connect(self.load_info)

        # list box setting
        self.list_widget = QListWidget()

        # Group Box 1
        layout1 = QVBoxLayout()
        layout1.addWidget(self.label_id)
        layout1.addWidget(self.line_edit_id)
        layout1.addWidget(self.label_age)
        layout1.addWidget(self.line_edit_age)
        layout1.addWidget(self.label_gender)
        layout1.addWidget(self.line_edit_gender)
        layout1.addWidget(self.label_country)
        layout1.addWidget(self.line_edit_country)

        group_box1.setLayout(layout1)

        # Group Box 2
        self.info_label = QLabel()
        layout2 = QVBoxLayout()
        layout2.addWidget(self.info_label)
        layout2.addWidget(self.button_view)
        layout2.addWidget(self.button_close)
        layout2.setContentsMargins(10, 10, 10, 10)
        group_box2.setLayout(layout2)

        # Group Box 3
        layout3 = QVBoxLayout()
        layout3.addWidget(self.button_save)
        layout3.addWidget(self.button_load)
        layout3.addWidget(self.list_widget)
        layout3.setContentsMargins(10, 10, 10, 10)
        group_box3.setLayout(layout3)

        # all layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(group_box1)
        main_layout.addWidget(group_box2)
        main_layout.addWidget(group_box3)

        self.setLayout(main_layout)

    def show_info(self) :
        id_ = self.line_edit_id.text()
        age = self.line_edit_age.text()
        gender = self.line_edit_gender.text()
        country = self.line_edit_country.text()

        info_text = f"id : {id_} \nage : {age}\ngender : {gender}\ncountry : {country}"
        self.info_label.setText(info_text)

    def close_info(self):
        self.line_edit_id.clear()
        self.line_edit_age.clear()
        self.line_edit_gender.clear()
        self.line_edit_country.clear()
        self.info_label.clear()

    def save_info(self):
        id_ = self.line_edit_id.text()
        age_ = self.line_edit_age.text()
        gender_ = self.line_edit_gender.text()
        country_ = self.line_edit_country.text()

        data = [id_, age_, gender_, country_]
        try :
            with open("data.csv" , 'a' , newline="", encoding='utf-8') as f :
                writer = csv.writer(f)
                writer.writerow(data)
            QMessageBox.information(self, "save ok" , "info save ok")
        except Exception as e :
            QMessageBox.critical(self, "save no", f"{str(e)}")

    def load_info(self):
        try :
            with open("data.csv", 'r') as f:
                reader = csv.reader(f)
                for row in reader :
                    data_text = f"id {row[0]}, age : {row[1]}," \
                                f" gender : {row[2]}, country : {row[3]}"
                    self.list_widget.addItem(data_text)

        except Exception as e :
            QMessageBox.information(self, "load no", f"{str(e)}")

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()