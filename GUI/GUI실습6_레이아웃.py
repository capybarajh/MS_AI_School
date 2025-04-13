import sys
from PySide6.QtWidgets import QApplication, QWidget,\
    QVBoxLayout, QBoxLayout, QGroupBox, QPushButton, QLineEdit, QLabel, QHBoxLayout

class MainWindow(QWidget) :
    def __init__(self):
        super().__init__()

        self.setWindowTitle("UI")

        # GroupBox 1
        group_box1 = QGroupBox("Group box 1")
        label1 = QLabel("name : ")
        line_edit1 = QLineEdit()
        button1 = QPushButton("save")

        layout1 = QVBoxLayout()
        layout1.addWidget(label1)
        layout1.addWidget(line_edit1)
        layout1.addWidget(button1)
        group_box1.setLayout(layout1)

        # GroupBox 2
        group_box2 = QGroupBox("Group box 1")
        label2 = QLabel("age : ")
        line_edit2 = QLineEdit()
        button2 = QPushButton("clear")

        layout2 = QHBoxLayout()
        layout2.addWidget(label2)
        layout2.addWidget(line_edit2)
        layout2.addWidget(button2)
        group_box2.setLayout(layout2)

        # all layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(group_box1)
        main_layout.addWidget(group_box2)

        self.setLayout(main_layout)

if __name__ == "__main__" :
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()