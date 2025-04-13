from PySide6.QtWidgets import QApplication, QPushButton

def handle_button_click() :
    print("button click ~~")

app = QApplication([])
button = QPushButton("Click")

button.clicked.connect(handle_button_click)

button.show()
app.exec()