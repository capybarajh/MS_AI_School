from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog

def open_file_dialog() :
    file_dialog = QFileDialog()
    file_dialog.setWindowTitle("file open test")
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    file_dialog.setViewMode(QFileDialog.Detail)

    if file_dialog.exec() :
        selected_files = file_dialog.selectedFiles()
        print("selected files >> " , selected_files)


app = QApplication([])
main_window = QMainWindow()
button = QPushButton("file open", main_window)
button.clicked.connect(open_file_dialog)
main_window.show()
app.exec()