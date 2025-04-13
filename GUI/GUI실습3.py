import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTableView
from PySide6.QtGui import QStandardItemModel
from PySide6.QtGui import QStandardItem

class MainWindow(QMainWindow) :
    def __init__(self):
        super().__init__()

        self.setWindowTitle("테이블 뷰 예제")

        table_view = QTableView(self)
        self.setCentralWidget(table_view)

        # 테이블 모델 생성
        model = QStandardItemModel(4,3, self)
        model.setHorizontalHeaderLabels(["이름", "나이", "성별"])

        # 데이터 추가
        model.setItem(0,0, QStandardItem("Alice"))
        model.setItem(0,1, QStandardItem("25"))
        model.setItem(0,2, QStandardItem("여성"))

        model.setItem(1,0, QStandardItem("Bob"))
        model.setItem(1,1, QStandardItem("29"))
        model.setItem(1,2, QStandardItem("남성"))

        model.setItem(2,0, QStandardItem("Daisy"))
        model.setItem(2,1, QStandardItem("25"))
        model.setItem(2,2, QStandardItem("여성"))

        model.setItem(3,0, QStandardItem("Join"))
        model.setItem(3,1, QStandardItem("27"))
        model.setItem(3,2, QStandardItem("남성"))

        table_view.setModel(model)
        table_view.resizeColumnsToContents()
        table_view.setEditTriggers(QTableView.NoEditTriggers) # 편집 불가능하게

if __name__ == "__main__" :
    app=QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())