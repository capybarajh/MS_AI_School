# 폴더 열기를 통해서 폴더를 선택하면 파일들을 Tree 형태로 해서 보여주는 탐색기 만들기
import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, \
    QPushButton, QVBoxLayout, QFileDialog, QTreeWidget, QTreeWidgetItem, QWidget

class FileExplorer(QMainWindow) :
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Explorer")
        self.resize(500,400)
        # button setting
        self.folder_button = QPushButton("folder open")
        self.folder_button.clicked.connect(self.open_folder_dialog)
        # button clicked connect fix code

        # tree setting
        self.tree_wiget = QTreeWidget()
        self.tree_wiget.setHeaderLabels(['file'])

        # main_layout setting
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.folder_button)
        main_layout.addWidget(self.tree_wiget)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.folder_path = ""

    def open_folder_dialog(self):
        """사용자가 폴더 대화 상자를 열고 선택한 폴더 경로를 설정하고 파일 표시하는 함수 """
        folder_dialog = QFileDialog(self)
        folder_dialog.setFileMode(QFileDialog.Directory) # 대화상자 폴더 선택 모드

        # 파일 대화 상자가 디렉토리만 표시 하도록 설정
        folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)

        # 호출 하면 사용자가 선택할때 마다 set_folder_path() 함수 실행됩니다.
        # 사용자가 선택한 폴더의 경로 설정하는 역할
        folder_dialog.directoryEntered.connect(self.set_folder_path)

        # 파일 표시 역활 하는 함수 호출
        folder_dialog.accepted.connect(self.display_files)

        folder_dialog.exec_()

    def set_folder_path(self, folder_path):
        self.folder_path = folder_path

    def display_files(self):
        if self.folder_path :
            self.tree_wiget.clear()

            root_item = QTreeWidgetItem(self.tree_wiget, [self.folder_path])
            self.tree_wiget.addTopLevelItem(root_item)

            for dir_path , _ , file_names in os.walk(self.folder_path) :
                dir_item = QTreeWidgetItem(root_item, [os.path.basename(dir_path)])
                root_item.addChild(dir_item)

                for file_name in file_names :
                    file_item = QTreeWidgetItem(dir_item, [file_name])
                    dir_item.addChild(file_item)

                root_item.setExpanded(True)


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    windows = FileExplorer()
    windows.show()
    app.exec()