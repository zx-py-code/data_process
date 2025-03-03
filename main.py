import sys
from PyQt5.QtWidgets import QApplication
from data_organizer_app import DataOrganizerApp
import ExcelManager
#
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 使用Fusion风格
    window = DataOrganizerApp()
    window.show()
    sys.exit(app.exec_())