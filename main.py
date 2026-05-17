import sys
from PyQt6 import QtWidgets

from graphic.main_win import MainWindow

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	with open("graphic/style.css", "r") as f:
		app.setStyleSheet(f.read())
	window = MainWindow(title="Program")
	window.show()
	sys.exit(app.exec())