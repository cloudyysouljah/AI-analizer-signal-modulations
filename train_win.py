from PyQt6 import QtCore, QtWidgets, QtGui
import rain

class TrainWindow(QtWidgets.QDialog):
	def __init__(self, parent=None, title=None, samples = None, path = None):
		super().__init__(parent)

		self.setWindowTitle(title or "Program")

		self.layout = QtWidgets.QVBoxLayout()

		self.samples_label = QtWidgets.QLabel(f"Количество сэмплов в датасете: {samples}")

		self.max_samples_line = QtWidgets.QLineEdit()
		self.max_samples_line.setPlaceholderText("Введите максимальное количество сэмплов")
		self.max_samples_line.setValidator(QtGui.QIntValidator())

		self.epochs_line = QtWidgets.QLineEdit()
		self.epochs_line.setPlaceholderText("Введите количество эпох")
		self.epochs_line.setValidator(QtGui.QIntValidator())

		self.train_btn = QtWidgets.QPushButton("Начать обучение")

		self.state_train = QtWidgets.QProgressBar()

		self.layout.addWidget(self.samples_label)
		self.layout.addWidget(self.max_samples_line)
		self.layout.addWidget(self.epochs_line)
		self.layout.addWidget(self.train_btn)
		self.layout.addWidget(self.state_train)

		self.setLayout(self.layout)

		# self.train_btn.clicked.connect()