from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtGui import QFontMetrics
import train
import threading

class TrainWindow(QtWidgets.QDialog):
	log_signal = QtCore.pyqtSignal(str)
	progress_signal = QtCore.pyqtSignal(int)
	progress_update = QtCore.pyqtSignal(int)
	training_signal = QtCore.pyqtSignal(bool)
	def __init__(self, parent=None, title=None, samples = None, path = None):
		super().__init__(parent)

		self.setWindowTitle(title or "Program")

		self.path = path
		self.samples = samples

		self.layout = QtWidgets.QVBoxLayout()

		self.model_grid = QtWidgets.QGridLayout()

		self.samples_label = QtWidgets.QLabel(f"Количество сэмплов в датасете: {samples}")

		self.max_samples_label = QtWidgets.QLabel("Максимальное количество сэмплов")
		self.max_samples_line = QtWidgets.QLineEdit()
		self.max_samples_line.setPlaceholderText("Введите максимальное количество сэмплов")
		self.max_samples_line.setValidator(QtGui.QIntValidator())

		self.epochs_label = QtWidgets.QLabel("Количество эпох")
		self.epochs_line = QtWidgets.QLineEdit()
		self.epochs_line.setPlaceholderText("Введите количество эпох")
		self.epochs_line.setValidator(QtGui.QIntValidator())

		self.batch_label = QtWidgets.QLabel("Размер батча")
		self.batch_line = QtWidgets.QLineEdit()
		self.batch_line.setPlaceholderText("Введите размер батча")
		self.batch_line.setValidator(QtGui.QIntValidator())

		self.learn_label = QtWidgets.QLabel("Learning rate")
		self.learn_line = QtWidgets.QLineEdit()
		self.learn_line.setPlaceholderText("Введите learning rate")

		self.snr_label = QtWidgets.QLabel("Минимальное отношение сигнал/шум дБ")
		self.snr_line = QtWidgets.QLineEdit()
		self.snr_line.setPlaceholderText("Введите минимальное отношение сигнал/шум дБ")

		self.patience_label = QtWidgets.QLabel("Кол-во эпох без улучшения")
		self.patience_line = QtWidgets.QLineEdit()
		self.patience_line.setPlaceholderText("Введите кол-во эпох без улучшения")
		self.patience_line.setValidator(QtGui.QIntValidator())

		self.terminal = QtWidgets.QTextEdit()
		self.terminal.setReadOnly(True)
		self.terminal.setObjectName("log")

		self.train_btn = QtWidgets.QPushButton("Начать обучение")

		self.train_state = False
		self.train_thread = None

		self.state_train = QtWidgets.QProgressBar()
		self.state_train.setTextVisible(False)

		self.model_grid.addWidget(self.max_samples_label, 0, 0)
		self.model_grid.addWidget(self.max_samples_line, 0, 1)
		self.model_grid.addWidget(self.epochs_label, 1, 0)
		self.model_grid.addWidget(self.epochs_line, 1, 1)
		self.model_grid.addWidget(self.batch_label, 2, 0)
		self.model_grid.addWidget(self.batch_line, 2, 1)
		self.model_grid.addWidget(self.learn_label, 3, 0)
		self.model_grid.addWidget(self.learn_line, 3, 1)
		self.model_grid.addWidget(self.snr_label, 4, 0)
		self.model_grid.addWidget(self.snr_line, 4, 1)
		self.model_grid.addWidget(self.patience_label, 5, 0)
		self.model_grid.addWidget(self.patience_line, 5, 1)

		self.layout.addWidget(self.samples_label)
		self.layout.addLayout(self.model_grid)
		self.layout.addWidget(self.train_btn)
		self.layout.addWidget(self.state_train)
		self.layout.addWidget(self.terminal)

		self.resize_to_placeholders()

		self.setLayout(self.layout)

		self.train_btn.clicked.connect(self.train_thr)

		self.log_signal.connect(self.log)
		self.progress_signal.connect(self.state_train.setMaximum)
		self.progress_update.connect(self.state_train.setValue)

	def resize_to_placeholders(self):
		fm = QFontMetrics(self.font())

		max_width = 0

		for line in [
			self.max_samples_line,
			self.epochs_line,
			self.batch_line,
			self.learn_line,
			self.snr_line,
			self.patience_line,
		]:
			text = line.placeholderText()
			w = fm.horizontalAdvance(text)
			max_width = max(max_width, w)

		total_width = max_width + 300

		self.setMinimumWidth(total_width)
		self.resize(total_width, self.height())

	def log(self, text):
		self.terminal.append(text)

	def train_thr(self):
		self.train_state = not self.train_state
		if self.train_state:
			self.train_thread = threading.Thread(
				target=self.train,
				daemon=True,
			)
			self.train_thread.start()
			self.train_btn.setText("Прервать обучение")
		else:
			self.training_signal.emit(False)
			self.train_btn.setText("Начать обучение")

	def train(self):
		try:
			max_samples = int(self.max_samples_line.text()) if self.max_samples_line.text() != "" else self.samples
			epochs = int(self.epochs_line.text())
			batch = int(self.batch_line.text())
			lr_rate = float(self.learn_line.text())
			snr_min = float(self.snr_line.text())
			patience = int(self.patience_line.text())
			self.progress_signal.emit(int(epochs))
			model_train = train.Train(parent = self, path = self.path,
									save_path = "best_model_test.pt", batch_size = batch,
									lr = lr_rate, epochs = epochs,
									patience = patience, snr_min = snr_min, max_samples = max_samples)
			self.training_signal.connect(model_train.state_train)
			self.training_signal.emit(True)
			model_train.run()
		except Exception as e:
			self.log_signal.emit(str(e))