from PyQt6 import QtCore, QtWidgets, QtGui
import rain
import threading
import sys
from types import SimpleNamespace

class StreamRedirect:
    def __init__(self, callback):
        self.callback = callback
    
    def write(self, text):
        if text.strip():
            self.callback(text)
    
    def flush(self):
        pass

class Toggle(QtWidgets.QPushButton):
	def __init__(self, text):
		super().__init__(text)
		self.setCheckable(True)

		self.clicked.connect(self.toggle)

	def toggle(self):
		if self.isChecked():
			self.setText("Прервать обучение")
		else:
			self.setText("Начать обучение")

class TrainWindow(QtWidgets.QDialog):
	log_signal = QtCore.pyqtSignal(str)
	progress_signal = QtCore.pyqtSignal(int)
	progress_update = QtCore.pyqtSignal(int)
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
		# self.learn_line.setValidator(QtGui.QIntValidator())

		self.terminal = QtWidgets.QTextEdit()
		self.terminal.setReadOnly(True)
		self.terminal.setObjectName("log")

		self.train_btn = Toggle("Начать обучение")

		self._stop_event = threading.Event()
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

		self.layout.addLayout(self.model_grid)
		self.layout.addWidget(self.train_btn)
		self.layout.addWidget(self.state_train)
		self.layout.addWidget(self.terminal)

		self.setLayout(self.layout)

		self.train_btn.clicked.connect(self.train_thr)

		self.log_signal.connect(self.log)
		self.progress_signal.connect(self.state_train.setMaximum)
		self.progress_update.connect(self.state_train.setValue)

	def log(self, text):
		self.terminal.append(text)

	def train_thr(self):
		if self.train_btn.isChecked():
			self._stop_event.clear()
			self.train_thread = threading.Thread(
				target=self.train,
				args = (self._stop_event,),
				daemon=True,
			)
			self.train_thread.start()
			self.train_btn.setText("Прервать обучение")
		else:
			self._stop_event.set()  # сигнал потоку остановиться
			self.train_btn.setText("Начать обучение")

	def train(self):
		sys.stdout = StreamRedirect(self.log_signal.emit)
		try:
			max_samples = self.max_samples_line.text() if self.max_samples_line.text() != "" else self.samples
			epochs = self.epochs_line.text()
			batch = self.batch_line.text()
			self.progress_signal.emit(int(epochs))
			args = SimpleNamespace(data = self.path, max_samples = int(max_samples), epochs = int(epochs), batch_size = int(batch), 
								snr_min = 0, dropout = 0.5, lr = 5e-4, save = "best_model_test.pt")
			rain.main(args, parent = self)
		finally:
			sys.stdout = sys.__stdout__