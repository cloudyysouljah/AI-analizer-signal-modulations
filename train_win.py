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

		self.samples_label = QtWidgets.QLabel(f"Количество сэмплов в датасете: {samples}")

		self.max_samples_line = QtWidgets.QLineEdit()
		self.max_samples_line.setPlaceholderText("Введите максимальное количество сэмплов")
		self.max_samples_line.setValidator(QtGui.QIntValidator())

		self.epochs_line = QtWidgets.QLineEdit()
		self.epochs_line.setPlaceholderText("Введите количество эпох")
		self.epochs_line.setValidator(QtGui.QIntValidator())

		self.batch_line = QtWidgets.QLineEdit()
		self.batch_line.setPlaceholderText("Введите размер батча")
		self.batch_line.setValidator(QtGui.QIntValidator())

		self.terminal = QtWidgets.QTextEdit()
		self.terminal.setReadOnly(True)
		self.terminal.setObjectName("log")

		self.train_btn = QtWidgets.QPushButton("Начать обучение")

		self.state_train = QtWidgets.QProgressBar()

		self.layout.addWidget(self.samples_label)
		self.layout.addWidget(self.max_samples_line)
		self.layout.addWidget(self.epochs_line)
		self.layout.addWidget(self.batch_line)
		self.layout.addWidget(self.train_btn)
		self.layout.addWidget(self.state_train)
		self.layout.addWidget(self.terminal)

		self.setLayout(self.layout)

		self.train_btn.clicked.connect(self.start_train)

		self.log_signal.connect(self.log)
		self.progress_signal.connect(self.state_train.setMaximum)
		self.progress_update.connect(self.state_train.setValue)

	def log(self, text):
		self.terminal.append(text)

	def start_train(self):
		train_thread = threading.Thread(
			target=self.train,
			daemon=True,
		)
		train_thread.start()

	def train(self):
		sys.stdout = StreamRedirect(self.log_signal.emit)
		try:
			max_samples = self.max_samples_line.text() if self.max_samples_line.text() != "" else self.samples
			epochs = self.epochs_line.text()
			batch = self.batch_line.text()
			self.progress_signal.emit(int(epochs))
			args = SimpleNamespace(data = self.path, max_samples = int(max_samples), epochs = int(epochs), batch_size = int(batch), 
								snr_min = -20, dropout = 0.4, lr = 2e-3, save = "best_model_test.pt")
			rain.main(args)
		finally:
			sys.stdout = sys.__stdout__