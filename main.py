from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pqtg
import sys
import os
import threading
import numpy as np
import h5py
import time

class base_win(QtWidgets.QWidget):
	def __init__(self, parent = None, title = None):
		super().__init__(parent)

		if title == None:
			title = "Program"

		self.setWindowTitle(title)

		self.layout = QtWidgets.QHBoxLayout()

		self.radio_graph = pqtg.PlotWidget(title = "Сигнал")
		self.spectre_graph = pqtg.PlotWidget(title = "Спектр")

		self.curve_I   = self.radio_graph.plot(pen=pqtg.mkPen('c', width=1), name='I')
		self.curve_Q   = self.radio_graph.plot(pen=pqtg.mkPen('y', width=1), name='Q')
		self.curve_fft = self.spectre_graph.plot(pen=pqtg.mkPen('g', width=1))

		self.left_box = QtWidgets.QVBoxLayout()
		self.left_box.addWidget(self.radio_graph)

		self.center_box = QtWidgets.QVBoxLayout()
		self.center_box.addWidget(self.spectre_graph)

		self.right_box = QtWidgets.QVBoxLayout()

		self.layout.addLayout(self.left_box)
		self.layout.addLayout(self.center_box)
		self.layout.addLayout(self.right_box)
		self.setLayout(self.layout)

	def draw_plot(self, data):
		data = np.array(data)
		I = data[:, 0]
		Q = data[:, 1]
		self.curve_I.setData(I[:1024].tolist())
		self.curve_Q.setData(Q[:1024].tolist())

		spectrum = np.abs(np.fft.fftshift(np.fft.fft(I + 1j * Q)))
		self.curve_fft.setData(spectrum.tolist())

class dataset_win(base_win):
	dataset_signal = QtCore.pyqtSignal(list)
	def __init__(self, parent = None, dataset = None, title = None):
		super().__init__(parent)
		
		self.setWindowTitle(title)

		self.dataset_path = dataset


		self.index_label = QtWidgets.QLabel("Номер сигнала")
		self.index_spin = QtWidgets.QSpinBox()
		self.step_btn = QtWidgets.QPushButton("Предыдущий")
		self.next_btn = QtWidgets.QPushButton("Следующий")
		self.show_dataset_btn = QtWidgets.QPushButton("Показать")

		self.dataset_thread = None
		self.data_thread = None
		self.dataset = None
		self.dataset_download = False
		self.dataset_index = 0

		self.right_box.addWidget(self.show_dataset_btn)
		self.right_box.addWidget(self.step_btn)
		self.right_box.addWidget(self.next_btn)

		self.show_dataset_btn.clicked.connect(self.start_parse_dataset)
		self.next_btn.clicked.connect(self.next_idx)
		self.step_btn.clicked.connect(self.step_idx)
		self.dataset_signal.connect(self.draw_plot)

	def start_parse_dataset(self):
		self.dataset_thread = threading.Thread(target = self.parse_dataset, daemon = True)
		self.dataset_download = True
		self.dataset_thread.start()

	def next_idx(self):
		self.dataset_index += 1
		# подгружаем следующий батч когда дошли до конца текущего
		if self.dataset_index >= len(self.dataset[0]):
			chunk_start = self.global_index  # глобальный индекс в файле
			with h5py.File(self.dataset_path, 'r') as f:
				total = f['X'].shape[0]
				if chunk_start >= total:
					chunk_start = 0  # зацикливаем
				X = f['X'][chunk_start:chunk_start + 1000]
				Y = f['Y'][chunk_start:chunk_start + 1000]
				Z = f['Z'][chunk_start:chunk_start + 1000]
			self.dataset = (X, Y, Z)
			self.dataset_index = 0
			self.global_index = chunk_start + 1000

		self.dataset_signal.emit(self.dataset[0][self.dataset_index])

	def step_idx(self):
		self.dataset_index -= 1
		# подгружаем предыдущий батч когда вышли за начало
		if self.dataset_index < 0:
			chunk_end = self.global_index - len(self.dataset[0])  # где начался текущий батч
			chunk_start = max(0, chunk_end - 1000)
			with h5py.File(self.dataset_path, 'r') as f:
				X = f['X'][chunk_start:chunk_end]
				Y = f['Y'][chunk_start:chunk_end]
				Z = f['Z'][chunk_start:chunk_end]
			self.dataset = (X, Y, Z)
			self.dataset_index = len(self.dataset[0]) - 1
			self.global_index = chunk_start

		self.dataset_signal.emit(self.dataset[0][self.dataset_index])

	def parse_dataset(self):
		with h5py.File(self.dataset_path, 'r') as f:
			X = f['X'][:1000]
			Y = f['Y'][:1000]
			Z = f['Z'][:1000]
		self.dataset = (X, Y, Z)
		self.dataset_index = 0
		self.global_index = 1000  # следующий старт
		self.dataset_signal.emit(self.dataset[0][0])
		self.dataset_download = False

class data_thread(QtCore.QThread):
	data_signal = QtCore.pyqtSignal(list)
	def __init__(self):
		super(data_thread, self).__init__()
		self.data = None
		self.freq = 1000
		self.phase = 0.0
		self.running = True

	def run(self):
		t = np.linspace(0, 2 * np.pi, 100)
		while self.running:
			self.data = (np.sin(t + self.phase) + np.random.randn(100) * 0.3).tolist()
			self.phase += 0.2
			self.data_signal.emit(self.data)
			self.msleep(int(self.freq))

class main_win(base_win):
	def __init__(self, title = None):
		super(main_win, self).__init__()
		
		if title == None:
			title = "Program"
		self.setWindowTitle(title)
		self.add_data_btn = QtWidgets.QPushButton("Загрузить данные")
		self.dataset_btn = QtWidgets.QPushButton("Загрузить датасет")
		self.freq_spin = QtWidgets.QSpinBox()
		self.freq_spin.setRange(50, 1000)
		self.freq_spin.setValue(1000)
		self.freq_label = QtWidgets.QLabel("Частота обновления (мс)")
		self.data_thread = None

		self.right_box.addWidget(self.freq_label, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
		self.right_box.addWidget(self.freq_spin, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
		self.right_box.addWidget(self.add_data_btn, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
		self.right_box.addWidget(self.dataset_btn, alignment=QtCore.Qt.AlignmentFlag.AlignTop)

		self.add_data_btn.clicked.connect(self.add_data)
		self.freq_spin.valueChanged.connect(self.change_freq)
		self.dataset_btn.clicked.connect(self.create_dataset_win)

	def create_dataset_win(self):
		dataset_path =  QtWidgets.QFileDialog.getOpenFileName(self, caption = "Выберите файл датасета", 
																	directory = f"{os.path.expanduser('~')}", 
																	filter = "*.hdf5")
		if dataset_path == "":
			return
		self.dataset_win = dataset_win(parent = self, dataset = dataset_path[0], title = "Dataset")
		self.dataset_win.setWindowFlag(QtCore.Qt.WindowType.Window)
		self.dataset_win.show()

	def change_freq(self):
		if self.data_thread is not None:
			self.data_thread.freq = self.freq_spin.value()

	def add_data(self):
		self.radio_graph.clear()

		self.data_thread = data_thread()
		self.data_thread.start()
		self.data_thread.data_signal.connect(self.draw_plot)

if __name__ == '__main__':
	app = QtWidgets.QApplication(sys.argv)
	main_win = main_win(title = "Program")
	main_win.show()
	sys.exit(app.exec())