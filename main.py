import os
import sys
import threading

import h5py
import numpy as np
import pyqtgraph as pqtg
from PyQt6 import QtCore, QtWidgets

classes = ["OOK", "ASK4", "ASK8",
		   "BPSK", "QPSK", "PSK8",
		   "PSK16", "PSK32", "APSK16",
		   "APSK32", "APSK64", "APSK128",
		   "QAM16", "QAM32", "QAM64", "QAM128",
		   "QAM256", "AM_SSB_WC", "AM_SSB_SC",
		   "AM_DSB_WC", "AM_DSB_SC",
		   "FM", "GMSK", "OQPS"]

class BaseWindow(QtWidgets.QWidget):
	def __init__(self, parent=None, title=None):
		super().__init__(parent)

		self.setWindowTitle(title or "Program")

		self.layout = QtWidgets.QHBoxLayout()

		self.radio_graph = pqtg.PlotWidget(title="Сигнал")
		self.spectre_graph = pqtg.PlotWidget(title="Спектр")

		self.curve_i = self.radio_graph.plot(pen=pqtg.mkPen("c", width=1), name="I")
		self.curve_q = self.radio_graph.plot(pen=pqtg.mkPen("y", width=1), name="Q")
		self.curve_fft = self.spectre_graph.plot(pen=pqtg.mkPen("g", width=1))

		self.iq_plot = pqtg.PlotWidget(title="IQ-созвездие")
		self.iq_plot.setAspectLocked(True)
		self.scatter = pqtg.ScatterPlotItem(
			size=2,
			pen=None,
			brush=pqtg.mkBrush(255, 100, 100, 120),
		)
		self.iq_plot.addItem(self.scatter)

		self.left_box = QtWidgets.QVBoxLayout()
		self.left_box.addWidget(self.radio_graph)
		self.left_box.addWidget(self.spectre_graph)
		self.left_box.addWidget(self.iq_plot)

		self.right_box = QtWidgets.QVBoxLayout()

		self.layout.addLayout(self.left_box)
		self.layout.addLayout(self.right_box)
		self.setLayout(self.layout)

	def draw_plot(self, data, datawin):
		data = np.array(data)
		i_data = data[:, 0]
		q_data = data[:, 1]

		self.curve_i.setData(i_data[:1024].tolist())
		self.curve_q.setData(q_data[:1024].tolist())

		spectrum = np.abs(np.fft.fftshift(np.fft.fft(i_data + 1j * q_data)))
		self.curve_fft.setData(spectrum.tolist())

		self.scatter.setData(x=i_data.tolist(), y=q_data.tolist())
		if datawin:
			self.info_label.setText(f"Индекс сигнала: {self.index_spin.value()}\n" 
									f"Класс: {classes[self.class_index]}\n"
									f"SNR: {self.dataset[2][self.index_spin.value() - self.range_arr.min()]} дБ\n")

class DatasetWindow(BaseWindow):
	dataset_signal = QtCore.pyqtSignal(list, bool)

	def __init__(self, parent=None, dataset=None, title=None):
		super().__init__(parent, title)

		self.dataset_path = dataset
		self.dataset_thread = None
		self.data_thread = None
		self.dataset = None
		self.dataset_index = 0

		self.dataset_parsing = False

		self.info_label = QtWidgets.QLabel("Индекс сигнала: - \nКласс: - \nSNR: -")
		self.index_spin = QtWidgets.QSpinBox()
		self.index_spin.setRange(0, 2555904)
		self.train_btn = QtWidgets.QPushButton("Начать обучение")
		self.prev_btn = QtWidgets.QPushButton("Предыдущий")
		self.next_btn = QtWidgets.QPushButton("Следующий")
		self.show_dataset_btn = QtWidgets.QPushButton("Показать")

		self.right_box.addWidget(
			self.index_spin,
			# alignment=QtCore.Qt.AlignmentFlag.AlignTop
			)
		self.right_box.addWidget(
			self.info_label,
			# alignment=QtCore.Qt.AlignmentFlag.AlignTop
		)
		self.right_box.addWidget(
			self.train_btn,
			# alignment=QtCore.Qt.AlignmentFlag.AlignTop
			)
		self.right_box.addWidget(
			self.show_dataset_btn,
			# alignment=QtCore.Qt.AlignmentFlag.AlignTop
			)
		self.right_box.addWidget(
			self.prev_btn,
			# alignment=QtCore.Qt.AlignmentFlag.AlignTop
			)
		self.right_box.addWidget(
			self.next_btn,
			# alignment=QtCore.Qt.AlignmentFlag.AlignTop
			)

		self.show_dataset_btn.clicked.connect(self.start_parse_dataset)
		self.next_btn.clicked.connect(self.next_index)
		self.prev_btn.clicked.connect(self.prev_index)
		self.dataset_signal.connect(self.draw_plot)
		self.index_spin.valueChanged.connect(self.index_spin_changed)

	def start_parse_dataset(self):
		self.dataset_parsing = True
		self.dataset_thread = threading.Thread(
			target=self.parse_dataset,
			daemon=True,
		)
		self.dataset_thread.start()

	def update_dataset(self, index):
		if index not in self.range_arr:
			with h5py.File(self.dataset_path, "r") as f:
				x_data = f["X"][index - 2500:index + 2500]
				y_data = f["Y"][index - 2500:index + 2500]
				z_data = f["Z"][index - 2500:index + 2500]
			self.dataset = [x_data, y_data, z_data]
			self.range_arr = np.arange(index - 2500, index + 2500)
			local = self.index_spin.value() - self.range_arr.min()
			for i in range(len(y_data[local])):
				if y_data[local][i] == 1:
					self.class_index = i

	def index_spin_changed(self):
		self.update_dataset(self.index_spin.value())
		local = self.index_spin.value() - self.range_arr.min()
		self.dataset_signal.emit(self.dataset[0][local], True)

	def next_index(self):
		self.index_spin.setValue(self.index_spin.value() + 1)
		local = self.index_spin.value() - self.range_arr.min()
		self.dataset_signal.emit(self.dataset[0][local], True)

	def prev_index(self):
		self.index_spin.setValue(self.index_spin.value() - 1)
		local = self.index_spin.value() - self.range_arr.min()
		self.dataset_signal.emit(self.dataset[0][local], True)

	def parse_dataset(self):
		while self.dataset_parsing:
			with h5py.File(self.dataset_path, "r") as f:
				x_data = f["X"][:5000]
				y_data = f["Y"][:5000]
				z_data = f["Z"][:5000]
			self.range_arr = np.arange(0, 5000)
			self.dataset = [x_data, y_data, z_data]
			for i in range(len(y_data[self.dataset_index])):
				if y_data[self.dataset_index][i] == 1:
					self.class_index = i
			self.dataset_parsing = False
		self.dataset_signal.emit(self.dataset[0][self.dataset_index], True)

class DataThread(QtCore.QThread):
	data_signal = QtCore.pyqtSignal(list, bool)

	def __init__(self):
		super().__init__()
		self.data = None
		self.freq = 1000
		self.phase = 0.0
		self.running = True

	def run(self):
		t = np.linspace(0, 2 * np.pi, 1024)
		while self.running:
			i = np.sin(t + self.phase) + np.random.randn(1024) * 0.3
			q = np.sin(t + self.phase)
			self.data = np.column_stack([i, q]).tolist()  # shape (100, 2)
			self.phase += 0.2
			self.data_signal.emit(self.data, False)
			self.msleep(int(self.freq))

class MainWindow(BaseWindow):
	def __init__(self, title=None):
		super().__init__()

		self.setWindowTitle(title or "Program")

		self.data_thread = None

		self.data_status = False

		self.freq_box = QtWidgets.QHBoxLayout()
		self.freq_label = QtWidgets.QLabel("Частота обновления (мс)")
		self.freq_spin = QtWidgets.QSpinBox()
		self.freq_spin.setRange(50, 1000)
		self.freq_spin.setValue(1000)
		self.freq_box.addWidget(self.freq_label)
		self.freq_box.addWidget(self.freq_spin)

		self.data_box = QtWidgets.QHBoxLayout()
		self.data_receive_btn = QtWidgets.QPushButton("Приём сигнала")
		self.data_status_label = QtWidgets.QLabel()
		self.data_status_label.setFixedSize(30, 30)
		self.data_status_label.setObjectName("data-receive-btn")
		self.data_box.addWidget(self.data_receive_btn)
		self.data_box.addWidget(self.data_status_label)
		
		self.dataset_btn = QtWidgets.QPushButton("Загрузить датасет")

		self.right_box.addLayout(self.freq_box)

		self.right_box.addLayout(
			self.data_box,
			# alignment=QtCore.Qt.AlignmentFlag.AlignTop,
		)
		self.right_box.addWidget(
			self.dataset_btn,
			# alignment=QtCore.Qt.AlignmentFlag.AlignTop,
		)

		self.data_receive_btn.clicked.connect(self.data_receive)
		self.freq_spin.valueChanged.connect(self.change_freq)
		self.dataset_btn.clicked.connect(self.open_dataset_window)

	def open_dataset_window(self):
		dataset_path, _ = QtWidgets.QFileDialog.getOpenFileName(
			self,
			caption="Выберите файл датасета HDF5",
			directory=os.path.expanduser("~"),
			filter="*.hdf5",
		)
		if not dataset_path:
			return
		self.dataset_win = DatasetWindow(
			parent=self,
			dataset=dataset_path,
			title="Dataset",
		)
		self.dataset_win.setWindowFlag(QtCore.Qt.WindowType.Window)
		self.dataset_win.show()

	def change_freq(self):
		if self.data_thread is not None:
			self.data_thread.freq = self.freq_spin.value()

	def data_receive(self):
		if not self.data_status:
			self.data_status_label.setObjectName("data-receive-btn-active")
			self.data_status_label.style().unpolish(self.data_status_label)
			self.data_status_label.style().polish(self.data_status_label)
			self.data_status_label.update()
			self.data_thread = DataThread()
			self.data_thread.start()
			self.data_thread.data_signal.connect(self.draw_plot)
			self.data_status = True
		else:
			self.data_status_label.setObjectName("data-receive-btn")
			self.data_status_label.style().unpolish(self.data_status_label)
			self.data_status_label.style().polish(self.data_status_label)
			self.data_status_label.update()
			self.data_thread.running = False
			self.data_status = False

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	with open("style.css", "r") as f:
		app.setStyleSheet(f.read())
	window = MainWindow(title="Program")
	window.show()
	sys.exit(app.exec())