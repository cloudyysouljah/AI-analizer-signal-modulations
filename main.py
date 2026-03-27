import os
import sys
import threading

import h5py
import numpy as np
import pyqtgraph as pqtg
from PyQt6 import QtCore, QtWidgets


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
			self.index_label.setText(f"Индекс сигнала: {self.dataset_index}")

class DatasetWindow(BaseWindow):
	dataset_signal = QtCore.pyqtSignal(list, bool)

	def __init__(self, parent=None, dataset=None, title=None):
		super().__init__(parent, title)

		self.dataset_path = dataset
		self.dataset_thread = None
		self.data_thread = None
		self.dataset = None
		self.dataset_download = False
		self.dataset_index = 0
		self.global_index = 0

		self.index_label = QtWidgets.QLabel("Индекс сигнала: 0")
		self.index_spin = QtWidgets.QSpinBox()
		self.prev_btn = QtWidgets.QPushButton("Предыдущий")
		self.next_btn = QtWidgets.QPushButton("Следующий")
		self.show_dataset_btn = QtWidgets.QPushButton("Показать")

		self.right_box.addWidget(
			self.index_label,
			alignment=QtCore.Qt.AlignmentFlag.AlignTop,
		)
		self.right_box.addWidget(self.show_dataset_btn)
		self.right_box.addWidget(self.prev_btn)
		self.right_box.addWidget(self.next_btn)

		self.show_dataset_btn.clicked.connect(self.start_parse_dataset)
		self.next_btn.clicked.connect(self.next_index)
		self.prev_btn.clicked.connect(self.prev_index)
		self.dataset_signal.connect(self.draw_plot)

	def start_parse_dataset(self):
		self.dataset_thread = threading.Thread(
			target=self.parse_dataset,
			daemon=True,
		)
		self.dataset_download = True
		self.dataset_thread.start()

	def next_index(self):
		self.dataset_index += 1

		if self.dataset_index >= len(self.dataset[0]):
			self.dataset_index = self.dataset_index - len(self.dataset[0])

		self.dataset_signal.emit(self.dataset[0][self.dataset_index], True)

	def prev_index(self):
		self.dataset_index -= 1

		if self.dataset_index < 0:
			self.dataset_index = len(self.dataset[0]) - 1

		self.dataset_signal.emit(self.dataset[0][self.dataset_index], True)

	def parse_dataset(self):
		with h5py.File(self.dataset_path, "r") as f:
			x_data = f["X"][-1000:]
			y_data = f["Y"][-1000:]
			z_data = f["Z"][-1000:]
		self.dataset = (x_data, y_data, z_data)
		self.dataset_index = 0
		self.global_index = 0
		self.dataset_signal.emit(self.dataset[0][0], True)
		self.dataset_download = False


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
			# self.data = (
			# 	np.sin(t + self.phase) + np.random.randn(100) * 0.3
			# ).tolist()
			self.phase += 0.2
			self.data_signal.emit(self.data, False)
			self.msleep(int(self.freq))


class MainWindow(BaseWindow):
	def __init__(self, title=None):
		super().__init__()

		self.setWindowTitle(title or "Program")

		self.data_thread = None

		self.freq_label = QtWidgets.QLabel("Частота обновления (мс)")
		self.freq_spin = QtWidgets.QSpinBox()
		self.freq_spin.setRange(50, 1000)
		self.freq_spin.setValue(1000)

		self.add_data_btn = QtWidgets.QPushButton("Загрузить данные")
		self.dataset_btn = QtWidgets.QPushButton("Загрузить датасет")

		self.right_box.addWidget(
			self.freq_label,
			alignment=QtCore.Qt.AlignmentFlag.AlignTop,
		)
		self.right_box.addWidget(
			self.freq_spin,
			alignment=QtCore.Qt.AlignmentFlag.AlignTop,
		)
		self.right_box.addWidget(
			self.add_data_btn,
			alignment=QtCore.Qt.AlignmentFlag.AlignTop,
		)
		self.right_box.addWidget(
			self.dataset_btn,
			alignment=QtCore.Qt.AlignmentFlag.AlignTop,
		)

		self.add_data_btn.clicked.connect(self.add_data)
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

	def add_data(self):
		self.radio_graph.clear()
		self.data_thread = DataThread()
		self.data_thread.start()
		self.data_thread.data_signal.connect(self.draw_plot)

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	window = MainWindow(title="Program")
	window.show()
	sys.exit(app.exec())