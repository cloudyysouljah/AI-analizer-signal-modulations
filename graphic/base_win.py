from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pqtg
import numpy as np
from constants import CLASSES as classes

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

		self.info_label = QtWidgets.QLabel("Индекс сигнала: - \nКласс: - \nSNR (Отношение сигнал/шум в дБ): -")

		self.right_box = QtWidgets.QVBoxLayout()

		self.right_box.addWidget(
			self.info_label,
			alignment = QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignCenter
		)

		self.layout.addLayout(self.left_box)
		self.layout.addLayout(self.right_box)
		self.setLayout(self.layout)

	def clear_info(self):
		self.info_label.setText("Индекс сигнала: - \nКласс: - \nSNR (Отношение сигнал/шум в дБ): -")

	def clear_plot(self):
		self.radio_graph.clear()
		self.spectre_graph.clear()
		self.iq_plot.clear()

		self.curve_i = self.radio_graph.plot(pen=pqtg.mkPen("c", width=1), name="I")
		self.curve_q = self.radio_graph.plot(pen=pqtg.mkPen("y", width=1), name="Q")
		self.curve_fft = self.spectre_graph.plot(pen=pqtg.mkPen("g", width=1))

		self.scatter = pqtg.ScatterPlotItem(
			size=2,
			pen=None,
			brush=pqtg.mkBrush(255, 100, 100, 120),
		)
		self.iq_plot.addItem(self.scatter)

	def update_info(self, class_ = None, conf = None, speed = None, snr = None, dataset_win = False):
		if dataset_win:
			index = getattr(self, "current_index", self.index_spin.value())
			snr_value = getattr(self, "current_snr", None)
			if snr_value is None:
				snr_value = self.dataset[2][index - self.range_arr.min()]
			self.info_label.setText(f"Индекс сигнала: {index}\n" 
									f"Класс: {classes[self.class_index]}\n"
									f"SNR (Отношение сигнал/шум в дБ): {snr_value} \n")
		else:
			if conf > 0.5:
				self.info_label.setText(f"Индекс сигнала: прием в реальном времени\n"
										f"Класс: {class_} Вероятность: {(conf):.0%} ✅\n"
										f"Время обработки нейросети: {speed:.2f} мс\n"
										f"SNR (Отношение сигнал/шум в дБ): -")
			else:
				self.info_label.setText(f"Индекс сигнала: прием в реальном времени\n"
										f"Класс: {class_} Вероятность: {(conf):.0%} ❌\n"
										f"Время обработки нейросети: {speed:.2f} мс\n"
										f"SNR (Отношение сигнал/шум в дБ): {snr:.2f}")

	def draw_plot(self, data):
		data = np.array(data)
		i_data = data[:, 0]
		q_data = data[:, 1]

		self.curve_i.setData(i_data.tolist())
		self.curve_q.setData(q_data.tolist())

		spectrum = np.abs(np.fft.fftshift(np.fft.fft(i_data + 1j * q_data)))
		self.curve_fft.setData(spectrum.tolist())

		self.scatter.setData(x=i_data.tolist(), y=q_data.tolist())

	def keyPressEvent(self, a0):
		if a0.key() == QtCore.Qt.Key.Key_F11 and not self.isFullScreen():
			self.showFullScreen()
		elif a0.key() == QtCore.Qt.Key.Key_F11 and self.isFullScreen():
			self.showNormal()
