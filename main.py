from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pqtg
import sys
import numpy as np
import time

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

class main_win(QtWidgets.QWidget):
	def __init__(self):
		super(main_win, self).__init__()
		self.setWindowTitle("Program")

		self.layout = QtWidgets.QHBoxLayout()

		self.radio_graph = pqtg.PlotWidget()
		self.radio_label = QtWidgets.QLabel("Сигнал")
		self.spectre_graph = pqtg.PlotWidget()
		self.spectre_label = QtWidgets.QLabel("Спектр")
		self.add_data_btn = QtWidgets.QPushButton("Загрузить данные")
		self.freq_spin = QtWidgets.QSpinBox()
		self.freq_spin.setRange(50, 1000)
		self.freq_spin.setValue(1000)
		self.freq_label = QtWidgets.QLabel("Частота обновления (мс)")
		self.data_thread = None
		
		self.left_box = QtWidgets.QVBoxLayout()
		self.left_box.addWidget(self.radio_graph)
		self.left_box.addWidget(self.radio_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

		self.center_box = QtWidgets.QVBoxLayout()
		self.center_box.addWidget(self.spectre_graph)
		self.center_box.addWidget(self.spectre_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

		self.right_box = QtWidgets.QVBoxLayout()
		self.right_box.addWidget(self.freq_label, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
		self.right_box.addWidget(self.freq_spin, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
		self.right_box.addWidget(self.add_data_btn, alignment=QtCore.Qt.AlignmentFlag.AlignTop)

		self.layout.addLayout(self.left_box)
		self.layout.addLayout(self.center_box)
		self.layout.addLayout(self.right_box)
		self.setLayout(self.layout)

		self.add_data_btn.clicked.connect(self.add_data)
		self.freq_spin.valueChanged.connect(self.change_freq)

	def change_freq(self):
		if self.data_thread is not None:
			self.data_thread.freq = self.freq_spin.value()

	def add_data(self):
		self.radio_graph.clear()

		self.data_thread = data_thread()
		self.data_thread.start()
		self.data_thread.data_signal.connect(self.draw_plot)

	def draw_plot(self, data):
		self.radio_graph.clear()
		self.radio_graph.plot(data)

		self.spectre_graph.clear()
		self.spectre_graph.plot(np.abs(np.fft.fft(data)))

if __name__ == '__main__':
	app = QtWidgets.QApplication(sys.argv)
	main_win = main_win()
	main_win.show()
	sys.exit(app.exec())