import os
import sys
import threading

import h5py
import onnxruntime as ort
import numpy as np
import rain, train_win
import torch
import time
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

	def update_info(self, class_ = None, conf = None, speed = None, dataset_win = False):
		if dataset_win:
			self.info_label.setText(f"Индекс сигнала: {self.index_spin.value()}\n" 
									f"Класс: {classes[self.class_index]}\n"
									f"SNR (Отношение сигнал/шум в дБ): {self.dataset[2][self.index_spin.value() - self.range_arr.min()]} \n")
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
										f"SNR (Отношение сигнал/шум в дБ): -")

	def draw_plot(self, data):
		data = np.array(data)
		i_data = data[:, 0]
		q_data = data[:, 1]

		self.curve_i.setData(i_data[:1024].tolist())
		self.curve_q.setData(q_data[:1024].tolist())

		spectrum = np.abs(np.fft.fftshift(np.fft.fft(i_data + 1j * q_data)))
		self.curve_fft.setData(spectrum.tolist())

		self.scatter.setData(x=i_data.tolist(), y=q_data.tolist())

class DatasetWindow(BaseWindow):
	dataset_signal = QtCore.pyqtSignal(list)
	info_signal = QtCore.pyqtSignal(bool)

	def __init__(self, parent=None, dataset=None, title=None):
		super().__init__(parent, title)

		self.dataset_path = dataset
		self.dataset_thread = None
		self.data_thread = None
		self.dataset = None
		self.dataset_index = 0

		self.dataset_parsing = False

		self.index_spin = QtWidgets.QSpinBox()
		self.index_spin.setRange(0, self.get_dataset_size(self.dataset_path) - 1)
		self.index_spin.setEnabled(False)
		self.train_btn = QtWidgets.QPushButton("Начать обучение")
		self.prev_btn = QtWidgets.QPushButton("Предыдущий")
		self.next_btn = QtWidgets.QPushButton("Следующий")
		self.show_dataset_btn = QtWidgets.QPushButton("Показать")

		self.right_box.addWidget(
			self.index_spin,
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

		self.train_btn.clicked.connect(self.start_train)
		self.show_dataset_btn.clicked.connect(self.start_parse_dataset)
		self.next_btn.clicked.connect(self.next_index)
		self.prev_btn.clicked.connect(self.prev_index)
		self.dataset_signal.connect(self.draw_plot)
		self.info_signal.connect(lambda info: self.update_info(dataset_win = info))
		self.index_spin.valueChanged.connect(self.index_spin_changed)

	def start_train(self):
		train_window = train_win.TrainWindow(parent = self, title = "Обучение", samples = self.get_dataset_size(self.dataset_path), path = self.dataset_path)
		train_window.show()

	def start_parse_dataset(self):
		self.dataset_parsing = True
		self.dataset_thread = threading.Thread(
			target=self.parse_dataset,
			daemon=True,
		)
		self.dataset_thread.start()
		self.index_spin.setEnabled(True)

	# def update_dataset(self, index):
	# 	if index not in self.range_arr:
	# 		with h5py.File(self.dataset_path, "r") as f:
	# 			x_data = f["X"][index - 2500:index + 2500]
	# 			y_data = f["Y"][index - 2500:index + 2500]
	# 			z_data = f["Z"][index - 2500:index + 2500]
	# 		self.dataset = [x_data, y_data, z_data]
	# 		self.range_arr = np.arange(index - 2500, index + 2500)
	# 		local = self.index_spin.value() - self.range_arr.min()
	# 		for i in range(len(y_data[local])):
	# 			if y_data[local][i] == 1:
	# 				self.class_index = i

	def update_dataset(self, index):
		if index in self.range_arr:
			return
		half = 2500
		start = max(0, index - half)
		end = min(self.get_dataset_size(self.dataset_path), index + half)
		with h5py.File(self.dataset_path, "r") as f:
			x_data = f["X"][start:end]
			y_data = f["Y"][start:end]
			z_data = f["Z"][start:end]
		self.dataset = [x_data, y_data, z_data]
		self.range_arr = np.arange(start, end)
		local = self.index_spin.value() - self.range_arr.min()
		for i in range(len(y_data[local])):
			if y_data[local][i] == 1:
				self.class_index = i

	def index_spin_changed(self):
		self.update_dataset(self.index_spin.value())
		local = self.index_spin.value() - self.range_arr.min()
		self.dataset_signal.emit(self.dataset[0][local])
		self.info_signal.emit(True)

	def next_index(self):
		self.index_spin.setValue(self.index_spin.value() + 1)

	def prev_index(self):
		self.index_spin.setValue(self.index_spin.value() - 1)

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
		self.dataset_signal.emit(self.dataset[0][self.dataset_index])
		self.info_signal.emit(True)

	def get_dataset_size(self, path):
		with h5py.File(path, "r") as f:
			len_dataset = f["X"].shape[0]
		return len_dataset

class DataThread(QtCore.QThread):
	data_signal = QtCore.pyqtSignal(list)
	info_signal = QtCore.pyqtSignal(str, float, float, bool)
	ai_signal = QtCore.pyqtSignal(bool)
	probs_signal = QtCore.pyqtSignal(np.ndarray, int)

	def __init__(self):
		super().__init__()
		self.data = None
		self.freq = 1000
		self.phase = 0.0
		self.running = True
		self.ai = False

		self.sess = ort.InferenceSession('models/best_model_ideal.onnx', providers=['CUDAExecutionProvider'])
		self.ai_signal.connect(self.ai_state)

	def run(self):
		# t = np.linspace(0, 2 * np.pi, 1024)
		while self.running:
			# i = np.sin(t + self.phase) + np.random.randn(1024) * 0.3
			# q = np.sin(t + self.phase)
			# iq = np.stack([i, q], axis=0)
			iq = self.generate_bpsk()
			# snr = self.estimate_snr(iq)
			self.data = iq.T.tolist()
			# self.data = np.column_stack([i, q]).tolist()  # shape (100, 2)
			if self.ai:
				pred_idx, confidence, speed_ai, probs = self.ai_proc(iq)
				self.info_signal.emit(classes[pred_idx], confidence, speed_ai, False)
				self.probs_signal.emit(probs, pred_idx)
			self.phase += 0.2
			self.data_signal.emit(self.data)
			self.msleep(int(self.freq))

	def generate_bpsk(self, n_samples=1024, sps=8, snr_db=None):
		n_symbols = n_samples // sps
		bits = np.random.randint(0, 2, n_symbols)
		symbols = (2 * bits - 1).astype(complex)

		# Прямоугольный фильтр (как в RadioML 2016)
		signal = np.repeat(symbols, sps)

		# Минимальный частотный сдвиг как в датасете
		t = np.arange(n_samples)
		freq_offset = np.random.uniform(-0.01, 0.01)  # почти 0
		signal = signal * np.exp(1j * 2 * np.pi * freq_offset * t)

		# Случайный SNR
		if snr_db is None:
			snr_db = np.random.choice(np.arange(-6, 30, 2))  # как в RadioML
		snr_linear = 10 ** (snr_db / 10)
		noise_std = 1 / np.sqrt(2 * snr_linear)
		noise = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) * noise_std
		signal = signal + noise
		# Нормализация по мощности
		signal = signal / (np.sqrt(np.mean(np.abs(signal) ** 2)) + 1e-8)

		return np.stack([signal.real, signal.imag], axis=0)

	def ai_proc(self, data):
		start = time.time()

		output = self.sess.run(None, {'input': torch.tensor(data, dtype=torch.float32).unsqueeze(0).numpy()})

		logits = output[0][0]  # сырые логиты
		#Применяем softmax вручную
		e = np.exp(logits - logits.max())  # вычитаем max для численной стабильности
		probs = e / e.sum()
		
		pred_idx = probs.argmax()
		confidence = probs[pred_idx]  # теперь от 0.0 до 1.0
		speed_ai = (time.time() - start) * 1000
		return pred_idx, confidence, speed_ai, probs

	def ai_state(self, state):
		self.ai = state

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

		self.conf_graph = pqtg.PlotWidget(title="Классификация")
		self.conf_graph.setLabel(axis='left', text='Классы')
		self.conf_graph.setLabel(axis='bottom', text='Вероятность классификации')
		self.conf_graph.getAxis('left').setTicks([list(enumerate(classes))])
		proc = [[(0, '0%'), (0.25, '25%'), (0.5, '50%'), (0.75, '75%'), (1, '100%')]]
		self.conf_graph.getAxis('bottom').setTicks(proc)

		self.conf_graph.setXRange(0, 1)
		self.conf_graph.setYRange(0, len(classes))

		self.bar_item = pqtg.BarGraphItem(
			x=np.zeros(len(classes)),
			x1=np.zeros(len(classes)),
			y=np.arange(len(classes)),
			height=0.6,
			width=np.zeros(len(classes)),
			brush=pqtg.mkBrush(100, 200, 255, 180),
		)
		self.conf_graph.addItem(self.bar_item)

		self.data_box = QtWidgets.QHBoxLayout()
		self.data_receive_btn = QtWidgets.QPushButton("Начать приём сигнала")
		self.data_status_label = QtWidgets.QLabel()
		self.data_status_label.setFixedSize(30, 30)
		self.data_status_label.setObjectName("data-receive-btn")
		self.data_box.addWidget(self.data_receive_btn)
		self.data_box.addWidget(self.data_status_label)

		self.dataset_btn = QtWidgets.QPushButton("Загрузить датасет")
		
		self.ai_chk = QtWidgets.QCheckBox("AI")
		self.ai_chk.setEnabled(False)

		self.right_box.addLayout(self.freq_box)

		self.right_box.addWidget(
			self.conf_graph,
			# alignment=QtCore.Qt.AlignmentFlag.AlignTop,
		)
		self.right_box.addLayout(
			self.data_box,
			# alignment=QtCore.Qt.AlignmentFlag.AlignTop,
		)

		self.right_box.addWidget(
			self.dataset_btn,
			# alignment=QtCore.Qt.AlignmentFlag.AlignTop,
		)

		self.right_box.addWidget(
			self.ai_chk,
			# alignment=QtCore.Qt.AlignmentFlag.AlignTop,
		)

		self.data_receive_btn.clicked.connect(self.data_receive)
		self.freq_spin.valueChanged.connect(self.change_freq)
		self.dataset_btn.clicked.connect(self.open_dataset_window)

	def update_conf(self, conf: np.ndarray, pred_idx: int):
		colors = []
		for i in range(len(classes)):
			if i == pred_idx:
				colors.append(pqtg.mkBrush(255, 80, 80, 220))
			else:
				colors.append(pqtg.mkBrush(100, 200, 255, 150))

		self.bar_item.setOpts(x1=conf, width=conf, brushes=colors)

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

	def clear_conf_plot(self):
		self.conf_graph.clear()
		self.bar_item = pqtg.BarGraphItem(
			x=np.zeros(len(classes)),
			x1=np.zeros(len(classes)),
			y=np.arange(len(classes)),
			height=0.6,
			width=np.zeros(len(classes)),
			brush=pqtg.mkBrush(100, 200, 255, 180),
		)
		self.conf_graph.addItem(self.bar_item)

	def data_receive(self):
		if not self.data_status:
			self.ai_chk.setEnabled(True)
			self.data_receive_btn.setText("Остановить приём сигнала")
			self.data_status_label.setObjectName("data-receive-btn-active")
			self.data_status_label.style().unpolish(self.data_status_label)
			self.data_status_label.style().polish(self.data_status_label)
			self.data_status_label.update()
			self.data_thread = DataThread()
			self.data_thread.start()
			self.data_thread.data_signal.connect(self.draw_plot)
			self.data_thread.info_signal.connect(self.update_info)
			self.data_thread.probs_signal.connect(self.update_conf)
			self.ai_chk.stateChanged.connect(lambda _: self.data_thread.ai_signal.emit(self.ai_chk.isChecked()))
			self.data_status = True
		else:
			if self.ai_chk.isChecked():
				self.ai_chk.setChecked(False)
			self.ai_chk.setEnabled(False)
			self.clear_info()
			self.clear_plot()
			self.clear_conf_plot()
			self.data_receive_btn.setText("Начать приём сигнала")
			self.data_status_label.setObjectName("data-receive-btn")
			self.data_status_label.style().unpolish(self.data_status_label)
			self.data_status_label.style().polish(self.data_status_label)
			self.data_status_label.update()
			self.data_thread.running = False
			self.data_status = False

	def closeEvent(self, event):
		if self.data_thread is not None:
			self.data_thread.running = False

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	with open("style.css", "r") as f:
		app.setStyleSheet(f.read())
	window = MainWindow(title="Program")
	window.show()
	sys.exit(app.exec())