import os

from datetime import datetime

import numpy as np
import pyqtgraph as pqtg
from PyQt6 import QtCore, QtWidgets

from graphic.base_win import BaseWindow
from graphic.dataset_win import DatasetWindow
from handler.inference import DataThread

from constants import CLASSES as classes

class MainWindow(BaseWindow):
	def __init__(self, title=None):
		super().__init__()

		self.setWindowTitle(title or "Program")

		self.data_thread = None

		self.data_status = False
		self.model_path = None

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

		self.signal_info_box = QtWidgets.QVBoxLayout()
		self.threshold_box = QtWidgets.QHBoxLayout()
		self.power_label = QtWidgets.QLabel("Мощность сигнала в дб: -")
		self.signal_label = QtWidgets.QLabel("◉ Нет сигнала")
		self.threshold_label = QtWidgets.QLabel("Уровень порога для определения сигнала в дб:")
		self.threshold_spin = QtWidgets.QDoubleSpinBox(minimum=-40, maximum=50, value=-10, decimals=2, singleStep=0.1)
		self.signal_info_box.addWidget(self.signal_label)
		self.signal_info_box.addWidget(self.power_label)
		self.threshold_box.addWidget(self.threshold_label)
		self.threshold_box.addWidget(self.threshold_spin)
		self.signal_info_box.addLayout(self.threshold_box)

		self.data_box = QtWidgets.QHBoxLayout()
		self.data_receive_btn = QtWidgets.QPushButton("Начать приём сигнала")
		self.data_status_label = QtWidgets.QLabel()
		self.data_status_label.setFixedSize(30, 30)
		self.data_status_label.setObjectName("data-receive-btn")
		self.ai_chk = QtWidgets.QCheckBox("AI")
		self.ai_chk.setEnabled(False)
		self.data_box.addWidget(self.data_receive_btn, 1)
		self.data_box.addWidget(self.data_status_label)
		self.data_box.addWidget(self.ai_chk)

		self.dataset_btn = QtWidgets.QPushButton("Загрузить датасет")

		self.right_box.addLayout(self.signal_info_box)
		self.right_box.addWidget(self.conf_graph)
		self.right_box.addLayout(self.data_box)

		self.right_box.addWidget(self.dataset_btn)

		self.data_receive_btn.clicked.connect(self.data_receive)
		self.dataset_btn.clicked.connect(self.open_dataset_window)

	def update_signal_info(self, power_db):
		self.power_label.setText(f"Мощность сигнала в дб: {power_db:.4f}")
		if power_db > self.threshold_spin.value():
			self.signal_label.setText("◉ Сигнал")
			self.signal_label.setStyleSheet("color: #00ff00; font-weight: bold;")
		else:
			self.signal_label.setText("◉ Нет сигнала")
			self.signal_label.setStyleSheet("color: #ff4444; font-weight: bold;")

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

	def get_model_path(self):
		model_path, _= QtWidgets.QFileDialog.getOpenFileName(
			self.parent(),
			caption="Выберите файл модели в формате onnx",
			directory=os.path.expanduser("~"),
			filter="*.onnx",
		)
		return model_path

	def data_receive(self):
		if not self.data_status:
			self.ai_chk.setEnabled(True)
			self.data_receive_btn.setText("Остановить приём сигнала")
			self.data_status_label.setObjectName("data-receive-btn-active")
			self.data_status_label.style().unpolish(self.data_status_label)
			self.data_status_label.style().polish(self.data_status_label)
			self.data_status_label.update()
			self.data_thread = DataThread(self.get_model_path())
			self.data_thread.start()
			self.data_thread.data_signal.connect(self.draw_plot)
			self.data_thread.info_signal.connect(self.update_info)
			self.data_thread.probs_signal.connect(self.update_conf)
			self.data_thread.power_signal.connect(self.update_signal_info)
			self.ai_chk.stateChanged.connect(lambda _: self.data_thread.ai_signal.emit(self.ai_chk.isChecked()))
			self.data_status = True
		else:
			if self.ai_chk.isChecked():
				self.ai_chk.setChecked(False)
			self.ai_chk.setEnabled(False)

			self.data_thread.rp.closeEvent(None)
			self.data_thread.rp = None
			self.data_receive_btn.setText("Начать приём сигнала")
			self.data_status_label.setObjectName("data-receive-btn")
			self.data_status_label.style().unpolish(self.data_status_label)
			self.data_status_label.style().polish(self.data_status_label)
			self.data_status_label.update()
			self.data_thread.running = False
			self.data_status = False
			self.clear_info()
			self.clear_plot()
			self.clear_conf_plot()

	def closeEvent(self, event):
		if self.data_thread is not None:
			self.data_thread.stop()