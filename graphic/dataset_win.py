from PyQt6 import QtCore, QtWidgets
import h5py
import numpy as np

from constants import CLASSES
from graphic.base_win import BaseWindow
from graphic.train_win import TrainWindow


class DatasetWindow(BaseWindow):
	dataset_signal = QtCore.pyqtSignal(list)
	info_signal = QtCore.pyqtSignal(bool)

	def __init__(self, parent=None, dataset=None, title=None):
		super().__init__(parent, title)

		self.dataset_path = dataset
		self.dataset = None
		self.range_arr = np.array([], dtype=np.int64)
		self.class_index = 0
		self.current_index = 0
		self.current_snr = None
		self.labels, self.snr_values = self.load_metadata(self.dataset_path)
		self.filtered_indices = np.arange(len(self.labels), dtype=np.int64)

		self.index_spin = QtWidgets.QSpinBox()
		self.index_spin.setRange(0, max(0, len(self.filtered_indices) - 1))
		self.index_spin.setEnabled(False)

		self.class_combo = QtWidgets.QComboBox()
		self.class_combo.addItem("Все классы", None)
		for class_index, class_name in enumerate(CLASSES):
			self.class_combo.addItem(class_name, class_index)

		self.snr_combo = QtWidgets.QComboBox()
		self.snr_combo.addItem("Все SNR", None)
		for snr in sorted(np.unique(self.snr_values).tolist()):
			self.snr_combo.addItem(f"{snr:g} dB", snr)

		self.train_btn = QtWidgets.QPushButton("Начать обучение")
		self.prev_btn = QtWidgets.QPushButton("Предыдущий")
		self.next_btn = QtWidgets.QPushButton("Следующий")
		self.show_dataset_btn = QtWidgets.QPushButton("Показать")

		self.right_box.addWidget(self.index_spin)
		self.right_box.addWidget(self.class_combo)
		self.right_box.addWidget(self.snr_combo)
		self.right_box.addWidget(self.train_btn)
		self.right_box.addWidget(self.show_dataset_btn)
		self.right_box.addWidget(self.prev_btn)
		self.right_box.addWidget(self.next_btn)

		self.train_btn.clicked.connect(self.start_train)
		self.show_dataset_btn.clicked.connect(self.start_parse_dataset)
		self.next_btn.clicked.connect(self.next_index)
		self.prev_btn.clicked.connect(self.prev_index)
		self.dataset_signal.connect(self.draw_plot)
		self.info_signal.connect(lambda info: self.update_info(dataset_win=info))
		self.index_spin.valueChanged.connect(self.index_spin_changed)
		self.class_combo.currentIndexChanged.connect(self.apply_filters)
		self.snr_combo.currentIndexChanged.connect(self.apply_filters)

	def start_train(self):
		self.train_window = TrainWindow(
			parent=self,
			title="Обучение",
			samples=self.get_dataset_size(self.dataset_path),
			path=self.dataset_path,
		)
		self.train_window.show()

	def start_parse_dataset(self):
		self.index_spin.setEnabled(len(self.filtered_indices) > 0)
		self.show_current_sample()

	def apply_filters(self):
		class_filter = self.class_combo.currentData()
		snr_filter = self.snr_combo.currentData()

		mask = np.ones(len(self.labels), dtype=bool)
		if class_filter is not None:
			mask &= self.labels == class_filter
		if snr_filter is not None:
			mask &= self.snr_values == snr_filter

		self.filtered_indices = np.where(mask)[0].astype(np.int64)
		self.dataset = None
		self.range_arr = np.array([], dtype=np.int64)

		self.index_spin.blockSignals(True)
		self.index_spin.setRange(0, max(0, len(self.filtered_indices) - 1))
		self.index_spin.setValue(0)
		self.index_spin.setEnabled(len(self.filtered_indices) > 0)
		self.index_spin.blockSignals(False)

		if len(self.filtered_indices) == 0:
			self.clear_plot()
			self.clear_info()
			return

		self.show_current_sample()

	def update_dataset(self, index):
		if self.dataset is not None and index in self.range_arr:
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

	def show_current_sample(self):
		if len(self.filtered_indices) == 0:
			return

		index = int(self.filtered_indices[self.index_spin.value()])
		self.update_dataset(index)
		local = index - self.range_arr.min()

		self.current_index = index
		self.current_snr = self.snr_values[index]
		self.class_index = int(np.argmax(self.dataset[1][local]))

		self.dataset_signal.emit(self.dataset[0][local])
		self.info_signal.emit(True)

	def index_spin_changed(self):
		self.show_current_sample()

	def next_index(self):
		self.index_spin.setValue(min(self.index_spin.value() + 1, self.index_spin.maximum()))

	def prev_index(self):
		self.index_spin.setValue(max(self.index_spin.value() - 1, self.index_spin.minimum()))

	def load_metadata(self, path):
		with h5py.File(path, "r") as f:
			y_data = f["Y"][:]
			z_data = f["Z"][:].reshape(-1)
		return np.argmax(y_data, axis=1).astype(np.int64), z_data.astype(float)

	def get_dataset_size(self, path):
		with h5py.File(path, "r") as f:
			len_dataset = f["X"].shape[0]
		return len_dataset
