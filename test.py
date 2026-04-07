import sys
import h5py
import numpy as np
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg

CLASSES = [
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
    '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
    '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
    'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK'
]

class DatasetBrowser(QtWidgets.QWidget):
    def __init__(self, hdf5_path):
        super().__init__()
        self.setWindowTitle("RadioML Browser")
        self.resize(1100, 600)

        self.idx = 0

        # Загружаем срез датасета
        with h5py.File(hdf5_path, 'r') as f:
            self.X = f['X'][:5000]   # shape [N, 1024, 2]
            self.Y = f['Y'][:5000]
            self.Z = f['Z'][:5000]

        self._build_ui()
        self._update()

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)

        # --- Левая панель: графики ---
        plots_layout = QtWidgets.QVBoxLayout()

        # Временной график
        self.time_plot = pg.PlotWidget(title="Временная область")
        self.time_plot.addLegend()
        self.curve_I = self.time_plot.plot(pen=pg.mkPen('c', width=1), name='I')
        self.curve_Q = self.time_plot.plot(pen=pg.mkPen('y', width=1), name='Q')

        # Спектр
        self.freq_plot = pg.PlotWidget(title="Спектр (FFT)")
        self.curve_fft = self.freq_plot.plot(pen=pg.mkPen('g', width=1))

        # IQ-созвездие
        self.iq_plot = pg.PlotWidget(title="IQ-созвездие")
        self.iq_plot.setAspectLocked(True)
        self.scatter = pg.ScatterPlotItem(size=2, pen=None, brush=pg.mkBrush(255, 100, 100, 120))
        self.iq_plot.addItem(self.scatter)

        plots_layout.addWidget(self.time_plot)
        plots_layout.addWidget(self.freq_plot)
        plots_layout.addWidget(self.iq_plot)

        # --- Правая панель: управление ---
        ctrl_layout = QtWidgets.QVBoxLayout()
        ctrl_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        # Информация о сигнале
        self.info_label = QtWidgets.QLabel()
        self.info_label.setStyleSheet("font-size: 14px;")
        self.info_label.setWordWrap(True)

        # Навигация по индексу
        nav_layout = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("◀ Пред")
        self.next_btn = QtWidgets.QPushButton("След ▶")
        self.prev_btn.clicked.connect(self._prev)
        self.next_btn.clicked.connect(self._next)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)

        # Прямой переход по индексу
        self.idx_spin = QtWidgets.QSpinBox()
        self.idx_spin.setRange(0, len(self.X) - 1)
        self.idx_spin.valueChanged.connect(self._goto)

        # Фильтр по классу
        self.class_combo = QtWidgets.QComboBox()
        self.class_combo.addItem("Все классы")
        self.class_combo.addItems(CLASSES)
        self.class_combo.currentIndexChanged.connect(self._filter_changed)

        # Фильтр по SNR
        self.snr_combo = QtWidgets.QComboBox()
        self.snr_combo.addItem("Все SNR")
        unique_snr = sorted(np.unique(self.Z[:, 0]).astype(int).tolist())
        for s in unique_snr:
            self.snr_combo.addItem(f"{s} дБ")
        self.snr_combo.currentIndexChanged.connect(self._filter_changed)

        # Собираем правую панель
        ctrl_layout.addWidget(QtWidgets.QLabel("Индекс:"))
        ctrl_layout.addWidget(self.idx_spin)
        ctrl_layout.addLayout(nav_layout)
        ctrl_layout.addSpacing(16)
        ctrl_layout.addWidget(QtWidgets.QLabel("Класс:"))
        ctrl_layout.addWidget(self.class_combo)
        ctrl_layout.addWidget(QtWidgets.QLabel("SNR:"))
        ctrl_layout.addWidget(self.snr_combo)
        ctrl_layout.addSpacing(16)
        ctrl_layout.addWidget(self.info_label)

        # Индексы после фильтрации
        self.filtered_indices = list(range(len(self.X)))
        self.filter_pos = 0

        root.addLayout(plots_layout, stretch=3)
        root.addLayout(ctrl_layout, stretch=1)

    def _get_filtered(self):
        class_filter = self.class_combo.currentIndex() - 1  # -1 = все
        snr_text = self.snr_combo.currentText()
        snr_filter = None if snr_text == "Все SNR" else int(snr_text.replace(" дБ", ""))

        indices = []
        for i in range(len(self.X)):
            if class_filter >= 0 and np.argmax(self.Y[i]) != class_filter:
                continue
            if snr_filter is not None and int(self.Z[i, 0]) != snr_filter:
                continue
            indices.append(i)
        return indices

    def _filter_changed(self):
        self.filtered_indices = self._get_filtered()
        self.filter_pos = 0
        if self.filtered_indices:
            self.idx = self.filtered_indices[0]
            self._update()

    def _prev(self):
        if not self.filtered_indices:
            return
        self.filter_pos = (self.filter_pos - 1) % len(self.filtered_indices)
        self.idx = self.filtered_indices[self.filter_pos]
        self._update()

    def _next(self):
        if not self.filtered_indices:
            return
        self.filter_pos = (self.filter_pos + 1) % len(self.filtered_indices)
        self.idx = self.filtered_indices[self.filter_pos]
        self._update()

    def _goto(self, value):
        self.idx = value
        self._update()

    def _update(self):
        sample = self.X[self.idx]          # [1024, 2]
        I = sample[:, 0]
        Q = sample[:, 1]
        label = CLASSES[np.argmax(self.Y[self.idx])]
        snr = int(self.Z[self.idx, 0])

        # Временной график — показываем первые 256 точек для читаемости
        self.curve_I.setData(I[:1024])
        self.curve_Q.setData(Q[:1024])

        # Спектр
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(I + 1j * Q)))
        self.curve_fft.setData(spectrum)

        # Созвездие
        self.scatter.setData(x=I.tolist(), y=Q.tolist())

        # Инфо
        self.info_label.setText(
            f"Индекс: {self.idx}\n"
            f"Класс:  {label}\n"
            f"SNR:    {snr} дБ\n"
            f"Всего по фильтру: {len(self.filtered_indices)}"
        )

        # Обновляем спиннер без рекурсии
        self.idx_spin.blockSignals(True)
        self.idx_spin.setValue(self.idx)
        self.idx_spin.blockSignals(False)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    path = '/home/leonid/Загрузки/archive (2)/GOLD_XYZ_OSC.0001_1024.hdf5'  # укажи свой путь
    win = DatasetBrowser(path)
    win.show()
    sys.exit(app.exec())