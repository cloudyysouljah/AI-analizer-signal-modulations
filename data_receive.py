# data_receive.py
import redpitaya_scpi as scpi
import numpy as np
import time
from PyQt6 import QtCore

class RedPitayaReader(QtCore.QThread):
	data_ready = QtCore.pyqtSignal(np.ndarray)  # сигнал с данными

	def __init__(self, host='192.168.1.205', port=5000, num_samples=1024, parent=None):
		super().__init__(parent)
		self.num_samples = num_samples
		self.running = True
		self._data = None

		self.rp = scpi.scpi(host=host, port=port)

	def run(self):
		while self.running:
			self.rp.acq_start()
			self.rp.tx_txt('ACQ:TRIG NOW')
			# Ждём триггер
			while self.running:
				self.rp.tx_txt('ACQ:TRIG:STAT?')
				if self.rp.rx_txt() == 'TD':
					break
				time.sleep(0.001)

			self.rp.acq_stop()
			ch1 = self.rp.acq_data(chan = 1, num_samples = self.num_samples, last = True)
			ch2 = self.rp.acq_data(chan = 2, num_samples = self.num_samples, last = True)
			iq = np.stack([ch1, ch2], axis=0)
			self._data = iq
			self.data_ready.emit(iq)

	def get_data(self):
		return self._data

	def stop(self):
		self.running = False
		self.rp.tx_txt('ACQ:STOP')
		self.rp.close()