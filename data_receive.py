import redpitaya_scpi as scpi
import numpy as np
import time
from PyQt6 import QtCore

class RedPitayaReader(QtCore.QThread):
	data_ready = QtCore.pyqtSignal(np.ndarray, float)

	def __init__(self, host='192.168.1.120', port=5000, num_samples=1024, parent=None):
		super().__init__(parent)
		self.num_samples = num_samples
		self.running = True
		self._data = None
		self.state_gen = False
		self.data_mutex = QtCore.QMutex()
		try:
			self.rp = scpi.scpi(host=host, port=port)
			self.set_generator_signal()
		except Exception as e:
			print("Error connecting to Red Pitaya:", e)
			self.running = False

	def run(self):
		while self.running:
			self.data_mutex.lock()
			self.rp.acq_set(dec = 16)
			self.rp.acq_start()
			# Ждём триггер
			while self.running:
				self.rp.tx_txt('ACQ:TRIG NOW')
				self.rp.tx_txt('ACQ:TRIG:STAT?')
				if self.rp.rx_txt() == 'TD':
					break
				time.sleep(0.001)

			while self.running:
				self.rp.tx_txt('ACQ:TRIG:FILL?')
				if self.rp.rx_txt() == '1':
					break
				time.sleep(0.001)

			ch2 = self.rp.acq_data(chan=2, num_samples=16384, last=True)
			
			iq, power = self.signal_to_iq(ch = ch2, out_samples = 1024)

			self._data = iq
			self.data_ready.emit(iq, power)
			self.data_mutex.unlock()

	def get_data(self):
		return self._data

	def set_generator_signal(self):
		data = self.generate_bpsk()
		
		data = np.real(data)
		data = np.nan_to_num(data)

		data = data / np.max(np.abs(data) + 1e-12)
		data = np.clip(data, -0.5, 0.5)

		data = np.ascontiguousarray(data, dtype=np.float32)
		# Real-only DAC transmission (baseband on one channel)
		i_sig = np.real(data).astype(np.float32)

		i_sig /= np.max(np.abs(i_sig)) + 1e-12

		self.rp.gen_set(
			chan=2,
			func=scpi.Waveform.ARBITRARY,
			volt=0.5,
			freq=3e5,
			data=data,
			sdrlab=True
		)

		self.rp.tx_txt("OUTPUT2:STATE ON")
		if self.rp.txrx_txt("OUTPUT2:STATE?") != "ON":
			self.state_gen = False
		else:
			self.state_gen = True

	def generate_bpsk(self):
		from scipy.signal import firwin, lfilter

		N_BUF = 16384
		sps = 32
		n_symbols = N_BUF // sps

		bits = np.random.randint(0, 2, n_symbols)
		symbols = (2 * bits - 1).astype(float)  # ±1, не комплексные

		baseband = np.repeat(symbols, sps)
		taps = firwin(129, 1.0 / sps, window='hamming')
		baseband = lfilter(taps, 1.0, baseband)
		baseband -= np.mean(baseband)
		baseband /= (np.max(np.abs(baseband)) + 1e-12)

		return baseband.astype(np.complex64)

	def signal_to_iq(self, ch, out_samples=1024):
		ch = ch.astype(np.float32)

		# DC removal
		ch -= np.mean(ch)
		iq1 = np.stack([ch, np.zeros_like(ch)], axis=0)
		snr = self.estimate_snr(ch)
		power_db = self.energy_detector(iq1, -40)
		print(f"SNR: {snr:.1f} dB, Power: {power_db:.1f} dBm")
		# RadioML-style normalization
		std = np.std(ch)
		if std < 1e-8:
			std = 1e-8

		ch /= std
		# downsample without distortion
		step = len(ch) // out_samples
		ch = ch[::step][:out_samples]

		iq = np.stack([ch, np.zeros_like(ch)], axis=0)

		return iq.astype(np.float32), power_db

	def energy_detector(self, iq, threshold):
		signal = iq[0] + 1j * iq[1]
		power_w = np.mean(np.abs(signal) ** 2) / 50.0
		power_dbm = 10 * np.log10(power_w * 1000 + 1e-12)  # дБм
		return power_dbm

	def estimate_snr(self, ch):
		"""SNR через метод моментов M2M4 для BPSK"""
		# Нормализуем
		x = ch - np.mean(ch)
		
		m2 = np.mean(x ** 2)
		m4 = np.mean(x ** 4)
		
		if m2 < 1e-12:
			return 0.0
		
		# Для BPSK теоретический куртозис = 1.0
		# При наличии шума: m4/m2² → 3 (гауссов шум) или 1 (чистый BPSK)
		kurtosis = m4 / (m2 ** 2 + 1e-12)
		print(f"Kurtosis: {kurtosis:.3f}")  # чистый BPSK=1, шум=3
		
		# SNR из куртозиса
		# kurtosis = (1 + 2/SNR + 3/SNR²) для BPSK+AWGN
		# Упрощённо: SNR ≈ 2 / (kurtosis - 1)
		if kurtosis <= 1.0:
			return 40.0  # очень чистый сигнал
		
		snr_linear = 2.0 / (kurtosis - 1.0 + 1e-12)
		snr_db = 10 * np.log10(max(snr_linear, 1e-12))
		
		return snr_db

	def stop(self):
		self.running = False
		self.data_mutex.lock()
		self.rp.acq_stop()
		self.data_mutex.unlock()

	def closeEvent(self, event):
		self.stop()
		if self.state_gen:
			self.rp.tx_txt("OUTPUT2:STATE OFF")
		self.rp.close()
