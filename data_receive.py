# data_receive.py
import redpitaya_scpi as scpi
import numpy as np
import time
from scipy.signal import hilbert
from PyQt6 import QtCore

class RedPitayaReader(QtCore.QThread):
	data_ready = QtCore.pyqtSignal(np.ndarray)  # сигнал с данными

	def __init__(self, host='192.168.0.56', port=5000, num_samples=1024, parent=None):
		super().__init__(parent)
		self.num_samples = num_samples
		self.running = True
		self._data = None
		try:
			self.rp = scpi.scpi(host=host, port=port)
			self.set_generator_signal()
		except Exception as e:
			print("Error connecting to Red Pitaya:", e)
			self.running = False

	def run(self):
		while self.running:
			self.rp.acq_set(dec = 256)
			self.rp.acq_start()
			# self.rp.acq_get_settings()
			self.rp.tx_txt('ACQ:TRIG NOW')
			# Ждём триггер
			while self.running:
				self.rp.tx_txt('ACQ:TRIG:STAT?')
				if self.rp.rx_txt() == 'TD':
					break
				time.sleep(0.001)

			self.rp.acq_stop()
			# ch1 = self.rp.acq_data(chan = 1, num_samples = self.num_samples, last = True)
			ch2 = self.rp.acq_data(chan = 2, num_samples = self.num_samples, last = True)

			# analytic = hilbert(ch2)

			# iq = np.stack([analytic.real, analytic.imag], axis=0)
			iq = self.demodulate_to_iq(ch2, dec = 256)

			# iq = np.stack([ch2, ch1], axis=0)
			self._data = iq
			self.data_ready.emit(iq)

	def get_data(self):
		return self._data

	def stop(self):
		self.rp.acq_stop()
		self.running = False
		self.rp.close()

	def set_generator_signal(self):
		data = self.generate_bpsk()
		# freq_buf = 125e6 / 
		self.rp.gen_set(chan=2, func = scpi.Waveform.ARBITRARY, volt=0.5, freq = 300000, data = data[1], sdrlab = True)
		# self.rp.gen_get_settings(chan=2)

		self.rp.tx_txt("OUTPUT2:STATE ON")

	def generate_bpsk(self, n_samples=1024, sps=8):
		from scipy.signal import firwin, lfilter
		
		N_BUF = 16384
		n_symbols = n_samples // sps
		bits = np.random.randint(0, 2, n_symbols)
		symbols = (2 * bits - 1).astype(float)  # ±1
		
		# Модуляция с несущей для подачи на DAC
		baseband = np.repeat(symbols, sps)[:n_samples]
		
		# RRC фильтр
		num_taps = 101
		rrc = firwin(num_taps, 1.0 / sps, window='hamming')
		baseband = lfilter(rrc, 1.0, baseband)
		baseband = baseband / (np.max(np.abs(baseband)) + 1e-8)
		
		# Модулируем на несущую 300 кГц для DAC
		fs_dac = 125e6        # DAC Red Pitaya
		fc = 300e3
		t = np.arange(N_BUF) / fs_dac
		
		baseband_tiled = np.tile(baseband, N_BUF // n_samples)
		signal_rp = baseband_tiled * np.cos(2 * np.pi * fc * t)
		signal_rp = np.clip(signal_rp * 0.45, -0.5, 0.5)
		
		# IQ для нейронки (реальные I/Q в baseband)
		iq = np.stack([baseband, np.zeros(n_samples)], axis=0).astype(np.float32)
		
		return iq, signal_rp
	
	def demodulate_to_iq(self, ch, carrier_freq=300000, dec=8, out_samples=1024):
		sample_rate = 125e6 / dec
		n = len(ch)
		t = np.arange(n) / sample_rate

		i_raw = ch * np.cos(2 * np.pi * carrier_freq * t)
		q_raw = ch * -np.sin(2 * np.pi * carrier_freq * t)

		from scipy.signal import butter, filtfilt
		bw = 50e3
		b, a = butter(4, bw / (sample_rate / 2), btype='low')
		i_f = filtfilt(b, a, i_raw)
		q_f = filtfilt(b, a, q_raw)

		# Коррекция фазового сдвига
		phase = np.arctan2(np.mean(q_f), np.mean(i_f))
		i_corr = i_f * np.cos(-phase) - q_f * np.sin(-phase)
		q_corr = i_f * np.sin(-phase) + q_f * np.cos(-phase)

		# Нормализация по мощности
		power = np.mean(i_corr**2 + q_corr**2) + 1e-8
		i_n = (i_corr / np.sqrt(power)).astype(np.float32)
		q_n = (q_corr / np.sqrt(power)).astype(np.float32)

		return np.stack([i_n[:out_samples], q_n[:out_samples]], axis=0)
	
	def closeEvent(self, event):
		self.stop()