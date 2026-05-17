from PyQt6 import QtCore
import numpy as np
import torch
import time
import os
import csv
from handler.data_receive import RedPitayaReader
from datetime import datetime
from constants import CLASSES as classes

class DataThread(QtCore.QThread):
	data_signal = QtCore.pyqtSignal(list)
	info_signal = QtCore.pyqtSignal(str, float, float, float, bool)
	ai_signal = QtCore.pyqtSignal(bool)
	probs_signal = QtCore.pyqtSignal(np.ndarray, int)
	power_signal = QtCore.pyqtSignal(float)

	def __init__(self, model_path):
		super().__init__()
		self.data = None
		self.running = True
		self.ai = False
		self.model_path = model_path
		self._latest_iq = None
		self.power = 0.0

		self.rp = RedPitayaReader()
		self.rp.data_ready.connect(self._on_iq_received)
		self.rp.start()

		try:
			import onnxruntime as ort
			self.sess = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
		except Exception as e:
			self.sess = None
			print("AI error" , e)
		self.ai_signal.connect(self.ai_state)

	def _on_iq_received(self, iq: np.ndarray, power_db):
		self._latest_iq = iq
		self.power = power_db

	def run(self):
		csv_file = self.csv_headers_write(["Индекс сигнала", "Статус сигнала", "Класс", "Вероятность в %", "Скорость обработки в мс"])
		self.index = 0
		while self.running:
			# iq = self.rp.get_data()
			# iq = self.generate_bpsk()
			# print(self._latest_iq)
			if self._latest_iq is not None:
				iq = self._latest_iq
			else:
				self.msleep(10)  # ждём первых данных
				continue

			
			# detected, power_db = self.energy_detector(iq, -40)
			# snr = self.estimate_snr_m2m4(iq)
			snr = 0.0

			self.data = iq.T.tolist()
			# i_norm = iq[0] / np.sqrt(np.mean(iq[0]**2 + iq[1]**2))
			# q_norm = iq[1] / np.sqrt(np.mean(iq[0]**2 + iq[1]**2))
			# iq = np.stack([i_norm, q_norm], axis=0)
			
			if self.ai:
				# if detected:
					pred_idx, confidence, speed_ai, probs = self.ai_proc(iq)
					self.info_signal.emit(classes[pred_idx], confidence, speed_ai, snr, False)
					self.probs_signal.emit(probs, pred_idx)
					self.csv_data_write(csv_file, self.index, True, classes[pred_idx], confidence, speed_ai)
					self.index += 1
				# else:
					# self.info_signal.emit("", 0.0, 0.0, snr, False)
					# self.csv_data_write(csv_file, self.index, False, "", 0.0, 0.0)
					# self.index += 1
			self.power_signal.emit(self.power)
			self.data_signal.emit(self.data)
			self.msleep(1000)

	def ai_proc(self, data):
		if self.sess is None:
			return
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
 
	def csv_headers_write(self, headers):
		if not os.path.exists("results"):
			os.mkdir("results")
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		csv_string = "results/result_" + str(timestamp) +".csv"
		with open(csv_string, "w", newline="") as f:
			writer = csv.writer(f, delimiter=";")
			writer.writerow(headers)

		return csv_string

	def csv_data_write(self, name_file, index, status_signal, class_index, confidence, speed_ai):
		confidence = f"{confidence * 100:.2f}"
		speed_ai = f'="{speed_ai:.2f}"'
		with open(name_file, "a", newline="") as f:
			writer = csv.writer(f, delimiter=";")
			writer.writerow([index, status_signal, class_index, confidence, speed_ai])

	# def energy_detector(self, iq, threshold):
	# 	signal = iq[0] + 1j * iq[1]

	# 	power_w = np.mean(np.abs(signal) ** 2) / 50.0
	# 	power_db = 10 * np.log10(power_w + 1e-12)

	# 	detected = power_db >= threshold
	# 	return detected, power_db
	
	def stop(self):
		if self.rp is not None:
			self.rp.closeEvent(None)
			self.rp = None
		self.running = False

	# def estimate_snr_m2m4(self, iq):
	# 	"""SNR через метод моментов M2M4. Работает для любой модуляции."""
	# 	signal = iq[0] + 1j * iq[1]
		
	# 	m2 = np.mean(np.abs(signal) ** 2)   # второй момент
	# 	m4 = np.mean(np.abs(signal) ** 4)   # четвёртый момент
		
	# 	# Отношение моментов
	# 	ratio = m4 / (m2 ** 2 + 1e-12)
		
	# 	# SNR из отношения моментов
	# 	snr_linear = 1 / (ratio - 1 + 1e-12)
	# 	snr_db = 10 * np.log10(max(snr_linear, 1e-12))
		
	# 	return snr_db
