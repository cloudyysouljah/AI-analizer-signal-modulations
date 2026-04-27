import torch
import gc
import os
import h5py
import time
import onnxruntime as ort
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from PyQt6.QtCore import QThread, pyqtSignal

SAMPLE_LEN = 1024

CLASSES = [
	"OOK",       "4ASK",      "8ASK",      "BPSK",
	"QPSK",      "8PSK",      "16PSK",     "32PSK",
	"16APSK",    "32APSK",    "64APSK",    "128APSK",
	"16QAM",     "32QAM",     "64QAM",     "128QAM",
	"256QAM",    "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC",
	"AM-DSB-SC", "FM",        "GMSK",      "OQPSK",
]

class ResBlock1D(nn.Module):
	def __init__(self, channels: int, kernel_size: int = 3):
		super().__init__()
		pad = kernel_size // 2
		self.block = nn.Sequential(
			nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
			nn.BatchNorm1d(channels),
			nn.GELU(),
			nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
			nn.BatchNorm1d(channels),
		)
		self.act = nn.GELU()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.act(x + self.block(x))


class RadioMLNet(nn.Module):
	def __init__(self, num_classes: int, dropout: float = 0.4):
		super().__init__()

		self.stem = nn.Sequential(
			nn.Conv1d(2, 32, 7, padding=3, bias=False),
			nn.BatchNorm1d(32),
			nn.GELU(),
			nn.MaxPool1d(2),
		)
		self.stage1 = nn.Sequential(
			ResBlock1D(32), ResBlock1D(32),
			nn.Conv1d(32, 64, 3, padding=1, bias=False),
			nn.BatchNorm1d(64),
			nn.GELU(),
			nn.MaxPool1d(2),
		)
		self.stage2 = nn.Sequential(
			ResBlock1D(64), ResBlock1D(64),
			nn.Conv1d(64, 128, 3, padding=1, bias=False),
			nn.BatchNorm1d(128),
			nn.GELU(),
			nn.MaxPool1d(2),
		)
		self.stage3 = nn.Sequential(
			ResBlock1D(128), ResBlock1D(128),
			nn.Conv1d(128, 256, 3, padding=1, bias=False),
			nn.BatchNorm1d(256),
			nn.GELU(),
			nn.MaxPool1d(2),
		)

		self.lstm = nn.LSTM(
			input_size=256,
			hidden_size=256,
			num_layers=2,
			batch_first=True,
			dropout=dropout,
			bidirectional=True,
		)
		self.head = nn.Sequential(
			nn.Linear(512, 512), nn.GELU(), nn.Dropout(dropout),
			nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout / 2),
			nn.Linear(256, num_classes),
		)

		self._init_weights()

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
			elif isinstance(m, nn.BatchNorm1d):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				nn.init.zeros_(m.bias)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.stem(x)
		x = self.stage1(x)
		x = self.stage2(x)
		x = self.stage3(x)
		x, _ = self.lstm(x.permute(0, 2, 1))
		return self.head(x[:, -1, :])

class Train(QThread):
	def __init__(
		self,
		parent=None,
		path: str = None,
		save_path: str = "best_model_test.pt",
		batch_size: int = 256,
		lr: float = 1e-3,
		epochs: int = 50,
		patience: int = 15,
		snr_min: int = -20,
		max_samples = None
	):
		super(Train, self).__init__()

		self.path      = path
		self.parent    = parent
		self.save_path = save_path
		self.batch_size = batch_size
		self.lr        = lr
		self.epochs    = epochs
		self.patience  = patience   # ✅ теперь атрибут, не локальная переменная
		self.snr_min   = snr_min
		self.max_samples = max_samples

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.model = RadioMLNet(num_classes=len(CLASSES), dropout=0.4)
		self.model.to(device=self.device)

	def run(self):
		with h5py.File(self.path, "r") as f:
			total = f["X"].shape[0]
			n     = min(self.max_samples, total) if self.max_samples else total  # ✅
			self.parent.log_signal.emit(f"📦 Загружаю {n:,} / {total:,} сэмплов")
			x_data = f["X"][:n].astype(np.float16)
			y_data = f["Y"][:n].astype(np.float32)
			y_data = np.argmax(y_data, axis=1).astype(np.int64)

		train_dataset, test_dataset, valid_dataset = self.make_tensor_datasets(x_data, y_data)
		del x_data, y_data
		gc.collect()
		self.parent.log_signal.emit("🧹 Исходный numpy массив освобождён")

		params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		self.parent.log_signal.emit(f"🧠 Параметров: {params:,}  |  Классов: {len(CLASSES)}: {CLASSES}")

		if os.name == "nt":
			nw, pw, pf = 0, False, None
			self.parent.log_signal.emit("💡 Windows: num_workers=0")
		else:
			nw, pw, pf = 4, True, 2
			self.parent.log_signal.emit(f"💡 Linux: num_workers={nw}, persistent_workers=True")

		pm = self.device.type == "cuda"
		loader_kwargs = dict(
			num_workers=nw,
			pin_memory=pm,
			persistent_workers=pw,
			**({"prefetch_factor": pf} if pf else {}),
		)

		tr_loader   = DataLoader(train_dataset, batch_size=self.batch_size,     shuffle=True,  **loader_kwargs)
		vl_loader   = DataLoader(valid_dataset, batch_size=self.batch_size * 2, shuffle=False, **loader_kwargs)
		test_loader = DataLoader(test_dataset,  batch_size=self.batch_size * 2, shuffle=False, **loader_kwargs)

		criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
		optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
		scheduler = optim.lr_scheduler.OneCycleLR(
			optimizer,
			max_lr=self.lr,
			epochs=self.epochs,
			steps_per_epoch=len(tr_loader),
			pct_start=0.1,
			anneal_strategy="cos",
		)
		scaler = GradScaler("cuda")

		history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
		best_val_acc = 0.0
		no_improve   = 0  # ✅ вынесено из цикла

		for epoch in range(1, self.epochs + 1):
			tr_loss, tr_acc, train_time = self.train_epoch(
				self.model, tr_loader, criterion, optimizer, scaler, scheduler, self.device)
			vl_loss, vl_acc = self.eval_epoch(self.model, vl_loader, criterion, self.device)

			history["train_loss"].append(tr_loss)
			history["val_loss"].append(vl_loss)
			history["train_acc"].append(tr_acc)
			history["val_acc"].append(vl_acc)

			vram_str = ""
			if self.device.type == "cuda":
				vram_str = f"  VRAM: {torch.cuda.memory_reserved(0) / 1024 ** 3:.1f} GB"

			saved = ""
			if vl_acc > best_val_acc:
				best_val_acc = vl_acc
				no_improve   = 0  # ✅ сбрасываем только при улучшении
				torch.save(
					{
						"epoch":       epoch,
						"model_state": self.model.state_dict(),
						"classes":     CLASSES,
						"val_acc":     vl_acc,
						"snr_min":     self.snr_min,
					},
					self.save_path,
				)
				saved = "  💾 сохранено"
			else:
				no_improve += 1  # ✅ инкрементируется корректно

			self.parent.progress_update.emit(epoch)
			self.parent.log_signal.emit(
				f"Epoch {epoch:3d}/{self.epochs} | "
				f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
				f"Acc {tr_acc:.4f}/{vl_acc:.4f} | "
				f"Train time {train_time:.2f} sec"
				f"{vram_str}{saved}"
			)

			if no_improve >= self.patience:  # ✅ используем self.patience
				self.parent.log_signal.emit(f"🛑 Слишком мало улучшений за последние {self.patience} эпох")
				break

		self.parent.log_signal.emit("\n🔄 Загружаю лучшую модель для финальной оценки...")
		ckpt = torch.load(self.save_path, map_location=self.device)
		self.model.load_state_dict(ckpt["model_state"])

		test_loss, test_acc = self.eval_epoch(self.model, test_loader, criterion, self.device)
		self.parent.log_signal.emit(f"🏁 Test Accuracy: {test_acc:.4f}  |  Test Loss: {test_loss:.4f}")
		self.parent.log_signal.emit(f"🏆 Лучший Val Accuracy: {best_val_acc:.4f}")

		self.model.eval()
		self.model.to("cpu")

		example_input = torch.randn(1, 2, 1024, dtype=torch.float32)

		onnx_path = self.save_path.replace(".pt", ".onnx")

		torch.onnx.export(
			self.model,
			example_input,
			onnx_path,
			export_params=True,
			opset_version=17,
			input_names=["input"],
			output_names=["output"],
			dynamic_axes={
				"input": {0: "batch"},
				"output": {0: "batch"},
			},
		)
		self.parent.log_signal.emit("✅ Успешно сохранил модель в ONNX")
		self.parent.progress_update.emit(self.epochs)

	def train_epoch(
		self,
		model: nn.Module,
		loader: DataLoader,
		criterion: nn.Module,
		optimizer: torch.optim.Optimizer,
		scaler: GradScaler,
		scheduler: torch.optim.lr_scheduler._LRScheduler,
		device: torch.device,
	) -> tuple[float, float, float]:
		model.train()
		total_loss = correct = total = 0
		t0 = time.time()

		for X_batch, y_batch in loader:
			X_batch = X_batch.to(device, dtype=torch.float32, non_blocking=True)
			y_batch = y_batch.to(device, non_blocking=True)

			optimizer.zero_grad(set_to_none=True)
			with autocast("cuda" if torch.cuda.is_available() else "cpu"):
				out  = model(X_batch)
				loss = criterion(out, y_batch)

			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()

			total_loss += loss.item() * len(y_batch)
			correct    += (out.argmax(1) == y_batch).sum().item()
			total      += len(y_batch)

		return total_loss / total, correct / total, time.time() - t0

	@torch.no_grad()
	def eval_epoch(
		self,
		model: nn.Module,
		loader: DataLoader,
		criterion: nn.Module,
		device: torch.device,
	) -> tuple[float, float]:
		model.eval()
		total_loss = correct = total = 0

		for X_batch, y_batch in loader:
			X_batch = X_batch.to(device, dtype=torch.float32, non_blocking=True)
			y_batch = y_batch.to(device, non_blocking=True)
			with autocast("cuda" if torch.cuda.is_available() else "cpu"):
				out  = model(X_batch)
				loss = criterion(out, y_batch)
			total_loss += loss.item() * len(y_batch)
			correct    += (out.argmax(1) == y_batch).sum().item()
			total      += len(y_batch)

		return total_loss / total, correct / total

	def make_tensor_datasets(
		self,
		x: np.ndarray,
		y: np.ndarray,
	) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
		self.parent.log_signal.emit("\n✂️  Разбиваю на train/val/test (70/10/20)...")
		idx = np.arange(len(y))

		idx_tr, idx_te, y_tr, _ = train_test_split(
			idx, y, test_size=0.20, random_state=42, stratify=y)
		idx_tr, idx_vl, y_tr, _ = train_test_split(
			idx_tr, y_tr, test_size=0.125, random_state=42, stratify=y_tr)

		def to_dataset(xi, yi):
			return TensorDataset(
				torch.from_numpy(x[xi]).permute(0, 2, 1),
				torch.from_numpy(y[yi]),
			)

		ds_tr = to_dataset(idx_tr, idx_tr)
		ds_vl = to_dataset(idx_vl, idx_vl)
		ds_te = to_dataset(idx_te, idx_te)
		self.parent.log_signal.emit(f" Train: {len(ds_tr):,}  Val: {len(ds_vl):,}  Test: {len(ds_te):,}")
		return ds_tr, ds_vl, ds_te