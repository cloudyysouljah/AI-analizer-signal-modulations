"""
RadioML Signal Modulation Classifier
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Архитектура: CNN (stem + 3 stage) → BiLSTM → FC → num_classes
Оптимизации:
  - Все данные грузятся в RAM как float32 за один проход
  - TensorDataset — нулевой overhead на __getitem__
  - AMP, cudnn.benchmark, TF32
  - Автоподбор batch_size под VRAM
  - OneCycleLR scheduler
  - Label smoothing 0.05

Запуск:
  python rain.py --data dataset.hdf5 --epochs 50
  python rain.py --data dataset.hdf5 --epochs 50 --snr_min -20
"""

import os
import gc
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import time

# ── GPU флаги ────────────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark        = True
torch.backends.cudnn.enabled          = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

SAMPLE_LEN = 1024

CLASSES = [
    "OOK",       "4ASK",      "8ASK",      "BPSK",
    "QPSK",      "8PSK",      "16PSK",     "32PSK",
    "16APSK",    "32APSK",    "64APSK",    "128APSK",
    "16QAM",     "32QAM",     "64QAM",     "128QAM",
    "256QAM",    "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC",
    "AM-DSB-SC", "FM",        "GMSK",      "OQPSK",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. ЗАГРУЗКА ДАТАСЕТА В RAM
# ─────────────────────────────────────────────────────────────────────────────

def load_all_into_ram(
    hdf5_path: str,
    snr_min: float = -20.0,
    max_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Загружает весь HDF5 в RAM за один проход.

    Параметры
    ----------
    hdf5_path   : путь к .hdf5 файлу
    snr_min     : отфильтровать примеры с SNR ниже порога
    max_samples : 0 = брать всё, иначе случайная выборка

    Возвращает
    ----------
    X       : np.ndarray (N, 2, 1024) float32
    y       : np.ndarray (N,)         int64
    classes : list[str]
    """
    print(f"📂 Открываю {hdf5_path} ...")
    with h5py.File(hdf5_path, "r") as f:
        n     = f["X"].shape[0]
        n_cls = f["Y"].shape[1]
        classes = CLASSES if n_cls == 24 else [f"mod_{i}" for i in range(n_cls)]
        print(f"   Всего записей : {n:,}  |  Классов: {n_cls}")

        print("   Читаю SNR...")
        snr  = f["Z"][:].ravel().astype(np.float32)
        mask = snr >= snr_min
        idx  = np.where(mask)[0]
        print(f"   После фильтрации SNR ≥ {snr_min} дБ : {len(idx):,} записей")

        if max_samples > 0 and len(idx) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(idx, size=max_samples, replace=False)
            idx.sort()
            print(f"   Ограничено до {max_samples:,} (--max_samples)")

        n_sel   = len(idx)
        ram_gb  = n_sel * 2 * SAMPLE_LEN * 4 / 1024 ** 3  # float32 = 4 байта
        print(f"   Загружаю {n_sel:,} записей (~{ram_gb:.1f} GB float32)")

        X = np.empty((n_sel, 2, SAMPLE_LEN), dtype=np.float16)
        y = np.empty((n_sel,),               dtype=np.int64)

        CHUNK = 50_000
        loaded = 0
        for start in range(0, n_sel, CHUNK):
            chunk_idx = idx[start : start + CHUNK]
            raw = f["X"][chunk_idx.tolist()]                     # (C, 1024, 2)
            X[loaded : loaded + len(chunk_idx)] = (
                raw.transpose(0, 2, 1).astype(np.float16)
            )
            y[loaded : loaded + len(chunk_idx)] = np.argmax(
                f["Y"][chunk_idx.tolist()], axis=1
            )
            loaded += len(chunk_idx)
            print(f"   {loaded:>8,} / {n_sel:,}  ({loaded / n_sel * 100:.1f}%)", end="\r")

    print(f"\n✅ Загружено: X={X.shape} dtype={X.dtype}, y={y.shape}")
    return X, y, classes


# ─────────────────────────────────────────────────────────────────────────────
# 2. ПОДГОТОВКА ТЕНЗОРОВ
# ─────────────────────────────────────────────────────────────────────────────

def make_tensor_datasets(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    numpy → TensorDataset, split 70/10/20.
    Возвращает (ds_train, ds_val, ds_test).
    """
    print("\n✂️  Разбиваю на train/val/test (70/10/20)...")
    idx = np.arange(len(y))

    idx_tr, idx_te, y_tr, _ = train_test_split(
        idx, y, test_size=0.20, random_state=42, stratify=y)
    idx_tr, idx_vl, y_tr, _ = train_test_split(
        idx_tr, y_tr, test_size=0.125, random_state=42, stratify=y_tr)
    # 0.125 от 0.80 = 0.10 от полного

    def to_dataset(xi, yi):
        return TensorDataset(
            torch.from_numpy(X[xi]),
            torch.from_numpy(y[yi]),
        )

    ds_tr = to_dataset(idx_tr, idx_tr)
    ds_vl = to_dataset(idx_vl, idx_vl)
    ds_te = to_dataset(idx_te, idx_te)
    print(f"   Train: {len(ds_tr):,}  Val: {len(ds_vl):,}  Test: {len(ds_te):,}")
    return ds_tr, ds_vl, ds_te


# ─────────────────────────────────────────────────────────────────────────────
# 3. АРХИТЕКТУРА МОДЕЛИ
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Два Conv1d с residual-связью и GELU."""

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
    """
    Вход  : (B, 2, 1024)
    Выход : (B, num_classes)

    Схема : stem → stage1 → stage2 → stage3 → BiLSTM → head
    После трёх MaxPool1d(2): (B, 256, 64)
    После BiLSTM(256, bidirectional=True): (B, 64, 512)
    Берём последний шаг LSTM: (B, 512) → head → (B, num_classes)
    """

    def __init__(self, num_classes: int, dropout: float = 0.4):
        super().__init__()

        # ── CNN backbone ──────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),                                      # → (B, 32, 512)
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(32), ResidualBlock(32),
            nn.Conv1d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),                                      # → (B, 64, 256)
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64), ResidualBlock(64),
            nn.Conv1d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),                                      # → (B, 128, 128)
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128), ResidualBlock(128),
            nn.Conv1d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2),                                      # → (B, 256, 64)
        )

        # ── BiLSTM ────────────────────────────────────────────────────────────
        # Вход: (B, 64, 256) после permute → (B, seq=64, features=256)
        # Выход последнего шага: (B, 512)  — hidden_size * 2 (bidirectional)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        # ── Классификационная голова ──────────────────────────────────────────
        # Входная размерность = 256 * 2 = 512 (bidirectional)
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
        x = self.stem(x)                       # (B, 32,  512)
        x = self.stage1(x)                     # (B, 64,  256)
        x = self.stage2(x)                     # (B, 128, 128)
        x = self.stage3(x)                     # (B, 256,  64)
        x, _ = self.lstm(x.permute(0, 2, 1))  # (B, 64, 512)
        return self.head(x[:, -1, :])          # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# 4. АВТОПОДБОР BATCH SIZE
# ─────────────────────────────────────────────────────────────────────────────

def find_max_batch_size(model, device, start=128, max_bs=4096):
    if device.type != "cuda":
        return start

    print("🔍 Автоподбор batch_size под VRAM...")
    scaler = GradScaler()
    lo, hi, best = start, max_bs, start

    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            torch.cuda.empty_cache()
            gc.collect()
            x_tmp = torch.randn(mid, 2, SAMPLE_LEN, device=device)
            y_tmp = torch.zeros(mid, dtype=torch.long, device=device)
            # autocast только если cuda
            with autocast("cuda"):
                loss = nn.CrossEntropyLoss()(model(x_tmp), y_tmp)
            scaler.scale(loss).backward()
            model.zero_grad(set_to_none=True)
            best = mid
            lo   = mid + 1
            print(f"   {mid:5d} ✅")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                hi = mid - 1
                print(f"   {mid:5d} ❌ OOM")
            else:
                # другая ошибка — пробрасываем
                raise
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    safe = 2 ** int(np.log2(max(start, int(best * 0.85))))
    print(f"✅ batch_size = {safe}  (max без OOM: {best})\n")
    return safe


# ─────────────────────────────────────────────────────────────────────────────
# 5. ОБУЧЕНИЕ
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
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
    for i, (X_batch, y_batch) in enumerate(loader):
        t1 = time.time()
        # t_load = time.time() - t0  # время загрузки батча из RAM

        # t1 = time.time()
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
    t_train = time.time() - t0
    # print(f"{(t_train):.2f} sec")
    return total_loss / total, correct / total , t_train


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = correct = total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, dtype=torch.float16, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        with autocast("cuda" if torch.cuda.is_available() else "cpu"):
            out  = model(X_batch)
            loss = criterion(out, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct    += (out.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# 6. ВИЗУАЛИЗАЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(
    history: dict,
    save_path: str = "training_history.png",
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"],   label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"],   label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 График сохранён: {save_path}")


def plot_confusion(
    model: nn.Module,
    loader: DataLoader,
    classes: list[str],
    device: torch.device,
    save_path: str = "confusion_matrix.png",
) -> None:
    model.eval()
    all_pred, all_true = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                pred = model(X_batch.to(device, dtype=torch.float16, non_blocking=True)).argmax(1).cpu()
            all_pred.extend(pred.numpy())
            all_true.extend(y_batch.numpy())

    present_labels = sorted(set(all_true) | set(all_pred))
    present_names  = [classes[i] for i in present_labels]

    cm      = confusion_matrix(all_true, all_pred, labels=present_labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n = len(present_labels)
    plt.figure(figsize=(max(12, n), max(10, n - 2)))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        xticklabels=present_names,
        yticklabels=present_names,
        cmap="Blues",
        linewidths=0.3,
    )
    plt.title("Confusion Matrix (нормализованная)")
    plt.ylabel("Истинный класс")
    plt.xlabel("Предсказанный класс")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 Confusion matrix сохранена: {save_path}")

    print("\n📋 Classification Report:")
    print(classification_report(
        all_true, all_pred,
        labels=present_labels,
        target_names=present_names,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # В main(), перед циклом обучения
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    # ── Устройство ───────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(0)
        print(f"🚀 GPU: {props.name}  |  VRAM: {props.total_memory / 1024 ** 3:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Apple MPS")
    else:
        device = torch.device("cpu")
        print("⚠️  CPU — обучение будет медленным")

    # ── Загрузка датасета ────────────────────────────────────────────────────
    X, y, classes = load_all_into_ram(
        args.data,
        snr_min=args.snr_min,
        max_samples=args.max_samples,
    )

    ds_tr, ds_vl, ds_te = make_tensor_datasets(X, y)
    del X, y
    gc.collect()
    print("🧹 Исходный numpy массив освобождён")

    # ── Модель ───────────────────────────────────────────────────────────────
    model  = RadioMLNet(num_classes=len(classes), dropout=args.dropout).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Параметров: {params:,}  |  Классов: {len(classes)}: {classes}")

    # ── Batch size ────────────────────────────────────────────────────────────
    batch_size = (
        find_max_batch_size(model, device)
        if args.batch_size == 0
        else args.batch_size
    )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    # На Windows многопроцессность медленнее для RAM-данных → num_workers=0
    if os.name == "nt":
        nw, pw, pf = 0, False, None
        print("💡 Windows: num_workers=0")
    else:
        nw, pw, pf = 4, True, 2
        print(f"💡 Linux: num_workers={nw}, persistent_workers=True")

    pm = device.type == "cuda"
    loader_kwargs = dict(
        num_workers=nw,
        pin_memory=pm,
        persistent_workers=pw,
        **({"prefetch_factor": pf} if pf else {}),
    )

    tr_loader = DataLoader(ds_tr, batch_size=batch_size,     shuffle=True,  **loader_kwargs)
    vl_loader = DataLoader(ds_vl, batch_size=batch_size * 2, shuffle=False, **loader_kwargs)
    te_loader = DataLoader(ds_te, batch_size=batch_size * 2, shuffle=False, **loader_kwargs)

    # ── Оптимизатор и расписание ──────────────────────────────────────────────
    # label_smoothing=0.05 — меньше чем 0.1, модель более уверена в предсказаниях
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(tr_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )
    scaler = GradScaler("cuda")

    # ── Цикл обучения ─────────────────────────────────────────────────────────
    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, train_time = train_epoch(
            model, tr_loader, criterion, optimizer, scaler, scheduler, device)
        vl_loss, vl_acc = eval_epoch(model, vl_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        vram_str = ""
        if device.type == "cuda":
            vram_str = f"  VRAM: {torch.cuda.memory_reserved(0) / 1024 ** 3:.1f} GB"

        saved = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "classes":     classes,
                    "val_acc":     vl_acc,
                    "snr_min":     args.snr_min,
                },
                args.save,
            )
            saved = "  💾 сохранено"

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
            f"Acc {tr_acc:.4f}/{vl_acc:.4f} | "
            f"Train time {train_time} sec"
            f"{vram_str}{saved}"
        )

    # ── Финальная оценка на тесте ─────────────────────────────────────────────
    print("\n🔄 Загружаю лучшую модель для финальной оценки...")
    ckpt = torch.load(args.save, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_acc = eval_epoch(model, te_loader, criterion, device)
    print(f"🏁 Test Accuracy: {test_acc:.4f}  |  Test Loss: {test_loss:.4f}")
    print(f"🏆 Лучший Val Accuracy: {best_val_acc:.4f}")

    plot_history(history)
    plot_confusion(model, te_loader, classes, device)


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RadioML Modulation Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", required=True,
        help="Путь к .hdf5 файлу датасета",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Количество эпох обучения",
    )
    parser.add_argument(
        "--batch_size", type=int, default=0,
        help="Размер батча (0 = автоподбор под GPU)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Максимальный learning rate для OneCycleLR",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.4,
        help="Dropout в LSTM и голове классификатора",
    )
    parser.add_argument(
        "--snr_min", type=float, default=-20.0,
        help="Минимальный SNR (дБ) для фильтрации датасета. "
             "-20 = брать все данные включая зашумлённые",
    )
    parser.add_argument(
        "--max_samples", type=int, default=0,
        help="Лимит записей в RAM (0 = без лимита)",
    )
    parser.add_argument(
        "--save", default="models/best_model.pt",
        help="Путь для сохранения лучшей модели",
    )
    main(parser.parse_args())