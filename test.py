import h5py
import numpy as np

with h5py.File('/home/leonid/Загрузки/archive (2)/GOLD_XYZ_OSC.0001_1024.hdf5', 'r') as f:
    X = f['X'][:]  # все сигналы — может быть тяжело, лучше срез
    Y = f['Y'][:]
    Z = f['Z'][:]

print(f"Сигналов всего: {X.shape[0]}")
print(f"Длина одного сигнала: {X.shape[1]} отсчётов")
print(f"Каналы IQ: {X.shape[2]}")
print(f"Уникальные SNR: {np.unique(Z)}")
print(f"Число классов: {Y.shape[1]}")