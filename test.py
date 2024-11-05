import numpy as np

d = np.load("/home/rayliu/get_a_grip/data/dataset/leap_test1/evalegrasp_config_dicts/core-bottle-10dff3c43200a7a7119862dbccbaa609_0_0809.npy", allow_pickle=True).item()

print(d.keys())
print(len(d['trans']))