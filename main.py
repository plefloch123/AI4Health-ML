import numpy as np
import matplotlib.pyplot as plt

#Import data using np.loadtxt
clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')