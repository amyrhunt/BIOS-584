import numpy as np
import scipy.io as sio
from self_py_fun.HW8Fun import produce_trun_mean_cov, plot_trunc_mean, plot_trunc_cov

mat_file_path = '/Users/amyhunt/Documents/GitHub/BIOS-584/data/K114_001_BCI_TRN_Truncated_Data_0.5_6.mat'
mat_data = sio.loadmat(mat_file_path)

data = mat_data['Signal'][:, :, np.newaxis]
target_labels = mat_data['Type'].flatten()

unique_types = np.unique(target_labels)
if len(unique_types) == 2:
    target_labels = (target_labels == unique_types[1]).astype(int)
else:
    target_labels = (target_labels == unique_types[0]).astype(int)

max_time = 100

results = produce_trun_mean_cov(data, target_labels, max_time)
plot_trunc_mean(results)
plot_trunc_cov(results)
