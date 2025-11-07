import os
import numpy as np
import matplotlib.pyplot as plt

def produce_trun_mean_cov(data, target_labels, max_time=100):
    data_trunc = data[:, :max_time, :]
    target_data = data_trunc[target_labels == 1]
    non_target_data = data_trunc[target_labels == 0]

    target_mean = np.mean(target_data, axis=0)
    non_target_mean = np.mean(non_target_data, axis=0)
    all_mean = np.mean(data_trunc, axis=0)

    target_flat = target_data.reshape(-1, target_data.shape[2])
    non_target_flat = non_target_data.reshape(-1, non_target_data.shape[2])
    all_flat = data_trunc.reshape(-1, data_trunc.shape[2])

    target_cov = np.cov(target_flat.T)
    non_target_cov = np.cov(non_target_flat.T)
    all_cov = np.cov(all_flat.T)

    return {
        'target_mean': target_mean,
        'non_target_mean': non_target_mean,
        'all_mean': all_mean,
        'target_cov': target_cov,
        'non_target_cov': non_target_cov,
        'all_cov': all_cov
    }

def plot_trunc_mean(results, save_dir='K114'):
    os.makedirs(save_dir, exist_ok=True)

    target_mean = results['target_mean']
    non_target_mean = results['non_target_mean']

    target_avg = np.mean(target_mean, axis=1)
    non_target_avg = np.mean(non_target_mean, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(target_avg, label='Target', linewidth=2)
    plt.plot(non_target_avg, label='Non-Target', linewidth=2)
    plt.xlabel('Time Points')
    plt.ylabel('Average Signal')
    plt.title('Truncated Mean Signals: Target vs Non-Target')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, 'Mean.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_trunc_cov(results, save_dir='K114'):
    os.makedirs(save_dir, exist_ok=True)

    cov_types = [
        ('target_cov', 'Target', 'Covariance_Target.png'),
        ('non_target_cov', 'Non-Target', 'Covariance_Non-Target.png'),
        ('all_cov', 'All', 'Covariance_All.png')
    ]

    for cov_key, title, filename in cov_types:
        cov_matrix = results[cov_key]
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[cov_matrix]])

        plt.figure(figsize=(8, 6))
        plt.imshow(cov_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='Covariance')
        plt.xlabel('Channel')
        plt.ylabel('Channel')
        plt.title(f'Covariance Matrix: {title}')

        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
