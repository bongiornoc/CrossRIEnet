import numpy as np
import scipy as sp
import tensorflow as tf


def sqrtm_psd(A, min_eig=1e-10):
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0.0)
    mask = eigvals > min_eig
    L = eigvecs[:, mask] * np.sqrt(eigvals[mask])
    return L


def dynamic_matrix_generator(
    batch_size=16, N_range=(20, 50), M_range=(20, 50), ndays_range=(200, 600)
):
    """Generates a batch of correctly shaped, dynamic data tensors."""
    while True:
        # Variable Dimensions
        N = np.random.randint(N_range[0], N_range[1] + 1)
        M = np.random.randint(M_range[0], M_range[1] + 1)
        ndays = np.random.randint(ndays_range[0], ndays_range[1] + 1)
        df = np.random.uniform(3.1, 4.2, size=2)

        X = np.random.standard_t(df=df[0], size=(batch_size, N, ndays))
        Y = np.random.standard_t(df=df[1], size=(batch_size, M, ndays))

        zX = sp.stats.zscore(X, axis=-1)
        zY = sp.stats.zscore(Y, axis=-1)

        cX = zX @ zX.transpose(0, 2, 1) / ndays
        cY = zY @ zY.transpose(0, 2, 1) / ndays

        L_lower_batchX = np.array([sqrtm_psd(cX[i]) for i in range(batch_size)])
        L_lower_batchY = np.array([sqrtm_psd(cY[i]) for i in range(batch_size)])

        rX = L_lower_batchX.shape[2]
        rY = L_lower_batchY.shape[2]

        X_batch = np.random.normal(size=(batch_size, rX, ndays))
        Y_batch = np.random.normal(size=(batch_size, rY, ndays))

        X_N = L_lower_batchX @ X_batch
        X_M = L_lower_batchY @ Y_batch

        zX_sample = sp.stats.zscore(X_N, axis=-1)
        zY_sample = sp.stats.zscore(X_M, axis=-1)

        Cxx = zX_sample @ zX_sample.transpose(0, 2, 1) / ndays
        Cyy = zY_sample @ zY_sample.transpose(0, 2, 1) / ndays

        Cxy_clean = zX @ zY.transpose(0, 2, 1) / ndays
        Cxy_noisy = zX_sample @ zY_sample.transpose(0, 2, 1) / ndays
        T_samples = tf.constant([float(ndays)] * batch_size)

        yield (Cxx, Cyy, Cxy_noisy, T_samples), Cxy_clean


def get_dynamic_dataset(
    batch_size=16, N_range=(20, 50), M_range=(20, 50), ndays_range=(200, 600)
):
    """Builds a tensorflow dataset out of the dynamic data pipeline."""
    return tf.data.Dataset.from_generator(
        lambda: dynamic_matrix_generator(batch_size, N_range, M_range, ndays_range),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),  # Cxx
                tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),  # Cyy
                tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),  # Cxy_noisy
                tf.TensorSpec(shape=(None,), dtype=tf.float32),  # T_samples
            ),
            tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),  # Cxy_clean
        ),
    )
