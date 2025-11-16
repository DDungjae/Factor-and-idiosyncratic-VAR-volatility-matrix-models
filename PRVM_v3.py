from numba import njit
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Global variables
alpha_u = 0.235
c0 = 4.0
std_y_bar_cache = {}

@njit
def g(x):
    return np.minimum(x, 1 - x)

@njit
def y_bar_ii(Y_i, k, K=19):
    total = 0.0
    for l in range(1, K):
        total += g(l / K) * (Y_i[l + k - 1])
    return total

@njit
def y_bar_ij(Y_i, Y_j, k, K=19):
    total = 0.0
    for l in range(1, K + 1):
        delta_i = Y_i[l + k - 2]
        delta_j = Y_j[l + k - 2]
        diff_g = g(l / K) - g((l - 1)/K)
        total += (diff_g ** 2) * delta_i * delta_j
    return total

@njit
def indicator_function(a, b):
    return 1 if a <= b else 0

@njit
def u_i_m(i, std_y_bar_ii_val, m=390):
    return c0 * std_y_bar_ii_val * (m ** (-alpha_u))

@njit
def calculate_std(arr):
    n = len(arr)
    if n <= 1:
        return 0.0
    mean = np.sum(arr) / n
    variance = np.sum((arr - mean) ** 2) / (n - 1)
    std = np.sqrt(variance)
    return std

@njit
def calculate_std_y_bar(Y_i, K, m=390):
    y_bar_ii_array = np.zeros(m - K + 1)
    for k in range(1, m - K + 2):
        y_bar_ii_array[k-1] = y_bar_ii(Y_i, k, K)
    y_bar_ii_array *= m**(1/4)
    std = calculate_std(y_bar_ii_array)
    return std

@njit
def prvm_estimator(Y_i, Y_j, i, j, K=19, m=390):
    psi_K = (1/12) * K
    total = 0.0
    
    # Calculate standard deviations
    std_y_bar_ii_val = calculate_std_y_bar(Y_i, K, m)
    std_y_bar_jj_val = calculate_std_y_bar(Y_j, K, m)
    
    # Calculate thresholds
    ui_m_i = u_i_m(i, std_y_bar_ii_val, m)
    ui_m_j = u_i_m(j, std_y_bar_jj_val, m)

    for k in range(1, m - K + 2):
        yb_ii = y_bar_ii(Y_i, k, K)
        yb_jj = y_bar_ii(Y_j, k, K)
        yb_ij = y_bar_ij(Y_i, Y_j, k, K)
        total += (yb_ii * yb_jj - 0.5 * yb_ij) * indicator_function(abs(yb_ii), ui_m_i) * indicator_function(abs(yb_jj), ui_m_j)

    return total / psi_K


def daily_volatility_parallel(log_returns, n_jobs=-1):
    # Convert DataFrame to numpy array
    log_returns_array = log_returns.values
    m, p = log_returns_array.shape
    K = 19
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(prvm_estimator)(log_returns_array[:, i], log_returns_array[:, j], i, j, K, m) 
        for i in range(p) for j in range(p)
    )
    
    # Reshape results into matrix
    volatility_matrix = np.zeros((p, p))
    for idx, value in enumerate(results):
        i = idx // p
        j = idx % p
        volatility_matrix[i, j] = value
    
    return pd.DataFrame(volatility_matrix)

# Main execution
if __name__ == "__main__":
    #day index: 251-1247
    for i in range(228, 1247):
        log_returns_d1 = pd.read_csv('log_return/d{}.csv'.format(i)).iloc[:, 1:]
        d1_volatility = daily_volatility_parallel(log_returns_d1, n_jobs=-1)  # 모든 CPU 사용
        d1_volatility.to_csv('daily_volatility_2016_2019/d{}.csv'.format(i), index=False)
        print(f"volatility calculated for day {i}")
