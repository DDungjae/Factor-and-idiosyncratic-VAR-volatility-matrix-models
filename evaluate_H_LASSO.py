import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import warnings
from sklearn.linear_model import SGDRegressor
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

period_1 = range(751, 1248)
period_2 = range(751,999)
period_3 = range(999, 1248) 
gics_list = pd.read_csv('gicslist.csv').values[1:, 0]

"""# PSD cone projection function
def project_to_psd_cone(matrix):
  
    # Ensure matrix is symmetric    
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(matrix)
    
    # Project to PSD cone: set negative eigenvalues to zero
    eigvals_projected = np.maximum(eigvals, 0)
    
    # Reconstruct matrix
    matrix_projected = eigvecs @ np.diag(eigvals_projected) @ eigvecs.T
    
    return matrix_projected

# Truncation function
def truncation_function(x, omega):
    return np.where(np.abs(x) <= omega, x, np.sign(x) * omega)

# 디자인 생성 함수 (factor part)
def build(arr, h):
    dim, n = arr.shape
    X = np.column_stack([arr[:, h-k:-k].T for k in range(1,h+1)])
    Y = arr[:, h:].T
    return X, Y

# 매 날짜별로 Huber Lasso 모델 학습하는 함수
def train_huber_lasso_model(current_day, window_size=251):

    # 학습 구간 설정 (current_day - window_size ~ current_day - 1)
    start_day = max(251, current_day - window_size)
    end_day = current_day - 1
    
    # 이전 22일 동안의 평균 factor eigen vector 계산 (PSD cone projection 적용)
    matrix_sum = np.zeros((200, 200))
    count = 0
    for d in range(max(251, current_day-22), current_day):
        matrix = pd.read_csv(f'daily_volatility_2016_2019/d{d}.csv', header=None).values[1:, :]
        # PSD cone projection 적용
        matrix_projected = project_to_psd_cone(matrix)
        matrix_sum += matrix_projected
        count += 1
    avg_matrix = matrix_sum / count
    eigvals, eigvecs = np.linalg.eigh(avg_matrix)
    eigvals = np.sort(eigvals)[::-1]
    eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
    eigvals = eigvals[:3]
    eigvecs = eigvecs[:, :3]
    
    # 이전 22일 동안의 평균 idio eigen vector 계산
    idio_matrix_sum = np.zeros((200, 200))
    count = 0
    for d in range(max(251, current_day-22), current_day):
        idio_matrix = pd.read_csv(f'daily_idio_matrix/d{d}.csv', header=None).values[1:, :]
        idio_matrix_sum += idio_matrix
        count += 1
    avg_idio_matrix = idio_matrix_sum / count
    idio_eigvals, idio_eigvecs = np.linalg.eigh(avg_idio_matrix)
    idio_eigvals = np.sort(idio_eigvals)[::-1]
    idio_eigvecs = idio_eigvecs[:, np.argsort(idio_eigvals)[::-1]]
    
    # 251일 동안의 eigen value를 역산 (병렬 처리)
    xi_f = np.zeros((3, window_size))
    xi_i = np.zeros((200, window_size))
    
    def process_day_data(d, i):
        # d일의 volatility matrix 로드
        volatility_matrix = pd.read_csv(f'daily_volatility_2016_2019/d{d}.csv', header=None).values[1:, :]
        volatility_matrix = project_to_psd_cone(volatility_matrix)
        # factor eigen value 역산
        factor_eigvals = np.zeros(3)
        for j in range(3):
            eigvec_j = eigvecs[:, j]
            eigval_j = (eigvec_j.T @ volatility_matrix @ eigvec_j)/200
            factor_eigvals[j] = eigval_j
        factor_eigvals = np.sort(factor_eigvals)[::-1]
        # idio eigen value 역산
        idio_matrix = pd.read_csv(f'daily_idio_matrix/d{d}.csv', header=None).values[1:, :]
        idio_eigvals = np.zeros(200)
        for j in range(200):
            eigvec_j = idio_eigvecs[:, j]
            eigval_j = eigvec_j.T @ idio_matrix @ eigvec_j
            idio_eigvals[j] = eigval_j
        idio_eigvals = np.sort(idio_eigvals)[::-1]
        return factor_eigvals, idio_eigvals
    
    # 병렬로 데이터 처리
    n_jobs = min(multiprocessing.cpu_count(), 4)  # 데이터 로딩은 적은 코어 사용
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_day_data)(d, i) for i, d in enumerate(range(start_day, end_day + 1))
    )
    
    for i, (factor_eigvals, idio_eigvals) in enumerate(results):
        sorted_factor_eigvals = np.sort(factor_eigvals)[::-1]
        sorted_idio_eigvals = np.sort(idio_eigvals)[::-1]
        xi_f[:, i] = sorted_factor_eigvals
        xi_i[:, i] = sorted_idio_eigvals
    
    # Truncation parameters 계산
    r, n = xi_f.shape
    p = xi_i.shape[0]
    sigma_F = np.sqrt(np.sum(xi_f**2) / (n * r))
    cF1, cI1 = 4, 4
    cF2, cI2 = 0.25, 4
    omega_F = cF1 * sigma_F * (n / np.log(p))**(1/4)
    omega_I = cI1 * (n / np.log(p))**(1/4)
    tau_F = cF2 * sigma_F * (n / np.log(p))**(1/4)
    tau_I = cI2 * (n / np.log(p))**(1/4)
    
    # Truncation 적용
    xi_f_truncated = truncation_function(xi_f, omega_F)
    # 표준화 데이터 분리
    xi_i_mean = xi_i.mean(axis=1, keepdims=False)  # (200,)
    xi_i_std = xi_i.std(axis=1, ddof=1, keepdims=False)  # (200,)
    xi_i_std_data = (xi_i - xi_i_mean[:, None]) / xi_i_std[:, None]  # (200, window_size)
    xi_i_truncated = truncation_function(xi_i_std_data, omega_I)
    
    # factor 표준화 정보 계산
    xi_f_mean = xi_f.mean(axis=1, keepdims=False)  # (3,)
    xi_f_std = xi_f.std(axis=1, ddof=1, keepdims=False)  # (3,)
    
    # 학습 데이터 구성
    Xf, Yf = xi_f_truncated[:, :-1].T, xi_f[:, 1:].T  # factor: truncated input, original output
    Xi_lag = xi_i_truncated[:, :-1].T  # (n-1, 200) - truncated input
    Y_idio = xi_i_std_data[:, 1:].T   # (n-1, 200) - standardized original output (not truncated)
    
    # Factor: Huber regression (더 정확하게)
    
    factor_coef_matrix = np.zeros((r, Xf.shape[1]))
    factor_intercept_vector = np.zeros(r)
    for k in range(r):
        lr = SGDRegressor(
            fit_intercept=True,
            loss='huber',
            penalty=None,
            epsilon=tau_F,
            max_iter=1000,        # iteration 수 줄임
            tol=1e-4,             # tolerance 완화
            learning_rate='adaptive',
            eta0=1e-3,            # learning rate 증가
            early_stopping=True,
            verbose=0,
            random_state=42
        )
        lr.fit(Xf, Yf[:, k])
        factor_coef_matrix[k, :] = lr.coef_.flatten()
        factor_intercept_vector[k] = lr.intercept_.item()

    # Idio: SGDRegressor with Huber loss + L1 penalty (개별 c_eta 선택)
    # Calculate tuning parameters for idio part
    
    # Apply truncation to STANDARDIZED idio data
    # xi_i_std_truncated = truncation_function(xi_i_std_data, omega_I) # This line is removed as per the edit hint
    
    # Create combined input for idio models: factor + idio
    # Factor data (lagged) - standardize first, then truncate
    xi_f_std = (xi_f - xi_f.mean(axis=1, keepdims=True)) / xi_f.std(axis=1, ddof=1, keepdims=True)
    factor_mean = xi_f.mean(axis=1, keepdims=True).flatten()
    factor_std = xi_f.std(axis=1, ddof=1, keepdims=True).flatten()
    xi_f_std_truncated = truncation_function(xi_f_std, omega_I)
    
    Xf_lag = xi_f_std_truncated[:, :-1].T  # (n-1, 3) - standardized and truncated factor lag
    # Xi_lag = xi_i_std_truncated[:, :-1].T  # (n-1, 200) - standardized and truncated idio lag # This line is removed as per the edit hint
    
    # Combined input: [factor_lag, idio_lag] = (n-1, 203)
    X_combined = np.column_stack([Xf_lag, Xi_lag])  # (n-1, 203)
    # Y_idio = xi_i_std[:, 1:].T  # (n-1, 200) - standardized idio target, row format # This line is removed as per the edit hint
    
    # Select c_eta by minimizing BIC for each eigenvalue individually (병렬 처리)
    c_eta_grid = np.geomspace(0.01, 10, 20)  # cη ∈ [0.01, 10] - 더 줄임
    
    def optimize_c_eta_for_eigenvalue(j):
        best_bic, best_c_eta, best_alpha = 1e18, None, None
        best_coef = None
        best_intercept = None
        for c_eta in c_eta_grid:
            alpha = c_eta * np.sqrt(np.log(p) / n)
            mdl = SGDRegressor(
                loss='huber', 
                learning_rate='adaptive',  # adaptive learning rate 사용
                eta0=1e-3,  # 초기 학습률 증가
                epsilon=tau_I, 
                alpha=alpha, 
                penalty='l1', 
                fit_intercept=True, 
                max_iter=500,      # iteration 수 줄임
                tol=1e-4,          # tolerance 완화
                early_stopping=True,
                verbose=0,
                random_state=42
            )
            mdl.fit(X_combined, Y_idio[:, j])
            coef = mdl.coef_
            intercept = mdl.intercept_
            pred = mdl.predict(X_combined)
            RSS = np.sum((Y_idio[:, j] - pred)**2)
            df = np.sum(np.abs(coef) > 0.0)
            N = Y_idio.shape[0]
            bic = N * np.log(RSS/N) + df * np.log(N)
            if bic < best_bic:
                best_bic, best_c_eta, best_alpha = bic, c_eta, alpha
                best_coef = coef.copy()
                best_intercept = intercept
        return best_c_eta, best_alpha, best_coef, best_intercept
    
    # 병렬로 c_eta 최적화 실행
    n_jobs = min(multiprocessing.cpu_count(), 8)  # CPU 코어 수에 따라 조정
    results = Parallel(n_jobs=n_jobs)(
        delayed(optimize_c_eta_for_eigenvalue)(j) for j in range(p)
    )
    
    best_c_eta_per_eigenvalue = np.zeros(p)
    best_alpha_per_eigenvalue = np.zeros(p)
    best_coefM = np.zeros((X_combined.shape[1], p))
    best_interceptM = np.zeros(p)
    
    for j, (best_c_eta, best_alpha, best_coef, best_intercept) in enumerate(results):
        best_c_eta_per_eigenvalue[j] = best_c_eta
        best_alpha_per_eigenvalue[j] = best_alpha
        best_coefM[:, j] = best_coef
        best_interceptM[j] = best_intercept.item()
    
    idio_coef_matrix = best_coefM.T  # (200, 203)로 변환
    idio_intercept_vector = best_interceptM  # (200,)로 변환
    
    return factor_intercept_vector, factor_coef_matrix, idio_coef_matrix, idio_intercept_vector, omega_F, omega_I, eigvecs, idio_eigvecs, xi_i_mean, xi_i_std, factor_mean, factor_std

eigvals_factors = []
eigvals_idio = []
estimated_volatillity = []

def predict_single_day(i):
    # 매 날짜별로 이전 251일 데이터로 Huber Lasso 모델 재학습
    factor_intercepts, factor_coefs, idio_coefs, idio_intercepts, omega_F, omega_I, eigvecs, idio_eigvecs, xi_i_mean, xi_i_std, xi_f_mean, xi_f_std = train_huber_lasso_model(i)
    
    # i-1일의 volatility matrix로 eigen value 역산
    volatility_matrix = pd.read_csv(f'daily_volatility_2016_2019/d{i-1}.csv', header=None).values[1:, :]
    volatility_matrix = project_to_psd_cone(volatility_matrix)
    factor_eigvals = np.zeros(3)
    for j in range(3):
        eigvec_j = eigvecs[:, j]
        eigval_j = (eigvec_j.T @ volatility_matrix @ eigvec_j) / 200
        factor_eigvals[j] = eigval_j
    factor_eigvals = np.sort(factor_eigvals)[::-1]
    # i-1일의 idio eigen values 역산
    idio_matrix = pd.read_csv(f'daily_idio_matrix/d{i-1}.csv', header=None).values[1:, :]
    idio_eigvals = np.zeros(200)
    for j in range(200):
        eigvec_j = idio_eigvecs[:, j]
        eigval_j = eigvec_j.T @ idio_matrix @ eigvec_j
        idio_eigvals[j] = eigval_j
    idio_eigvals = np.sort(idio_eigvals)[::-1]
    # Apply truncation to factor data
    factor_eigvals_truncated = truncation_function(factor_eigvals, omega_F)
    # Apply standardization and truncation to idio data
    idio_eigvals_std = (idio_eigvals - xi_i_mean) / xi_i_std
    idio_eigvals_truncated = truncation_function(idio_eigvals_std, omega_I)
    
    # Predict next day's factor eigenvalues (non-standardized data 사용)
    factor_input = factor_eigvals_truncated.reshape(1, -1)
    predicted_factor_eigvals = factor_input @ factor_coefs.T + factor_intercepts
    predicted_factor_eigvals = np.sort(predicted_factor_eigvals)[::-1]
    # Predict next day's idio eigenvalues (표준화된 데이터 사용)
    # Create combined input for prediction: [factor_lag, idio_lag]
    factor_input_std = (factor_eigvals - xi_f_mean) / xi_f_std
    factor_input_std_truncated = truncation_function(factor_input_std, omega_I)
    
    combined_input = np.concatenate([factor_input_std_truncated, idio_eigvals_truncated]).reshape(1, -1)
    predicted_idio_eigvals_std = combined_input @ idio_coefs.T + idio_intercepts  # (1, 200) - standardized prediction with intercept
    
    # Convert back to original scale
    predicted_idio_eigvals = predicted_idio_eigvals_std * xi_i_std + xi_i_mean
    predicted_factor_eigvals = np.clip(predicted_factor_eigvals, 0, None)
    predicted_idio_eigvals = np.clip(predicted_idio_eigvals, 0, None)
    
    # Construct estimated volatility matrix
    factor_matrix = 200*(eigvecs @ np.diag(predicted_factor_eigvals.flatten()) @ eigvecs.T)
    idio_matrix = idio_eigvecs @ np.diag(predicted_idio_eigvals.flatten()) @ idio_eigvecs.T
    estimated_matrix = factor_matrix + idio_matrix

    # Save estimated volatility matrix
    pd.DataFrame(estimated_matrix).to_csv(f'estimated_matrix/d{i}.csv', index=False)
    
    return f'd{i} done'

print("Starting rolling window Huber Lasso prediction...")
# 병렬로 예측 실행 (메모리 사용량 고려하여 적은 코어 사용)
n_jobs = min(multiprocessing.cpu_count(), 2)  # 예측은 메모리 집약적이므로 적은 코어 사용
results = Parallel(n_jobs=n_jobs)(
    delayed(predict_single_day)(i) for i in tqdm(period_1, desc="Predicting")
)

for result in results:
    print(result)
"""
# Calculate MSPE
MSPE_period_1_list = []
MSPE_period_2_list = []
MSPE_period_3_list = []

print("Calculating MSPE...")
for i in period_1:
    if i == 1247:
        continue
    estimated_matrix = pd.read_csv(f'estimated_matrix/d{i}.csv').values[:, :]
    # 실제 volatility matrix와 비교 (POET-PRVM이 아닌 원본)
    real_matrix = pd.read_csv(f'estimated_matrix_POET_PRVM/d{i}.csv', header=None).values[:, :]
    diff = estimated_matrix - real_matrix
    mspe = np.linalg.norm(diff, ord='fro')**2
    MSPE_period_1_list.append(mspe)
    if i in period_2:
        MSPE_period_2_list.append(mspe)
    if i in period_3:
        MSPE_period_3_list.append(mspe)
    # print(f"Day {i}: MSPE = {mspe:.6f}")

MSPE_period_1 = np.mean(MSPE_period_1_list)
MSPE_period_2 = np.mean(MSPE_period_2_list)
MSPE_period_3 = np.mean(MSPE_period_3_list)

print("\n" + "="*50)
print("MSPE RESULTS (scaled by 10^4)")
print("="*50)
print(f"MSPE_period_1 (751-1247): {MSPE_period_1*(10**4):.4f}")
print(f"MSPE_period_2 (751-998):  {MSPE_period_2*(10**4):.4f}")
print(f"MSPE_period_3 (999-1247): {MSPE_period_3*(10**4):.4f}")
print("="*50)

# Additional statistics
print(f"\nNumber of days in period 1: {len(MSPE_period_1_list)}")
print(f"Number of days in period 2: {len(MSPE_period_2_list)}")
print(f"Number of days in period 3: {len(MSPE_period_3_list)}")
print(f"MSPE std in period 1: {np.std(MSPE_period_1_list)*(10**4):.4f}")
print(f"MSPE std in period 2: {np.std(MSPE_period_2_list)*(10**4):.4f}")
print(f"MSPE std in period 3: {np.std(MSPE_period_3_list)*(10**4):.4f}")

import matplotlib.pyplot as plt

# dates = list(period_1)
valid_dates = [i for i, mspe in zip(period_1, MSPE_period_1_list) if mspe is not None]

plt.figure(figsize=(14, 6))
plt.plot(valid_dates, MSPE_period_1_list, label='MSPE (All)', color='blue')
plt.axvline(x=min(period_3), color='red', linestyle='--', label='Period 3 Start (999)')
plt.title('MSPE by Day- H-LASSO')
plt.xlabel('Day (d)')
plt.ylabel('MSPE')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('h_lasso_mspe_by_day.png', dpi=300)
plt.show()

print("MSPE plot saved as 'h_lasso_mspe_by_day.png'") 

