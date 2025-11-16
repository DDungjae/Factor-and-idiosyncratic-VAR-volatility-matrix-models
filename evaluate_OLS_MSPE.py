import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from eigen_value_decomposition import project_to_psd_cone
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 평가 구간 설정
period_1 = range(751, 1248)
period_2 = range(751,999)
period_3 = range(999, 1248)
"""
# Truncation function
def truncation_function(x, omega):
    return np.where(np.abs(x) <= omega, x, np.sign(x) * omega)

# 디자인 생성 함수 (factor part)
def build(arr, h):
    dim, n = arr.shape
    X = np.column_stack([arr[:, h-k:-k].T for k in range(1,h+1)])
    Y = arr[:, h:].T
    return X, Y

# 매 날짜별로 OLS 모델 학습하는 함수
def train_ols_model(current_day, window_size=251):

    # 학습 구간 설정 (current_day - window_size ~ current_day - 1)
    start_day = max(251, current_day - window_size)
    end_day = current_day - 1
    
    # 이전 22일 동안의 평균 eigen vector 계산 (미리 계산된 것 사용)
    eigvecs = pd.read_csv(f'factor_eigen_vectors/d{current_day-1}.csv', header=None).values[1:, :]
    
    # 251일 동안의 eigen value를 역산 (병렬 처리)
    xi_f = np.zeros((3, window_size))
    
    def process_day_data(d, i):
        # d일의 volatility matrix 로드
        volatility_matrix = pd.read_csv(f'daily_volatility_2016_2019/d{d}.csv', header=None).values[1:, :]
        eigval, eigvec = project_to_psd_cone(volatility_matrix)
        volatility_matrix_clipped = eigvec @ np.diag(eigval) @ eigvec.T 
        # avg_eigen_vector를 사용해서 eigen value 역산
        factor_eigvals = np.zeros(3)
        for j in range(3):
            eigvec_j = eigvecs[:, j]
            # Rayleigh quotient로 eigen value 계산
            eigval_j = (eigvec_j.T @ volatility_matrix_clipped @ eigvec_j) / 200
            factor_eigvals[j] = eigval_j
        factor_eigvals = np.sort(factor_eigvals)[::-1]
        return factor_eigvals
    
    # 병렬로 데이터 처리
    n_jobs = multiprocessing.cpu_count()  # 풀코어 사용
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_day_data)(d, i) for i, d in enumerate(range(start_day, end_day + 1))
    )
    
    for i, factor_eigvals in enumerate(results):
        xi_f[:, i] = factor_eigvals
    
    # 디버깅: eigenvalue 통계 출력
    if current_day in [751, 800, 900, 1000, 1100, 1200]:
        print(f"\nDay {current_day} eigenvalue statistics:")
        print(f"  Eigenvalue range: [{xi_f.min():.6f}, {xi_f.max():.6f}]")
        print(f"  Eigenvalue mean: {xi_f.mean():.6f}")
        print(f"  Eigenvalue std: {xi_f.std():.6f}")
        print(f"  Negative eigenvalues: {np.sum(xi_f < 0)}")
    
    # Truncation parameters 계산
    r, n = xi_f.shape
    p = 200
    sigma_F = np.sqrt(np.sum(xi_f**2) / (n * r))
    cF1 = 4
    omega_F = cF1 * sigma_F * (n / np.log(p))**(1/4)
    
    # Truncation 적용
    # xi_f_truncated = truncation_function(xi_f, omega_F)
    
    # 학습 데이터 구성
    Xf, Yf = xi_f[:, :-1].T, xi_f[:, 1:].T
    
    # OLS 모델 학습
    factor_coef_matrix = np.zeros((r, Xf.shape[1]))
    factor_intercept_vector = np.zeros(r)
    
    for k in range(r):
        ols = LinearRegression(fit_intercept=True)
        ols.fit(Xf, Yf[:, k])
        factor_coef_matrix[k, :] = ols.coef_.flatten()
        factor_intercept_vector[k] = ols.intercept_.item()
    
    return factor_intercept_vector, factor_coef_matrix, omega_F, eigvecs

def predict_single_day_ols(i):
    # 매 날짜별로 이전 251일 데이터로 OLS 모델 재학습
    factor_intercepts, factor_coefs, omega_F, eigvecs = train_ols_model(i)
    
    # i-1일의 volatility matrix로 eigen value 역산
    volatility_matrix = pd.read_csv(f'daily_volatility_2016_2019/d{i-1}.csv', header=None).values[1:, :]
    eigval, eigvec = project_to_psd_cone(volatility_matrix)
    volatility_matrix_clipped = eigvec @ np.diag(eigval) @ eigvec.T
    factor_eigvals = np.zeros(3)
    for j in range(3):
        eigvec_j = eigvecs[:, j]
        eigval_j = (eigvec_j.T @ volatility_matrix_clipped @ eigvec_j) / 200
        factor_eigvals[j] = eigval_j
    
    # --- Factor: OLS로 예측 ---
    # factor_input_for_factor = truncation_function(factor_eigvals, omega_F).reshape(1, -1)
    factor_input_for_factor = factor_eigvals.reshape(1, -1)
    predicted_factor_eigvals = factor_input_for_factor @ factor_coefs.T + factor_intercepts
    # 예측된 eigen value로 factor matrix 구성
    factor_matrix = np.zeros((200, 200))
    predicted_factor_eigvals = predicted_factor_eigvals.flatten()
        
    # 디버깅: 예측 품질 확인
    if i in [751, 800, 900, 1000, 1100, 1200]:
        print(f"\nDay {i} prediction quality:")
        print(f"  Input eigenvalues: {factor_eigvals}")
        print(f"  Predicted eigenvalues: {predicted_factor_eigvals}")
        print(f"  Prediction error: {np.abs(predicted_factor_eigvals - factor_eigvals)}")
    
    for j in range(len(predicted_factor_eigvals)):
        eigval_i = predicted_factor_eigvals[j] * 200
        eigvec_i = eigvecs[:, j]  # train_ols_model과 동일한 eigen vector 사용
        factor_matrix += eigval_i * np.outer(eigvec_i, eigvec_i)
    
    # --- Idio: 이전 22일의 idio matrix 평균 사용 (병렬 처리) ---
    def load_idio_matrix(d):
        return pd.read_csv(f'daily_idio_matrix/d{d}.csv', header=None).iloc[1:, :].values
    
    # 병렬로 idio matrix 로드
    n_jobs = multiprocessing.cpu_count()
    idio_matrices = Parallel(n_jobs=n_jobs)(
        delayed(load_idio_matrix)(d) for d in range(max(251, i-22), i)
    )
    predicted_idio_matrix = np.mean(idio_matrices, axis=0)
        
    # Factor matrix는 이미 위에서 구성됨, idio와 합치기
    estimated_matrix = factor_matrix + predicted_idio_matrix

    # 디버깅: 최종 matrix 속성 확인
    if i in [751, 800, 900, 1000, 1100, 1200]:
        print(f"  Final matrix range: [{estimated_matrix.min():.6f}, {estimated_matrix.max():.6f}]")
        print(f"  Negative elements: {np.sum(estimated_matrix < 0)}")
        print(f"  Matrix norm: {np.linalg.norm(estimated_matrix, 'fro'):.6f}")

    # Save estimated volatility matrix
    pd.DataFrame(estimated_matrix).to_csv(f'estimated_matrix_OLS/d{i}.csv', index=False)
    
    return f'd{i} done'

eigvals_factors = []
eigvals_idio = []
estimated_volatillity = []

print("Starting rolling window OLS prediction...")
# 병렬로 예측 실행
n_jobs = multiprocessing.cpu_count()  # 메모리 고려하여 적은 코어 사용
results = Parallel(n_jobs=n_jobs)(
    delayed(predict_single_day_ols)(i) for i in tqdm(period_1, desc="Predicting")
)

for result in results:
    print(result)
"""
def qlike(M1, M2):
    return np.log(np.linalg.det(M1))+np.linalg.trace(np.linalg.inv(M1)@M2)

# Calculate MSPE (병렬 처리)
def calculate_mspe_for_day(i):
    estimated_matrix = pd.read_csv(f'estimated_matrix_OLS/d{i}.csv', header=None).values[1:, :]
    if i == 1247:
        return None
    else:
        real_matrix = pd.read_csv(f'estimated_matrix_POET_PRVM/d{i}.csv', header=None).values[:, :]
    
    # 디버깅: matrix 크기와 값 범위 확인
    if i in [751, 998, 999, 1246]:  # 각 period의 시작과 끝 확인
        print(f"Day {i}:")
        print(f"  Estimated matrix shape: {estimated_matrix.shape}")
        print(f"  Real matrix shape: {real_matrix.shape}")
        print(f"  Estimated matrix range: [{estimated_matrix.min():.6f}, {estimated_matrix.max():.6f}]")
        print(f"  Real matrix range: [{real_matrix.min():.6f}, {real_matrix.max():.6f}]")
    
    diff = estimated_matrix - real_matrix
    mspe = np.linalg.norm(diff, ord='fro')**2
    return mspe

print("Calculating MSPE...")
n_jobs = multiprocessing.cpu_count()
mspe_results = Parallel(n_jobs=n_jobs)(
    delayed(calculate_mspe_for_day)(i) for i in period_1
)

MSPE_period_1_list = [mspe for mspe in mspe_results if mspe is not None]
MSPE_period_2_list = [mspe for i, mspe in zip(period_1, mspe_results) if mspe is not None and i in period_2]
MSPE_period_3_list = [mspe for i, mspe in zip(period_1, mspe_results) if mspe is not None and i in period_3]

# 디버깅: 각 period의 MSPE 값들 확인
print(f"\nPeriod 2 MSPE range: [{min(MSPE_period_2_list):.6f}, {max(MSPE_period_2_list):.6f}]")
print(f"Period 3 MSPE range: [{min(MSPE_period_3_list):.6f}, {max(MSPE_period_3_list):.6f}]")
print(f"Period 2 MSPE first 5: {MSPE_period_2_list[:5]}")
print(f"Period 3 MSPE first 5: {MSPE_period_3_list[:5]}")

MSPE_period_1 = np.mean(MSPE_period_1_list)
MSPE_period_2 = np.mean(MSPE_period_2_list)
MSPE_period_3 = np.mean(MSPE_period_3_list)

print("\n" + "="*50)
print("OLS MSPE RESULTS (scaled by 10^4)")
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

# ================= QLIKE 계산 및 출력 =================
def calculate_qlike_for_day(i):
    estimated_matrix = pd.read_csv(f'estimated_matrix_OLS/d{i}.csv', header=None).values[1:, :]
    if i == 1247:
        return None
    else:
        real_matrix = pd.read_csv(f'estimated_matrix_POET_PRVM/d{i}.csv', header=None).values[:, :]
    try:
        estimated_matrix = estimated_matrix + 0.1 * np.eye(200)
        qlike_val = qlike(estimated_matrix, real_matrix)
    except Exception as e:
        print(f"QLIKE error at day {i}: {e}")
        qlike_val = np.nan
    return qlike_val

print("\nCalculating QLIKE...")
qlike_results = Parallel(n_jobs=n_jobs)(
    delayed(calculate_qlike_for_day)(i) for i in period_1
)

QLIKE_period_1_list = [q for q in qlike_results if q is not None and not np.isnan(q)]
QLIKE_period_2_list = [q for i, q in zip(period_1, qlike_results) if q is not None and not np.isnan(q) and i in period_2]
QLIKE_period_3_list = [q for i, q in zip(period_1, qlike_results) if q is not None and not np.isnan(q) and i in period_3]

print(f"\nPeriod 2 QLIKE range: [{min(QLIKE_period_2_list):.6f}, {max(QLIKE_period_2_list):.6f}]")
print(f"Period 3 QLIKE range: [{min(QLIKE_period_3_list):.6f}, {max(QLIKE_period_3_list):.6f}")
print(f"Period 2 QLIKE first 5: {QLIKE_period_2_list[:5]}")
print(f"Period 3 QLIKE first 5: {QLIKE_period_3_list[:5]}")

QLIKE_period_1 = np.mean(QLIKE_period_1_list)
QLIKE_period_2 = np.mean(QLIKE_period_2_list)
QLIKE_period_3 = np.mean(QLIKE_period_3_list)

print("\n" + "="*50)
print("OLS QLIKE RESULTS")
print("="*50)
print(f"QLIKE_period_1 (751-1247): {QLIKE_period_1:.4f}")
print(f"QLIKE_period_2 (751-998):  {QLIKE_period_2:.4f}")
print(f"QLIKE_period_3 (999-1247): {QLIKE_period_3:.4f}")
print("="*50)

# Additional statistics
print(f"\nNumber of days in period 1: {len(QLIKE_period_1_list)}")
print(f"Number of days in period 2: {len(QLIKE_period_2_list)}")
print(f"Number of days in period 3: {len(QLIKE_period_3_list)}")
print(f"QLIKE std in period 1: {np.std(QLIKE_period_1_list):.4f}")
print(f"QLIKE std in period 2: {np.std(QLIKE_period_2_list):.4f}")
print(f"QLIKE std in period 3: {np.std(QLIKE_period_3_list):.4f}") 

import matplotlib.pyplot as plt

# dates = list(period_1)
valid_dates = [i for i, mspe in zip(period_1, mspe_results) if mspe is not None]

plt.figure(figsize=(14, 6))
plt.plot(valid_dates, MSPE_period_1_list, label='MSPE (All)', color='blue')
plt.axvline(x=min(period_3), color='red', linestyle='--', label='Period 3 Start (999)')
plt.title('MSPE by Day- OLS')
plt.xlabel('Day (d)')
plt.ylabel('MSPE')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ols_mspe_by_day.png', dpi=300)
plt.show()

print("MSPE plot saved as 'ols_mspe_by_day.png'") 