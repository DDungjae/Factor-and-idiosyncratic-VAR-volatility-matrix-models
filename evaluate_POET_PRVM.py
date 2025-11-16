import numpy as np, pandas as pd, os

path = "daily_volatility_2016_2019"   # PRVM 저장 폴더
output_path = "estimated_matrix_POET_PRVM"  # POET-PRVM 결과 저장 폴더
period_1 = range(751, 1248)
period_2 = range(751,999)
period_3 = range(999, 1248)              # 997→ 251~1247 (예측은 252~1248)
r     = 3                             # factor rank
tau   = 3e-4                          # soft-threshold 수준 (논문 값)
gicslist = pd.read_csv('gicslist.csv').values
print(gicslist.shape)
# 출력 폴더 생성
os.makedirs(output_path, exist_ok=True)


def poet_prvm(G, r=3, tau=3e-4, gicslist=gicslist):
    """
    G : (p,p)  jump-adjusted PRVM (PSD)
    returns : (p,p)  POET-threshold 행렬
    """
    # 1) eigen ↓ 내림차순 정렬
    eigval, eigvec = np.linalg.eigh(G)
    eigval = np.clip(eigval, 0, None)
    G_projected = eigvec @ np.diag(eigval) @ eigvec.T
    idx = eigval.argsort()[::-1]
    eigval_f, eigvec_f = eigval[idx], eigvec[:, idx]
    # 2) factor 부분
    F = eigvec_f @ np.diag(eigval_f) @ eigvec_f.T

    # 3) idio 잔차 + soft-threshold
    S = G_projected - F
    S_thr = S.copy()
    # soft thresholding
    # S_thr[np.abs(S_thr) < tau] = 0.0
    # hard thresholding
    for i in range(200):
        for j in range(200):
            if gicslist[i] != gicslist[j]:
                S_thr[i, j] = 0.0
            if i == j:
                if S_thr[i, j] < 0:
                    S_thr[i, j] = 0.0

    return F + S_thr


def qlike(G1, G2):
    return np.log(np.linalg.det(G1)) + np.trace(np.linalg.inv(G1) @ G2)

# ───────────────────────────  MSPE 계산  ───────────────────────────────
errs_poet_1 = []
errs_poet_2 = []
errs_poet_3 = []

qlike_1 = []
qlike_2 = []
qlike_3 = []

for d in period_1[:-1]:   
    
    G_today = pd.read_csv(os.path.join(path, f"d{d}.csv")).values
    G_next  = pd.read_csv(os.path.join(path, f"d{d+1}.csv")).values
    G_next = poet_prvm(G_next, r=r)
    G_pred  = poet_prvm(G_today, r=r)

    err = np.linalg.norm(G_pred - G_next, 'fro')**2
    errs_poet_1.append(err)
    if d in period_2:
        errs_poet_2.append(err)
    if d in period_3:
        errs_poet_3.append(err)
    print("day", d, "done")

mspe_poet_1 = np.mean(errs_poet_1)
mspe_poet_2 = np.mean(errs_poet_2)
mspe_poet_3 = np.mean(errs_poet_3)
qlike_1 = np.mean(qlike_1)
qlike_2 = np.mean(qlike_2)
qlike_3 = np.mean(qlike_3)
print(f"POET-PRVM period 1 MSPE (Frobenius²)  = {mspe_poet_1*(10**4)}")
print(f"POET-PRVM period 2 MSPE (Frobenius²)  = {mspe_poet_2*(10**4)}")
print(f"POET-PRVM period 3 MSPE (Frobenius²)  = {mspe_poet_3*(10**4)}")


