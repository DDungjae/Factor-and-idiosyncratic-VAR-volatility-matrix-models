# FIVAR Volatility Modeling — Implementation and Experiments

This repository contains code and notes for implementing and reproducing key ideas from the paper:

> **Factor and Idiosyncratic VAR Volatility Matrix Models for Heavy-Tailed High-Frequency Financial Observations**  
> Minseok Shin (POSTECH), Donggyu Kim (UC Riverside),  
> Yazhen Wang (University of Wisconsin–Madison / NSF), Jianqing Fan (Princeton University)  
> *Journal of Econometrics*, Vol. 252, Part A, 2025  
> arXiv: 2109.05227  
> SSRN: Working Paper (First posted 2024-03-21, Revised 2025-09-24)

This repository focuses on understanding the paper by **implementing its modeling pipeline in code**, including volatility matrix construction, factor–idiosyncratic decomposition, and evaluation of competing estimators under heavy-tailed high-frequency data.  
The seminar slides used for the accompanying presentation are also included.

---

## Repository Structure

### **1. `PRVM_v3.py` — Pre-Averaging Realized Volatility Matrix (PRVM)**
Implements the **pre-averaging realized volatility matrix estimator**, designed to:
- Mitigate microstructure noise
- Control heavy-tailed returns via truncation
- Provide a stable matrix for decomposition

### **2. `log_return.py` — High-Frequency Log-Return Generator**
Creates standardized **log-return files** from raw price data.  
These returns serve as inputs for PRVM and subsequent modeling.

### **3. Evaluation Scripts**

#### **`evaluate_POET_PRVM.py`**
Implements **POET + PRVM** as the benchmark estimator:
- POET factor extraction  
- Previous-day PRVM as the volatility predictor  
- Used as “ground truth” in MSPE/QLIKE comparisons

#### **`evaluate_H_LASSO.py`**
Implements **Huber-LASSO (H-LASSO)** for robust VAR estimation:
- Handles heavy-tailed innovations  
- L1-penalized sparse regression  
- BIC-based coefficient selection  

#### **`evaluate_OLS_MSPE.py`**
Implements **OLS-based VAR** for the factor component and simple averaging for idiosyncratic volatility:
- Baseline estimator  
- Sensitive to outliers and heavy tails

---

## 📈 Workflow Overview

### 1. Construct High-Frequency Log Returns
`log_return.py`

### 2. Compute PRVM
`PRVM_v3.py`  
- Pre-averaging  
- Truncation  
- PSD projection  

### 3. Decompose Volatility Matrix
- Estimate time-invariant eigenvectors  
- Split into **factor** and **idiosyncratic** components  
- Hard-threshold idiosyncratic structure (e.g., GICS sector-based)

### 4. Fit VAR Dynamics
- Factor VAR: no penalty, low-dimensional  
- Idiosyncratic VAR: Huber-LASSO for sparsity and robustness  

### 5. Predict Next-Day Volatility
- Predict eigenvalues via VAR  
- Reconstruct volatility matrix using fixed eigenvectors  
- Combine factor + idiosyncratic estimates  

### 6. Evaluate with MSPE / QLIKE
Using:
- `evaluate_POET_PRVM.py`
- `evaluate_H_LASSO.py`
- `evaluate_OLS_MSPE.py`

---

## Numerical Highlights
- Estimated rank \( r = 3 \) in all sample periods  
- Optimal VAR lag \( h = 1 \) via BIC  
- POET-PRVM and OLS show stable behavior  
- H-LASSO requires careful idiosyncratic matrix construction  
- MSPE / QLIKE used as evaluation metrics  

---

## Seminar Slides
`seminar.pdf` contains the full presentation summarizing:
- Motivation  
- FIVAR model  
- Heavy-tailed estimation  
- PRVM construction  
- Numerical results  
- Future directions  


---

## Citation

If you use or extend this repository, please cite the original paper:

```
Shin, M., Kim, D., Wang, Y., & Fan, J. (2025). 
Factor and Idiosyncratic VAR Volatility Matrix Models for Heavy-Tailed High-Frequency Financial Observations. 
Journal of Econometrics, 252(Part A).
```

