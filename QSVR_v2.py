# Manual Quantum Kernel Regression (QSVR) for Lottery Prediction
# Quantum Regression Model with Qiskit


import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector


from qiskit_machine_learning.utils import algorithm_globals
import random

# ================= SEED PARAMETERS =================
SEED = 39
random.seed(SEED)
np.random.seed(SEED)
algorithm_globals.random_seed = SEED
# ==================================================


# Use the existing dataframe
df_raw = pd.read_csv('/Users/4c/Desktop/GHQ/data/loto7hh_4586_k24.csv')
# 4586 historical draws of Lotto 7/39 (Serbia)

_MIN_POS = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
_MAX_POS = np.array([33, 34, 35, 36, 37, 38, 39], dtype=int)


def compute_quantum_kernel(X1, X2, feature_map):
    """
    Manually compute the quantum kernel matrix K(i, j) = |<phi(x1_i)|phi(x2_j)>|^2
    """
    n1 = len(X1)
    n2 = len(X2)
    kernel_matrix = np.zeros((n1, n2))
    
    # Pre-compute statevectors for efficiency
    sv1 = []
    for x in X1:
        bound_circuit = feature_map.assign_parameters(x)
        sv1.append(Statevector.from_instruction(bound_circuit))
        
    sv2 = []
    for x in X2:
        bound_circuit = feature_map.assign_parameters(x)
        sv2.append(Statevector.from_instruction(bound_circuit))
        
    for i in range(n1):
        for j in range(n2):
            # Fidelity = |<psi|phi>|^2
            fidelity = np.abs(np.vdot(sv1[i].data, sv2[j].data))**2
            kernel_matrix[i, j] = fidelity
            
    return kernel_matrix

def quantum_kernel_regression_predict(df):
    df = df.copy()
    cols = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6', 'Num7']
    predictions = {}
    
    # Use 2 lags to map to a 2-qubit quantum space
    num_lags = 2
    num_qubits = 2
    
    # Use a small window for the kernel method (O(N^2) complexity)
    train_window = 36
    
    # Define a ZZFeatureMap for 2 qubits
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='linear')
    
    for idx, col in enumerate(cols):
        # Prepare data with lags
        df_col = pd.DataFrame(df[col])
        for i in range(1, num_lags + 1):
            df_col[f'lag_{i}'] = df_col[col].shift(i)
        
        df_col = df_col.dropna().tail(train_window + 1)
        
        X = df_col[[f'lag_{i}' for i in range(1, num_lags + 1)]].values
        y = df_col[col].values
        
        X_train = X[:-1]
        y_train = y[:-1]
        X_next = X[-1:]
        
        # Scale to [0, 2*pi] for the feature map
        scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
        X_train_scaled = scaler.fit_transform(X_train)
        X_next_scaled = scaler.transform(X_next)
        
        # Compute Kernel Matrices
        K_train = compute_quantum_kernel(X_train_scaled, X_train_scaled, feature_map)
        K_test = compute_quantum_kernel(X_next_scaled, X_train_scaled, feature_map)
        
        # Fit SVR with precomputed kernel
        svr = SVR(kernel='precomputed', C=8.0, epsilon=0.08)
        svr.fit(K_train, y_train)
        
        # Predict
        pred = svr.predict(K_test)
        lo, hi = int(_MIN_POS[idx]), int(_MAX_POS[idx])
        predictions[col] = int(round(np.clip(pred[0], lo, hi)))
        
    return predictions

print()
print("Computing predictions using Manual Quantum Kernel Regression (QSVR) ...")
q_kernel_results = quantum_kernel_regression_predict(df_raw)

# Format for display
q_kernel_df = pd.DataFrame([q_kernel_results])
# q_kernel_df.index = ['Quantum Kernel Regression (QSVR)']

print()
print("Lottery prediction generated using a manual Quantum Kernel implementation.")
print()


print()
print("Quantum Kernel Regression (QSVR) Results:")
print(q_kernel_df.to_string(index=True))
print()
"""
Computing predictions using Manual Quantum Kernel Regression (QSVR) ...

feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='linear')

Lottery prediction generated using a manual Quantum Kernel implementation.


Quantum Kernel Regression (QSVR) Results:
   Num1  Num2  Num3  Num4  Num5  Num6  Num7
0     x    11    18    15    25    y    z
"""



"""
df.copy() da se ne menja ulaz
train_window 30 -> 36
ZZFeatureMap reps 1 -> 2
SVR finiji: C=8.0, epsilon=0.08
clip po pozicijama za sortirani 7/39 (1..33 … 7..39)
"""
