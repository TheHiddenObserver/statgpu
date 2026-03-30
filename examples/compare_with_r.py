"""
Compare statgpu with R's lm() function.
"""

import numpy as np
import subprocess
import tempfile
import os

print("=" * 80)
print("Comparison: statgpu vs R lm()")
print("=" * 80)

# Generate data with fixed seed (same as R's set.seed(42))
np.random.seed(42)
n_samples, n_features = 100, 5
X = np.random.randn(n_samples, n_features)
true_coef = np.array([1.5, -2.0, 3.0, 0.5, -1.0])
true_intercept = 5.0
y = X @ true_coef + true_intercept + np.random.randn(n_samples) * 0.5

# Save data for R
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    data_file = f.name
    f.write("y," + ",".join([f"x{i+1}" for i in range(n_features)]) + "\n")
    for i in range(n_samples):
        row = [y[i]] + list(X[i])
        f.write(",".join([str(v) for v in row]) + "\n")

print(f"\nDataset: {n_samples} samples, {n_features} features")
print(f"Data saved to: {data_file}")

# ============================================
# 1. statgpu
# ============================================
print("\n" + "=" * 80)
print("statgpu LinearRegression")
print("=" * 80)

from statgpu.linear_model import LinearRegression
from statgpu._config import set_device

set_device('cpu')
model = LinearRegression(device='cpu')
model.fit(X, y)
model.summary()

# ============================================
# 2. R lm()
# ============================================
print("\n" + "=" * 80)
print("R lm() output")
print("=" * 80)

r_script = f'''
data <- read.csv("{data_file}")
model <- lm(y ~ ., data=data)
summary(model)
'''

# Run R script
try:
    result = subprocess.run(
        ['Rscript', '-e', r_script],
        capture_output=True,
        text=True,
        timeout=30
    )
    print(result.stdout)
    if result.stderr:
        print("R stderr:", result.stderr)
except Exception as e:
    print(f"Error running R: {e}")

# Cleanup
os.unlink(data_file)

print("\n" + "=" * 80)
print("Comparison Complete")
print("=" * 80)
