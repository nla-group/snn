from snnpy import snnomp
import numpy as np

# Test data
n, d = 7, 3
data_float = np.array([
    1.2, 2.0, 3.0,
    2.0, 2.4, 2.0,
    2.0, 1.0, 2.0,
    2.0, 3.2, 1.2,
    2.0, 3.1, 2.0,
    2.0, 2.2, 1.0,
    2.0, 2.1, 1.0
], dtype=np.float32).reshape(n, d)

data_double = data_float.astype(np.float64)  # Same data, double precision

# Test SNN_FLOAT
print("Testing SNN_FLOAT (float32):")
snn_float = snnomp.SNN_FLOAT(data_float, num_threads=4)

new_data_float = np.array([2.3, 3.2, 1.0], dtype=np.float32)
R_float = 2.0
indices_float = snn_float.query_radius(new_data_float, R_float)
print(f"Single query indices: {indices_float}")

new_data_batch_float = np.array([
    [2.3, 3.2, 1.0],
    [1.5, 2.5, 2.5],
    [2.1, 1.8, 1.2]
], dtype=np.float32)

all_indices_float = snn_float.query_radius_batch(new_data_batch_float, R_float)
print("Multiple query indices:")
for j, indices in enumerate(all_indices_float):
    print(f"Query {j}: {indices}")

print(f"Mean: {snn_float.mean}")
print(f"First PC: {snn_float.first_pc}")

# Test SNN_DOUBLE
print("\nTesting SNN_DOUBLE (float64):")
snn_double = snnomp.SNN_DOUBLE(data_double, num_threads=4)

new_data_double = np.array([2.3, 3.2, 1.0], dtype=np.float64)
R_double = 2.0
indices_double = snn_double.query_radius(new_data_double, R_double)
print(f"Single query indices: {indices_double}")

new_data_batch_double = np.array([
    [2.3, 3.2, 1.0],
    [1.5, 2.5, 2.5],
    [2.1, 1.8, 1.2]
], dtype=np.float64)
all_indices_double = snn_double.query_radius_batch(new_data_batch_double, R_double)
print("Multiple query indices:")
for j, indices in enumerate(all_indices_double):
    print(f"Query {j}: {indices}")

print(f"Mean: {snn_double.mean}")
print(f"First PC: {snn_double.first_pc}")