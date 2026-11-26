import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Ensure output directory exists
os.makedirs("../quantized_data", exist_ok=True)

# 1. Generate 4-Class Motor Vibration Features (8 features per sample)
# Features: [Mean, RMS, StdDev, Skewness, Kurtosis, Peak2Peak, CrestFactor, ShapeFactor]
np.random.seed(42)
samples_per_class = 300
X_list, y_list = [], []

class_profiles = [
    ([0.02, 0.15, 0.14, 0.05, 3.0, 0.85, 2.8, 1.25], 0),  # Class 0: Normal
    ([0.05, 0.45, 0.44, 0.35, 6.2, 3.10, 4.5, 1.48], 1),  # Class 1: Inner Race Fault
    ([0.04, 0.62, 0.61, 0.20, 4.8, 3.80, 3.9, 1.38], 2),  # Class 2: Outer Race Fault
    ([0.03, 0.32, 0.31, 0.60, 8.5, 2.75, 5.4, 1.62], 3),  # Class 3: Ball Fault
]

for means, label in class_profiles:
    noise = np.random.normal(0, 0.12, size=(samples_per_class, 8)) * np.array(means)
    noise += np.random.normal(0, 0.03, size=(samples_per_class, 8))
    X_list.append(np.array(means) + noise)
    y_list.append(np.full(samples_per_class, label))

X = np.vstack(X_list)
y = np.concatenate(y_list)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# 2. Train Compact FP32 MLP: 8 Inputs -> 16 Hidden (ReLU) -> 4 Outputs
mlp = MLPClassifier(hidden_layer_sizes=(16,), activation='relu', max_iter=500, random_state=42)
mlp.fit(X_train_norm, y_train)

fp32_preds = mlp.predict(X_test_norm)
fp32_acc = accuracy_score(y_test, fp32_preds) * 100
print(f"FP32 Model Accuracy: {fp32_acc:.2f}%")

W1_fp32 = mlp.coefs_[0]       # Shape: (8, 16)
b1_fp32 = mlp.intercepts_[0]  # Shape: (16,)
W2_fp32 = mlp.coefs_[1]       # Shape: (16, 4)
b2_fp32 = mlp.intercepts_[1]  # Shape: (4,)

# 3. Hardware-Friendly Power-of-2 INT8 Quantization
def get_shift_scale(max_val, target_max=127.0):
    raw_scale = target_max / max(max_val, 1e-8)
    shift = int(np.floor(np.log2(raw_scale)))
    return 2 ** shift, shift

in_scale, in_shift = get_shift_scale(np.max(np.abs(X_train_norm)), 127.0)
X_test_int8 = np.clip(np.round(X_test_norm * in_scale), -128, 127).astype(np.int8)

w1_scale, w1_shift = get_shift_scale(np.max(np.abs(W1_fp32)), 127.0)
W1_int8 = np.clip(np.round(W1_fp32 * w1_scale), -128, 127).astype(np.int8)
b1_int32 = np.round(b1_fp32 * (in_scale * w1_scale)).astype(np.int32)

l1_raw_acc = (X_test_int8.astype(np.int32) @ W1_int8.astype(np.int32)) + b1_int32
l1_relu = np.maximum(0, l1_raw_acc)
_, shift_l1 = get_shift_scale(np.max(l1_relu), 127.0)
shift_l1 = abs(shift_l1)

l1_act_scale = (in_scale * w1_scale) / (2 ** shift_l1)
w2_scale, w2_shift = get_shift_scale(np.max(np.abs(W2_fp32)), 127.0)
W2_int8 = np.clip(np.round(W2_fp32 * w2_scale), -128, 127).astype(np.int8)
b2_int32 = np.round(b2_fp32 * (l1_act_scale * w2_scale)).astype(np.int32)

l1_out_int8 = np.clip(l1_relu >> shift_l1, 0, 127).astype(np.int8)
l2_raw_acc = (l1_out_int8.astype(np.int32) @ W2_int8.astype(np.int32)) + b2_int32
_, shift_l2 = get_shift_scale(np.max(np.abs(l2_raw_acc)), 127.0)
shift_l2 = abs(shift_l2)

# 4. Pure Integer Hardware-Exact Inference Simulation
def hardware_infer(x_int8):
    acc1 = (x_int8.astype(np.int32) @ W1_int8.astype(np.int32)) + b1_int32
    relu1 = np.maximum(0, acc1)
    act1 = np.clip(relu1 >> shift_l1, 0, 127).astype(np.int8)

    acc2 = (act1.astype(np.int32) @ W2_int8.astype(np.int32)) + b2_int32
    out2 = np.clip(acc2 >> shift_l2, -128, 127).astype(np.int8)
    pred = int(np.argmax(out2))
    return acc1, act1, acc2, out2, pred

int8_preds = [hardware_infer(x)[4] for x in X_test_int8]
int8_acc = accuracy_score(y_test, int8_preds) * 100
print(f"INT8 Hardware-Exact Model Accuracy: {int8_acc:.2f}%")
print(f"Hardware Constants -> SHIFT_L1 = {shift_l1}, SHIFT_L2 = {shift_l2}")

# 5. Export .mem Hex Files for Verilog $readmemh
def save_hex(filename, array, bits):
    mask = (1 << bits) - 1
    hex_chars = bits // 4
    with open(f"../quantized_data/{filename}", "w") as f:
        for val in array.flatten():
            f.write(f"{int(val) & mask:0{hex_chars}x}\n")

save_hex("weights_l1.mem", W1_int8.T, 8)
save_hex("biases_l1.mem", b1_int32, 32)
save_hex("weights_l2.mem", W2_int8.T, 8)
save_hex("biases_l2.mem", b2_int32, 32)
save_hex("test_inputs.mem", X_test_int8[:20], 8)

with open("../quantized_data/golden_outputs.txt", "w") as f:
    f.write(f"# SHIFT_L1={shift_l1}, SHIFT_L2={shift_l2}\n")
    for idx in range(20):
        acc1, act1, acc2, out2, pred = hardware_infer(X_test_int8[idx])
        f.write(f"Sample {idx:02d} | True={y_test[idx]} | Pred={pred} | L1_Act={act1.tolist()} | L2_Out={out2.tolist()}\n")

print("Exported .mem files and golden_outputs.txt to quantized_data/ successfully!")