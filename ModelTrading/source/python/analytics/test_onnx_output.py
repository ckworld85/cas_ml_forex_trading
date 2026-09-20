import os
import numpy as np
import onnxruntime as ort

# Test ONNX model outputs to see what they actually return
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GENERATED_DIR = os.path.join(BASE_DIR, "generated")

# Load a sample input (just random for testing structure)
import pandas as pd
X = pd.read_parquet(os.path.join(GENERATED_DIR, "X.parquet"))
sample_input = X.iloc[100:101].values.astype(np.float32)

print("Testing ONNX Model Outputs")
print("=" * 60)
print(f"Sample input shape: {sample_input.shape}")
print()

# Test main classifier
model_path = os.path.join(GENERATED_DIR, "strategy_model.onnx")
if os.path.exists(model_path):
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    
    result = sess.run(None, {input_name: sample_input})
    
    print("Main Classifier (strategy_model.onnx):")
    print(f"  Input name: {input_name}")
    print(f"  Output names: {output_names}")
    print(f"  Number of outputs: {len(result)}")
    for i, r in enumerate(result):
        print(f"  Output {i} shape: {r.shape}, dtype: {r.dtype}")
        print(f"  Output {i} value: {r}")
    print()

# Test hit30 model
model_hit30_path = os.path.join(GENERATED_DIR, "strategy_model_hit30_long.onnx")
if os.path.exists(model_hit30_path):
    sess_hit30 = ort.InferenceSession(model_hit30_path)
    input_name = sess_hit30.get_inputs()[0].name
    output_names = [o.name for o in sess_hit30.get_outputs()]
    
    result = sess_hit30.run(None, {input_name: sample_input})
    
    print("Hit30 Long Model (strategy_model_hit30_long.onnx):")
    print(f"  Input name: {input_name}")
    print(f"  Output names: {output_names}")
    print(f"  Number of outputs: {len(result)}")
    for i, r in enumerate(result):
        print(f"  Output {i} shape: {r.shape}, dtype: {r.dtype}")
        print(f"  Output {i} value: {r}")
    print()

print("=" * 60)
print("Test complete. Check which output index contains probabilities.")
