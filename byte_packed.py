import numpy as np

def pack_ternary_weights(w_q):
    """
    Compresses ternary weights (-1, 0, 1) to C++ binary format.
Mapping: -1 -> 0, 0 -> 1, 1 -> 2 (Base-3 packing or 2-bit representation per weight).
    """
    w_mapped = (w_q.detach().cpu().numpy() + 1).astype(np.uint8)
    flat_weights = w_mapped.flatten()
    padding_len = (4 - (len(flat_weights) % 4)) % 4
    if padding_len > 0:
        flat_weights = np.pad(flat_weights, (0, padding_len), 'constant')
    packed = np.zeros(len(flat_weights) // 4, dtype=np.uint8)
    for i in range(4):
        packed |= (flat_weights[i::4] & 0x03) << (i * 2)
        
    return packed

def export_bitlinear_to_cpp(model, filepath):
    """
    Exports BitLinear parameters to a custom binary file for the C++ runtime.
    """
    state_dict = model.state_dict()
    with open(filepath, "wb") as f:
        for name, param in state_dict.items():
            if "weight" in name:
                w_q, gamma = TernaryQuantizeSTE.apply(param)
                f.write(np.array([gamma.item()], dtype=np.float32).tobytes())
                packed_bytes = pack_ternary_weights(w_q)
                f.write(np.array([len(packed_bytes)], dtype=np.int32).tobytes())
                f.write(packed_bytes.tobytes())
            elif "bias" in name and param is not None:
                f.write(param.detach().cpu().numpy().astype(np.float32).tobytes())
    print(f"Model successfully exported to {filepath} for C++ runtime.")
