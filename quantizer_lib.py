import torch
import torch.nn as nn
import copy
import os

class ManualQuantizer:
    def __init__(self, model):
        # Work on a copy to avoid modifying the original during sweep
        self.model = copy.deepcopy(model)
        self.hooks = []
        self.device = next(model.parameters()).device
        
    # -------------------------------------------
    # Quantization Math
    # -------------------------------------------
    def _quantize(self, x, bits, per_channel=False):
        q_min = -2**(bits - 1)
        q_max = 2**(bits - 1) - 1
        
        if per_channel and x.dim() == 4:
            # Per-Channel Scale: [C_out, 1, 1, 1]
            x_flat = x.view(x.size(0), -1)
            max_val = x_flat.abs().max(dim=1)[0]
            scale = max_val / q_max
            scale = scale.view(-1, 1, 1, 1)
        else:
            # Per-Tensor Scale
            max_val = x.abs().max()
            scale = max_val / q_max
            
        scale = torch.clamp(scale, min=1e-8)
        
        # Fake Quantization logic
        x_int = torch.round(x / scale)
        x_int = torch.clamp(x_int, q_min, q_max)
        x_dequant = x_int * scale
        
        return x_dequant, scale, x_int

    def apply_weight_quantization(self, w_bits):
        """Quantizes weights in place and marks them for saving."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Sensitivity check: Keep first and last layers at min 8-bit
                current_bits = w_bits
                if "features.0.0" in name or "classifier" in name:
                    current_bits = max(8, w_bits)
                
                per_channel = isinstance(module, nn.Conv2d)
                
                with torch.no_grad():
                    w_dequant, scale, w_int = self._quantize(
                        module.weight, current_bits, per_channel
                    )
                    module.weight.copy_(w_dequant)
                    
                # Store info needed for saving later
                module.quant_info = {
                    'bits': current_bits,
                    'scale': scale,
                    'w_int': w_int,
                    'per_channel': per_channel
                }

    def register_activation_hooks(self, a_bits):
        """Registers runtime hooks for activation quantization."""
        if a_bits >= 32: return

        def hook_fn(module, input, output):
            # Asymmetric Quantization for ReLUs [0, max]
            q_min, q_max = 0, 2**a_bits - 1
            min_val, max_val = output.min(), output.max()
            scale = (max_val - min_val) / (q_max - q_min) + 1e-8
            zero_point = torch.round(-min_val / scale)
            
            x_int = torch.round(output / scale + zero_point)
            x_int = torch.clamp(x_int, q_min, q_max)
            return (x_int - zero_point) * scale

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.ReLU6)):
                self.hooks.append(module.register_forward_hook(hook_fn))
                module.a_bits = a_bits # Mark for metadata

    # -------------------------------------------
    # Compression (Saving) Logic - FIXED
    # -------------------------------------------
    def save_compressed_model(self, path):
        """
        Saves the FULL model state, compressing only the quantized weights.
        """
        print(f"[*] Compressing and saving to {path}...")
        
        # 1. Start with the full state dictionary (includes BatchNorms!)
        original_state_dict = self.model.state_dict()
        compressed_state_dict = {}
        metadata = {}

        # 2. Identify Quantized Layers
        quantized_layers = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'quant_info'):
                quantized_layers[f"{name}.weight"] = (name, module.quant_info)

        # 3. Build Compressed Dict
        for key, value in original_state_dict.items():
            if key in quantized_layers:
                # This is a weight we want to compress
                layer_name, q_info = quantized_layers[key]
                
                # Store INT8 weight and Scale instead of Float32 weight
                compressed_state_dict[f"{layer_name}.weight_int"] = q_info['w_int'].to(torch.int8)
                compressed_state_dict[f"{layer_name}.scale"] = q_info['scale']
                
                # Metadata to know how to load it back
                metadata[layer_name] = {'type': 'quantized'}
            else:
                # This is BatchNorm, Bias, or unquantized weight -> Keep as is
                compressed_state_dict[key] = value

        # 4. Save Activation Hook Info
        a_bits = 32
        for m in self.model.modules():
            if hasattr(m, 'a_bits'): a_bits = m.a_bits; break
        metadata['activation_bits'] = a_bits
        
        # 5. Save to Disk
        torch.save({'state_dict': compressed_state_dict, 'metadata': metadata}, path)
        
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"    -> Saved Size: {size_mb:.2f} MB")
        return size_mb

# -------------------------------------------
# Decompression (Loading) Logic - FIXED
# -------------------------------------------
def load_compressed_model(path, device):
    """
    Loads the custom compressed format.
    Reconstructs Float32 weights from Int8 + Scale.
    """
    checkpoint = torch.load(path, map_location=device)
    compressed_state = checkpoint['state_dict']
    metadata = checkpoint['metadata']
    
    # 1. Initialize empty model structure
    from common import get_mobilenet_cifar10
    model = get_mobilenet_cifar10(pretrained=False)
    model.to(device)
    
    # 2. Reconstruct the Standard State Dict
    final_state_dict = {}
    
    # We loop over the keys in the COMPRESSED dict
    # But we need to handle the fact that some keys are split (weight_int + scale)
    
    # Helper to check if a key is part of a quantized pair
    processed_keys = set()
    
    for key in compressed_state.keys():
        if key in processed_keys: continue
        
        if key.endswith(".weight_int"):
            # It's a compressed weight. Find its base name.
            base_name = key.replace(".weight_int", "")
            
            w_int = compressed_state[key].to(device).float()
            scale = compressed_state[f"{base_name}.scale"].to(device)
            
            # Dequantize
            w_float = w_int * scale
            
            # Add to final dict with the standard PyTorch name
            final_state_dict[f"{base_name}.weight"] = w_float
            
            # Mark scale as processed so we don't copy it raw later (though usually filtered by logic)
            processed_keys.add(f"{base_name}.scale")
        
        elif key.endswith(".scale") and f"{key.replace('.scale','.weight_int')}" in compressed_state:
            # This is a scale file, handled above. Skip.
            continue
            
        else:
            # Standard item (BatchNorms, Biases, etc.)
            final_state_dict[key] = compressed_state[key]
            
    # 3. Load into model
    model.load_state_dict(final_state_dict, strict=True)

    # 4. Re-apply Activation Quantization Hooks
    a_bits = metadata.get('activation_bits', 32)
    if a_bits < 32:
        quantizer = ManualQuantizer(model)
        quantizer.register_activation_hooks(a_bits)
        return quantizer.model
        
    return model
