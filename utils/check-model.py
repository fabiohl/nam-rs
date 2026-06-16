#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

import sys
import json
import os

# Define standard A1 WaveNet topologies
STD_DILATIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
LITE_DILATIONS_1 = [1, 2, 4, 8, 16, 32, 64]
LITE_DILATIONS_2 = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Standard LSTM configs from model mappings
STANDARD_LSTMS = {
    1: {3, 8, 12, 16, 24, 40},
    2: {8, 12, 16, 24}
}

def classify_model(data, filename=""):
    arch = data.get("architecture")
    config = data.get("config", {})
    version = data.get("version", "unknown")
    
    metadata = data.get("metadata", {})
    modeled_by = metadata.get("modeled_by") if metadata else None
    model_name = metadata.get("name") if metadata else None
    
    # Extract metadata properties
    author_str = modeled_by if modeled_by else "Unknown Author"
    name_str = model_name if model_name else os.path.basename(filename)
    
    if arch == "WaveNet":
        layers = config.get("layers", [])
        
        # Check if it is a container
        if config.get("submodels") is not None:
            return {
                "name": name_str,
                "author": author_str,
                "version": version,
                "arch": "SlimmableContainer",
                "details": f"SlimmableContainer with {len(config['submodels'])} submodels",
                "status": "Matches F12/F5: Real Slimmable Container",
                "is_goal": True
            }
            
        # Is it A2 shape?
        is_a2 = False
        a2_reason = ""
        if len(layers) == 1:
            l = layers[0]
            if l.get("kernel_sizes") is not None or l.get("bottleneck") is not None:
                is_a2 = True
                a2_reason = "A2 Shape (1 layer, bottleneck/kernel_sizes present)"
            elif isinstance(l.get("activation"), list):
                is_a2 = True
                a2_reason = "A2 activation array"
                
        # Also any condition_size > 1 or gating or slimmable or FiLM
        for i, l in enumerate(layers):
            if l.get("condition_size", 1) > 1:
                return {
                    "name": name_str,
                    "author": author_str,
                    "version": version,
                    "arch": "WaveNet A2 (FiLM)",
                    "details": f"Layer {i} has condition_size={l.get('condition_size')}",
                    "status": "Matches F12/F2/F3: A2 FiLM / Multi-Condição",
                    "is_goal": True
                }
                
        if is_a2:
            return {
                "name": name_str,
                "author": author_str,
                "version": version,
                "arch": "WaveNet A2",
                "details": a2_reason,
                "status": "Matches F12/F3: A2 Geral",
                "is_goal": True
            }
            
        # Check standard A1 topologies
        if len(layers) == 2:
            l0 = layers[0]
            l1 = layers[1]
            ch0 = l0.get("channels")
            dils0 = l0.get("dilations", [])
            dils1 = l1.get("dilations", [])
            
            # Lite topology check matching topology.rs:
            # 12 channels, dils_0 == LITE_DILATIONS_1, dils_1 == LITE_DILATIONS_2
            if ch0 == 12 and dils0 == LITE_DILATIONS_1 and dils1 == LITE_DILATIONS_2:
                return {
                    "name": name_str,
                    "author": author_str,
                    "version": version,
                    "arch": "WaveNet A1 Lite (CH=12)",
                    "details": "Standard A1 Lite topology",
                    "status": "Matches F12: Real WaveNet Lite (CH=12) (Valid target to replace BossWN-lite.nam)",
                    "is_goal": True
                }
                
            # Other standard shapes: 16 (Standard), 8 (Feather), 4 (Nano)
            is_standard = False
            if ch0 == 16 and dils0 == STD_DILATIONS and dils1 == STD_DILATIONS:
                is_standard = True
            elif ch0 == 8 and dils0 == LITE_DILATIONS_1 and dils1 == LITE_DILATIONS_2:
                is_standard = True
            elif ch0 == 4 and dils0 == LITE_DILATIONS_1 and dils1 == LITE_DILATIONS_2:
                is_standard = True
                
            if not is_standard:
                return {
                    "name": name_str,
                    "author": author_str,
                    "version": version,
                    "arch": "WaveNet A1 (Custom)",
                    "details": f"Non-standard shape: CH={ch0}, dilations_len={len(dils0)}/{len(dils1)}",
                    "status": "Matches F12/F1: A1 Custom Geometry (Negative fixture target)",
                    "is_goal": True
                }
            else:
                return {
                    "name": name_str,
                    "author": author_str,
                    "version": version,
                    "arch": f"WaveNet A1 (Standard CH={ch0})",
                    "details": "Standard A1 topology",
                    "status": "Standard Supported Model (Already fully supported in main branch)",
                    "is_goal": False
                }
        else:
            return {
                "name": name_str,
                "author": author_str,
                "version": version,
                "arch": "WaveNet (Custom Layers)",
                "details": f"Number of layers is {len(layers)} (expected 2)",
                "status": "Matches F12/F1: A1 Custom Geometry",
                "is_goal": True
            }
            
    elif arch == "LSTM":
        num_layers = config.get("num_layers")
        hidden_size = config.get("hidden_size")
        
        if num_layers is None or hidden_size is None:
            return {
                "name": name_str,
                "author": author_str,
                "version": version,
                "arch": "LSTM (Invalid Config)",
                "details": "Missing num_layers or hidden_size",
                "status": "Invalid model structure",
                "is_goal": False
            }
            
        allowed = STANDARD_LSTMS.get(num_layers, set())
        if hidden_size not in allowed:
            return {
                "name": name_str,
                "author": author_str,
                "version": version,
                "arch": f"LSTM {num_layers}x{hidden_size} (Custom)",
                "details": f"Non-standard geometry: layers={num_layers}, hidden={hidden_size}",
                "status": "Matches F12/F7: LSTM Custom Shape",
                "is_goal": True
            }
        else:
            return {
                "name": name_str,
                "author": author_str,
                "version": version,
                "arch": f"LSTM {num_layers}x{hidden_size} (Standard)",
                "details": "Standard LSTM topology",
                "status": "Standard Supported Model (Already fully supported in main branch)",
                "is_goal": False
            }
    elif arch == "Linear":
        rf = config.get("receptive_field", 0)
        bias = config.get("bias", False)
        return {
            "name": name_str,
            "author": author_str,
            "version": version,
            "arch": f"Linear (RF={rf}, bias={bias})",
            "details": "Linear model structure",
            "status": "Standard Supported Model",
            "is_goal": False
        }
    else:
        return {
            "name": name_str,
            "author": author_str,
            "version": version,
            "arch": f"Unknown ({arch})",
            "details": "Unsupported architecture",
            "status": "Unknown / Unsupported",
            "is_goal": False
        }

def main():
    if len(sys.argv) < 2:
        print("Usage: utils/check-model.py <path_to_model.nam> [path_to_model2.nam ...]")
        sys.exit(1)
        
    for filepath in sys.argv[1:]:
        if not os.path.exists(filepath):
            print(f"\033[91mError: File not found: {filepath}\033[0m")
            continue
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"\033[91mError: Failed to parse JSON for {filepath}: {e}\033[0m")
            continue
            
        info = classify_model(data, filepath)
        
        # Color coding for F12 goal targets
        if info["is_goal"]:
            color = "\033[92m" # Green
            tag = "[TARGET F12 MATCH]"
        else:
            color = "\033[94m" # Blue
            tag = "[STANDARD/SUPPORTED]"
            
        print(f"================================================================")
        print(f"File: {filepath}")
        print(f"Name: {info['name']} (by {info['author']})")
        print(f"Version: {info['version']}")
        print(f"Architecture: {info['arch']}")
        print(f"Details: {info['details']}")
        print(f"Status: {color}{tag} {info['status']}\033[0m")
        print(f"================================================================")

if __name__ == "__main__":
    main()
