# tools/profiler_python_utils.py
# Adapted for Chatterbox TTS profiling
import json
import os
import logging
from collections import defaultdict, Counter

def _recursive_walk(node, registry):
    func_name = node.get('function', '<unknown>')
    file_path = node.get('file_path_short') or node.get('file_path') or ''
    norm_path = file_path.replace('\\', '/')
    
    is_library = any(x in norm_path for x in ["site-packages", "dist-packages", "lib/python", "ray/"])
    # ---- PATCHED: recognize chatterbox project paths ----
    is_project = any(x in norm_path for x in [
        "chatterbox", "s3gen", "s3tokenizer", "t3/", "voice_encoder",
        "profile_chatterbox", "src/", "models/",
    ])
    
    is_user_code = is_project or (not is_library and "python" not in norm_path.lower())

    key = f"{func_name}::{file_path}"
    time_inclusive = node.get('time', 0.0)
    children_time = sum(c.get('time', 0.0) for c in node.get('children', []))
    time_exclusive = max(0.0, time_inclusive - children_time)
    
    if key not in registry:
        registry[key] = {
            'function': func_name, 'file': file_path,
            'total_time': 0.0, 'self_time': 0.0, 'calls': 0,
            'is_user_code': is_user_code
        }
    
    registry[key]['total_time'] += time_inclusive
    registry[key]['self_time'] += time_exclusive
    registry[key]['calls'] += 1

    for child in node.get('children', []):
        _recursive_walk(child, registry)

def process_profile_json(json_path: str, output_txt_path: str):
    if not os.path.exists(json_path): return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content: return
            data = json.loads(content)
            
        root_frame = data.get('root_frame', data)
        registry = {}
        _recursive_walk(root_frame, registry)
        
        stats_list = [s for s in registry.values() if s['is_user_code']]
        stats_list.sort(key=lambda x: x['total_time'], reverse=True)
        
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"--- CHATTERBOX PERFORMANCE SUMMARY (Python) ---\nSource: {json_path}\n")
            f.write(f"FUNCTION | TOTAL (s) | SELF (s) | CALLS | LOCATION\n---\n")
            for item in stats_list[:50]:
                f.write(f"{item['function']} | {item['total_time']:.4f} | {item['self_time']:.4f} | {item['calls']} | {item['file']}\n")
    except Exception as e:
        logging.error(f">>> PROFILER: Error generating summary: {e}")

def _clean_kernel_name(name: str) -> str:
    if not name or not isinstance(name, str): return "unknown"
    if not name.startswith("_Z"):
        if "pybind11" in name: return "[pybind11_wrapper]"
        return name

    lower_name = name.lower()
    if "gemm" in lower_name: return "[GEMM Kernel]"
    if "convolution" in lower_name or "conv" in lower_name: return "[Conv Kernel]"
    if "elementwise" in lower_name: return "[Elementwise Kernel]"
    if "layer_norm" in lower_name: return "[LayerNorm Kernel]"
    if "softmax" in lower_name: return "[Softmax Kernel]"
    if "copy" in lower_name: return "[Copy Kernel]"
    if "add" in lower_name: return "[Add Kernel]"
    if "mul" in lower_name: return "[Mul Kernel]"
    if "activation" in lower_name or "silu" in lower_name or "relu" in lower_name: return "[Activation Kernel]"
    if "fft" in lower_name: return "[FFT Kernel]"
    if "scatter" in lower_name: return "[Scatter Kernel]"
    if "gather" in lower_name: return "[Gather Kernel]"
    if "index" in lower_name: return "[Indexing Kernel]"
    if "vectorized" in lower_name: return "[Vectorized Kernel]"
    return "[CUDA Kernel]"

def _is_user_frame(name: str) -> bool:
    """Helper to determine if a trace stack frame belongs to Chatterbox project code."""
    if not name: return False
    norm = name.replace('\\', '/')
    
    # 1. IGNORE: profiler itself, root modules, library internals
    ignore_list = [
        "profiling_wrapper.py", "profiler_python_utils.py", "sync_hunter.py",
        "<module>",
        "torch/", "asyncio/", "threading.py", "anyio/", "gradio/", "uvicorn/", 
        "starlette/", "queue.py", "contextvars", "concurrent/futures", 
        "<built-in", "lib/python", "site-packages", "transformers/"
    ]
    if any(ig in norm for ig in ignore_list): return False
        
    # 2. MATCH: Chatterbox project paths
    project_keywords = [
        "chatterbox", "s3gen", "s3tokenizer", "t3/", "t3.py",
        "voice_encoder", "hifigan", "flow_matching", "flow.py",
        "profile_chatterbox", "tts.py", "vc.py", "mtl_tts",
        "decoder.py", "text_encoder", "perceiver", "cond_enc",
        "f0_predictor", "xvector", "melspec", "mel.py",
        "models/", "src/",
    ]
    if any(k in norm for k in project_keywords): return True
        
    # 3. Fallback
    return ".py" in norm

def process_torch_trace_json(json_path: str, output_txt_path: str, top_k=50):
    if not os.path.exists(json_path): return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        if '"""' in raw_content:
            raw_content = raw_content.replace('"""', "'''") 
        data = json.loads(raw_content)
    except Exception as e:
        logging.error(f"Error reading Torch trace {json_path}: {e}")
        return

    events = data.get('traceEvents', [])
    events = [e for e in events if e.get('ph') == 'X' and 'dur' in e]
    if not events: return
    events.sort(key=lambda x: x['ts'])

    stats = defaultdict(lambda: {
        'total': 0.0, 'self': 0.0, 'calls': 0, 
        'callers': Counter(),       
        'user_callers': Counter(),  
        'shapes': Counter() 
    })
    
    stack = []
    
    user_frame_cache = {}
    def fast_is_user_frame(n):
        if n not in user_frame_cache:
            user_frame_cache[n] = _is_user_frame(n)
        return user_frame_cache[n]

    for e in events:
        start = e['ts']
        dur = e['dur']
        end = start + dur
        name = e.get('name', 'unknown')
        
        args = e.get('args', {})
        shape_str = str(args['Input Dims']).replace(" ", "") if 'Input Dims' in args else ""
        
        stats[name]['total'] += dur
        stats[name]['calls'] += 1
        stats[name]['self'] += dur 
        if shape_str:
            stats[name]['shapes'][shape_str] += 1

        while stack and stack[-1]['end'] <= start:
            stack.pop()

        parent_user_frame = None

        if stack:
            parent = stack[-1]
            stats[parent['name']]['self'] -= dur
            if stats[parent['name']]['self'] < 0: stats[parent['name']]['self'] = 0
            stats[name]['callers'][parent['name']] += dur
            
            parent_user_frame = parent['active_user_frame']
            if parent_user_frame:
                stats[name]['user_callers'][parent_user_frame] += dur

        is_user = fast_is_user_frame(name)
        active_user_frame = name if is_user else parent_user_frame

        stack.append({
            'name': name, 
            'end': end,
            'active_user_frame': active_user_frame
        })

    results = []
    for name, data in stats.items():
        most_common_shape = ""
        if data['shapes']:
            most_common_shape = data['shapes'].most_common(1)[0][0]
            if len(most_common_shape) > 40: most_common_shape = most_common_shape[:37] + "..."

        results.append({
            'name': _clean_kernel_name(name),
            'self_ms': data['self'] / 1000.0,
            'total_ms': data['total'] / 1000.0,
            'calls': data['calls'],
            'top_callers': data['callers'].most_common(3),           
            'top_user_callers': data['user_callers'].most_common(2), 
            'shape': most_common_shape
        })

    results.sort(key=lambda x: x['self_ms'], reverse=True)

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"--- CHATTERBOX TORCH KERNEL SUMMARY ---\nSource: {json_path}\n")
        f.write(f"FUNCTION/KERNEL | SELF (ms) | TOTAL (ms) | CALLS | INPUT SHAPES (Top)\n---\n")
        
        for r in results[:top_k]:
            shape_display = r['shape'] if r['shape'] else "-"
            f.write(f"{r['name']} | {r['self_ms']:.2f} | {r['total_ms']:.2f} | {r['calls']} | {shape_display}\n")
            
            if r['top_user_callers']:
                f.write(f"  [Origin Scripts]:\n")
                for caller_name, caller_dur_us in r['top_user_callers']:
                    caller_ms = caller_dur_us / 1000.0
                    pct = (caller_ms / r['total_ms']) * 100 if r['total_ms'] > 0 else 0
                    f.write(f"   -> {caller_name} ({caller_ms:.1f}ms - {pct:.0f}%)\n")
            
            if r['top_callers']:
                f.write(f"  [Immediate Internal Callers]:\n")
                for caller_name, caller_dur_us in r['top_callers']:
                    if any(caller_name == u[0] for u in r['top_user_callers']):
                        continue
                    caller_ms = caller_dur_us / 1000.0
                    pct = (caller_ms / r['total_ms']) * 100 if r['total_ms'] > 0 else 0
                    clean_caller = _clean_kernel_name(caller_name)
                    f.write(f"   -> {clean_caller} ({caller_ms:.1f}ms - {pct:.0f}%)\n")