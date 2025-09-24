---
title: "Megablocks Only Test"
author: "uvnote"
theme: "light"
syntax_theme: "monokai"
show_line_numbers: true
collapse_code: false
custom_css: |
    #output-setup {
        overflow-x: auto;
    }
    .cell-output {
        overflow: scroll;
    }
    .cell-stdout {
        width: max-content;
        overflow: scroll;
    }
    .cell-stderr {
        width: max-content;
        overflow: scroll;
        max-height: 300px;
    }
---

```python id=nv
import subprocess

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

# No Kernels

First, we run the model without any custom kernels to get a reference point.

## Forward 

```python id=no_kernels
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "accelerate>=1.10.1",
#     "torch>=2.7.0",
#     "kernels==0.10.0",
#     "transformers@https://github.com/huggingface/transformers.git",
#     "ipdb>=0.13.13",
#     "matplotlib>=3.7.2",
#     "numpy>=1.24.3",
# ]
# ///

import torch
from transformers import GptOssForCausalLM, PreTrainedTokenizerFast, Mxfp4Config
import time
import torch.nn as nn
from kernels import register_kernel_mapping, Mode, LayerRepository, replace_kernel_forward_from_hub
import sys
import torch.profiler
import gc
import logging
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRMSNorm

# set to debug logging
logging.basicConfig(level=logging.INFO)

def reset_peak_memory_stats():
    """Clear CUDA cache and reset memory allocation counters."""
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

def get_memory_stats():
    """Get current and peak CUDA memory usage."""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "peak_gb": 0, "reserved_gb": 0}
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
    }

def override_kernel_layer_name(cls_name: str, value) -> bool:
    """Helper to dynamically override the kernel_layer_name in a model class."""
    for mod in sys.modules.values():
        if mod is None:
            continue
        obj = getattr(mod, cls_name, None)
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            setattr(obj, "kernel_layer_name", value)
            print(f"Overrode {cls_name}.kernel_layer_name to {value}")
            return True
    return False


# Init the model the normal way
model_id = "openai/gpt-oss-20b"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
quantization_config = Mxfp4Config(dequantize=True)



model = GptOssForCausalLM.from_pretrained(
    model_id,
    dtype="bfloat16",
    device_map="auto",
    use_kernels=False,
    quantization_config=quantization_config,
).eval()

messages = [
    {"role": "system", "content": "What is Tensor Parallelism?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="low",
).to("cuda")

max_tokens = 256

with torch.inference_mode():
    start_time = time.perf_counter()
    generated = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=None,
    )
    end_time = time.perf_counter()

print(tokenizer.decode(generated[0], skip_special_tokens=False))
print(f"Generation took {end_time - start_time:.2f} seconds")

```

## Forward and Backward

Next, we'll attempt to run a forward and backward pass without any custom kernels. This will likely run out of memory since the default implementation is not optimized for memory usage.

```python id=forward_and_backward_no_kernel timeout=1200
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "accelerate>=1.10.1",
#     "torch>=2.7.0",
#     "kernels==0.10.0",
#     "transformers@https://github.com/huggingface/transformers.git",
#     "ipdb>=0.13.13",
#     "matplotlib>=3.7.2",
#     "numpy>=1.24.3",
# ]
# ///

import torch
from transformers import GptOssForCausalLM, PreTrainedTokenizerFast, Mxfp4Config
import time
import torch.nn as nn
from kernels import register_kernel_mapping, Mode, LayerRepository, replace_kernel_forward_from_hub
import sys
import torch.profiler
import gc
import logging
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRMSNorm

# remove liger kernel for testing 
replace_kernel_forward_from_hub(GptOssRMSNorm, None)

# set to debug logging
logging.basicConfig(level=logging.INFO)

def reset_peak_memory_stats():
    """Clear CUDA cache and reset memory allocation counters."""
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

def get_memory_stats():
    """Get current and peak CUDA memory usage."""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "peak_gb": 0, "reserved_gb": 0}
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
    }

def override_kernel_layer_name(cls_name: str, value) -> bool:
    """Helper to dynamically override the kernel_layer_name in a model class."""
    for mod in sys.modules.values():
        if mod is None:
            continue
        obj = getattr(mod, cls_name, None)
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            setattr(obj, "kernel_layer_name", value)
            print(f"Overrode {cls_name}.kernel_layer_name to {value}")
            return True
    return False


# Init the model the normal way
model_id = "openai/gpt-oss-20b"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
quantization_config = Mxfp4Config(dequantize=True)

model = GptOssForCausalLM.from_pretrained(
    model_id,
    dtype="bfloat16",
    device_map="auto",
    use_kernels=False,
    quantization_config=quantization_config,
).eval()

messages = [
    {"role": "system", "content": "What is Tensor Parallelism?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="low",
).to("cuda")

max_tokens = 128  # Reduced to help with memory usage

# Clear memory before backward pass
reset_peak_memory_stats()
print(f"Pre-generation memory: {get_memory_stats()}")

# forward and backward pass
with torch.autograd.set_grad_enabled(True):
    start_time = time.perf_counter()
    generated = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=None,
    )
    end_time = time.perf_counter()
    print(tokenizer.decode(generated[0], skip_special_tokens=False))
    print(f"Generation took {end_time - start_time:.2f} seconds")
    print(f"Post-generation memory: {get_memory_stats()}")

    # Use gradient checkpointing to reduce memory usage
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing")

    # Reduce sequence length if needed for memory
    max_seq_len = 512  # Limit sequence length for backward pass
    if generated.size(1) > max_seq_len:
        print(f"Truncating sequence from {generated.size(1)} to {max_seq_len} tokens")
        full_sequence = generated[:, -max_seq_len:]
    else:
        full_sequence = generated

    # Get model outputs for the full sequence
    model.train()  # Enable dropout and other training behaviors

    try:
        outputs = model(
            input_ids=full_sequence,
            labels=full_sequence,  # This will compute loss internally
            return_dict=True
        )
        print(f"Post-forward memory: {get_memory_stats()}")

        # If model doesn't compute loss, compute it manually
        if outputs.loss is None:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = full_sequence[..., 1:].contiguous()

            # Use CrossEntropyLoss with ignore_index for padding tokens
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        else:
            loss = outputs.loss

        print(f"Loss: {loss.item():.4f}")

        # Clear intermediate tensors to save memory
        del outputs
        torch.cuda.empty_cache()

        # Perform backward pass with memory management
        print("Running backward pass...")
        print(f"Pre-backward memory: {get_memory_stats()}")

        loss.backward()
        print(f"Post-backward memory: {get_memory_stats()}")

    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM during forward/backward pass: {e}")
        print("Try reducing max_tokens or max_seq_len")
        raise

    # Calculate gradient statistics and print sample gradients
    total_norm = 0.0
    param_count = 0
    grad_samples = {}

    for name, p in model.named_parameters():
        if p.grad is not None:
            param_count += 1
            grad_norm = p.grad.data.norm(2).item()
            total_norm += grad_norm ** 2

            # Collect gradient statistics for key layers
            if any(key in name for key in ['embed', 'lm_head', 'mlp.up', 'mlp.down', 'self_attn.q_proj', 'norm']):
                grad_samples[name] = {
                    'norm': grad_norm,
                    'mean': p.grad.data.mean().item(),
                    'std': p.grad.data.std().item(),
                    'max': p.grad.data.max().item(),
                    'min': p.grad.data.min().item(),
                }

    total_norm = total_norm ** 0.5

    print(f"\nGradient norm: {total_norm:.4f}")
    print(f"Parameters with gradients: {param_count}")

    # Print sample gradients from important layers
    print("\nSample gradient statistics:")
    for i, (name, stats) in enumerate(list(grad_samples.items())[:10]):
        print(f"  {name[:60]:<60} | norm: {stats['norm']:.4e} | mean: {stats['mean']:.4e} | std: {stats['std']:.4e}")

    # Optional: zero gradients for next iteration
    model.zero_grad()
    model.eval()  # Switch back to eval mode


```

# Kernels

Next we can run with Megablocks kernels enabled.

### Forward

First, we run a forward pass with Megablocks kernels.

```python id=forward_only
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "accelerate>=1.10.1",
#     "torch>=2.7.0",
#     "kernels==0.10.0",
#     "transformers@https://github.com/huggingface/transformers.git",
#     "ipdb>=0.13.13",
#     "matplotlib>=3.7.2",
#     "numpy>=1.24.3",
# ]
# ///

import torch
from transformers import GptOssForCausalLM, PreTrainedTokenizerFast, Mxfp4Config
import time
import torch.nn as nn
from kernels import register_kernel_mapping, Mode, LayerRepository, replace_kernel_forward_from_hub
import sys
import torch.profiler
import gc
import logging
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRMSNorm


replace_kernel_forward_from_hub(GptOssRMSNorm, None)

# set to debug logging
logging.basicConfig(level=logging.INFO)

def reset_peak_memory_stats():
    """Clear CUDA cache and reset memory allocation counters."""
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

def get_memory_stats():
    """Get current and peak CUDA memory usage."""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "peak_gb": 0, "reserved_gb": 0}
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
    }

def override_kernel_layer_name(cls_name: str, value) -> bool:
    """Helper to dynamically override the kernel_layer_name in a model class."""
    for mod in sys.modules.values():
        if mod is None:
            continue
        obj = getattr(mod, cls_name, None)
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            setattr(obj, "kernel_layer_name", value)
            print(f"Overrode {cls_name}.kernel_layer_name to {value}")
            return True
    return False


# Init the model the normal way
model_id = "openai/gpt-oss-20b"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
quantization_config = Mxfp4Config(dequantize=True)



model = GptOssForCausalLM.from_pretrained(
    model_id,
    dtype="bfloat16",
    device_map="auto",
    use_kernels=True,
    quantization_config=quantization_config,
).eval()

messages = [
    {"role": "system", "content": "What is Tensor Parallelism?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="low",
).to("cuda")

max_tokens = 256

with torch.inference_mode():
    start_time = time.perf_counter()
    generated = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=None,
    )
    end_time = time.perf_counter()

print(tokenizer.decode(generated[0], skip_special_tokens=False))
print(f"Generation took {end_time - start_time:.2f} seconds")

```

## Forward and Backward

Next, we run a forward and backward pass with Megablocks kernels enabled. This should be more memory efficient and allow us to complete the backward pass without running out of memory.

```python id=forward_and_backward timeout=1200
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "accelerate>=1.10.1",
#     "torch>=2.7.0",
#     "kernels==0.10.0",
#     "transformers@https://github.com/huggingface/transformers.git",
#     "ipdb>=0.13.13",
#     "matplotlib>=3.7.2",
#     "numpy>=1.24.3",
# ]
# ///

import torch
from transformers import GptOssForCausalLM, PreTrainedTokenizerFast, Mxfp4Config
import time
import torch.nn as nn
from kernels import register_kernel_mapping, Mode, LayerRepository, replace_kernel_forward_from_hub
import sys
import torch.profiler
import gc
import logging
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRMSNorm

# remove liger kernel for testing 
replace_kernel_forward_from_hub(GptOssRMSNorm, None)

# set to debug logging
logging.basicConfig(level=logging.INFO)

def reset_peak_memory_stats():
    """Clear CUDA cache and reset memory allocation counters."""
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

def get_memory_stats():
    """Get current and peak CUDA memory usage."""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "peak_gb": 0, "reserved_gb": 0}
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
    }

def override_kernel_layer_name(cls_name: str, value) -> bool:
    """Helper to dynamically override the kernel_layer_name in a model class."""
    for mod in sys.modules.values():
        if mod is None:
            continue
        obj = getattr(mod, cls_name, None)
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            setattr(obj, "kernel_layer_name", value)
            print(f"Overrode {cls_name}.kernel_layer_name to {value}")
            return True
    return False


# Init the model the normal way
model_id = "openai/gpt-oss-20b"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
quantization_config = Mxfp4Config(dequantize=True)

model = GptOssForCausalLM.from_pretrained(
    model_id,
    dtype="bfloat16",
    device_map="auto",
    use_kernels=True,
    quantization_config=quantization_config,
).eval()

messages = [
    {"role": "system", "content": "What is Tensor Parallelism?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="low",
).to("cuda")

max_tokens = 128  # Reduced to help with memory usage

# Clear memory before backward pass
reset_peak_memory_stats()
print(f"Pre-generation memory: {get_memory_stats()}")

# forward and backward pass
with torch.autograd.set_grad_enabled(True):
    start_time = time.perf_counter()
    generated = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=None,
    )
    end_time = time.perf_counter()
    print(tokenizer.decode(generated[0], skip_special_tokens=False))
    print(f"Generation took {end_time - start_time:.2f} seconds")
    print(f"Post-generation memory: {get_memory_stats()}")

    # Use gradient checkpointing to reduce memory usage
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing")

    # Reduce sequence length if needed for memory
    max_seq_len = 512  # Limit sequence length for backward pass
    if generated.size(1) > max_seq_len:
        print(f"Truncating sequence from {generated.size(1)} to {max_seq_len} tokens")
        full_sequence = generated[:, -max_seq_len:]
    else:
        full_sequence = generated

    # Get model outputs for the full sequence
    model.train()  # Enable dropout and other training behaviors

    try:
        outputs = model(
            input_ids=full_sequence,
            labels=full_sequence,  # This will compute loss internally
            return_dict=True
        )
        print(f"Post-forward memory: {get_memory_stats()}")

        # If model doesn't compute loss, compute it manually
        if outputs.loss is None:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = full_sequence[..., 1:].contiguous()

            # Use CrossEntropyLoss with ignore_index for padding tokens
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        else:
            loss = outputs.loss

        print(f"Loss: {loss.item():.4f}")

        # Clear intermediate tensors to save memory
        del outputs
        torch.cuda.empty_cache()

        # Perform backward pass with memory management
        print("Running backward pass...")
        print(f"Pre-backward memory: {get_memory_stats()}")

        loss.backward()
        print(f"Post-backward memory: {get_memory_stats()}")

    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM during forward/backward pass: {e}")
        print("Try reducing max_tokens or max_seq_len")
        raise

    # Calculate gradient statistics and print sample gradients
    total_norm = 0.0
    param_count = 0
    grad_samples = {}

    for name, p in model.named_parameters():
        if p.grad is not None:
            param_count += 1
            grad_norm = p.grad.data.norm(2).item()
            total_norm += grad_norm ** 2

            # Collect gradient statistics for key layers
            if any(key in name for key in ['embed', 'lm_head', 'mlp.up', 'mlp.down', 'self_attn.q_proj', 'norm']):
                grad_samples[name] = {
                    'norm': grad_norm,
                    'mean': p.grad.data.mean().item(),
                    'std': p.grad.data.std().item(),
                    'max': p.grad.data.max().item(),
                    'min': p.grad.data.min().item(),
                }

    total_norm = total_norm ** 0.5

    print(f"\nGradient norm: {total_norm:.4f}")
    print(f"Parameters with gradients: {param_count}")

    # Print sample gradients from important layers
    print("\nSample gradient statistics:")
    for i, (name, stats) in enumerate(list(grad_samples.items())[:10]):
        print(f"  {name[:60]:<60} | norm: {stats['norm']:.4e} | mean: {stats['mean']:.4e} | std: {stats['std']:.4e}")

    # Optional: zero gradients for next iteration
    model.zero_grad()
    model.eval()  # Switch back to eval mode


```
