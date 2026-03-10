# llmsys_f25_hw4

Public repository for Assignment 4 of 11-868 LLM Systems.

## Problem 3: Fused Kernels Speed Comparison

After integrating the improved CUDA kernels into the transformer, we compared training speed for one epoch with and without fused softmax/layernorm:


| Use Fused Kernel | Epoch Time | Speedup |
| ---------------- | ---------- | ------- |
| False            | 1:02:34    | 1.00×   |
| True             | yyy.yy     | z.zz×   |


```bash
python project/run_machine_translation.py --use-fused-kernel False
python project/run_machine_translation.py --use-fused-kernel True
```

*According to Amdahl's law, only modest speedup is expected, but a speedup ≈ 1.1× should be observed.*