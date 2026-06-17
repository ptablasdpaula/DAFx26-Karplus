"""GPU smoke test: build the KS synth and run forward+backward on CUDA for both
implementations. Time-domain (tKSA) exercises philtorch.lpv -> torchlpc on CUDA
(numba path); frequency-sampling (fKSA) exercises the FFT path. Run from repo root:
    PYTHONPATH=. .pixi/envs/cuda/bin/python experiments/jobs/gpu_smoke.py
"""
import torch
from src.synths.synth import Synth, SynthConfig
from src.synths.ddsp import Implementation

assert torch.cuda.is_available(), "CUDA not available!"
dev = torch.device("cuda")
print("device:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| cuda", torch.version.cuda)
try:
    from torchlpc import EXTENSION_LOADED
    print("torchlpc EXTENSION_LOADED:", EXTENSION_LOADED, "(False => numba CUDA path)")
except Exception as e:
    print("torchlpc import note:", e)

fs, num_samples = 16000, 64000
ok = True
for impl, name in [(Implementation.TIME_DOMAIN, "tKSA"), (Implementation.FREQUENCY_SAMPLING, "fKSA")]:
    cfg = SynthConfig(num_samples=num_samples, fs=fs, implementation=impl)
    model = Synth(cfg).to(dev)
    params = {
        'exists':         torch.ones(1, 4, device=dev),
        'time':           torch.tensor([[0.0, 0.25, 0.5, 0.75]], device=dev),
        'f0':             torch.tensor([[220., 330., 440., 550.]], device=dev),
        'burst_gain':     torch.full((1, 4), 0.8, device=dev, requires_grad=True),
        'pluck_position': torch.full((1, 4), 0.5, device=dev),
        'dynamic_level':  torch.full((1, 4), 0.5, device=dev),
        'a1':             torch.full((1, 4), 0.5, device=dev),
        'decay':          torch.full((1, 4), 0.995, device=dev),
    }
    try:
        y, _ = model(params)
        loss = y.float().pow(2).mean()
        loss.backward()
        g = params['burst_gain'].grad
        nan = bool(torch.isnan(y).any() or (g is not None and torch.isnan(g).any()))
        print(f"{name}: forward {tuple(y.shape)} on {y.device}, backward OK, grad_ok={g is not None and not nan}, output_nan={bool(torch.isnan(y).any())}")
        ok = ok and (g is not None) and not nan
    except Exception as e:
        ok = False
        print(f"{name}: FAILED -> {type(e).__name__}: {str(e)[:200]}")

print("SMOKE_OK" if ok else "SMOKE_FAILED")
