"""Shared benchmark: the same synthetic test on every PC of the NeuroState epilepsy study.

No files are read. Measures:
  - EEGNet training (MicroStateNet(use_ms=False), input 19 x 512, Adam, full 32-bit,
    cudnn.deterministic=True, benchmark=False) at batch 64 and 128: windows per second, peak GPU memory;
  - inference at batch 512: windows per second;
  - CPU: MNE FIR filters 2-20 Hz and 1-40 Hz on 19 channels x 10 min at 256 Hz, plus resampling to 128 Hz,
    repeated 20 times: seconds per recording-equivalent.

Prints one summary table and writes bench_<hostname>.json to the current folder (or --out).

Usage (from the repo root):
    python -m experiments.epilepsy.bench_gpu                 # full, on cuda
    python -m experiments.epilepsy.bench_gpu --quick         # 100 training steps, fewer repeats
    python -m experiments.epilepsy.bench_gpu --device cpu
"""
import argparse
import json
import os
import platform
import socket
import subprocess
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from experiments.models import MicroStateNet

N_CHANS, N_TIMES = 19, 512          # 4-s windows at 128 Hz
CPU_SFREQ, CPU_MINUTES = 256.0, 10  # the CPU test signal


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _driver_version():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu_name():
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or None


def system_info(device):
    info = {
        "hostname": socket.gethostname(),
        "date": datetime.now().isoformat(timespec="seconds"),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": str(device),
        "cpu": _cpu_name(),
        "cpu_logical_cores": os.cpu_count(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(gpu=props.name, gpu_memory_gb=round(props.total_memory / 1024 ** 3, 2), driver=_driver_version())
    try:
        import mne
        info["mne"] = mne.__version__
    except ImportError:
        info["mne"] = None
    return info


def bench_train(device, batch, steps, warmup=20, seed=0):
    torch.manual_seed(seed)
    model = MicroStateNet(n_chans=N_CHANS, n_times=N_TIMES, n_classes=2, use_ms=False).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator(device="cpu").manual_seed(seed)
    pool = 8  # a few distinct batches, made once, so data making is not timed
    xs = torch.randn(pool, batch, N_CHANS, N_TIMES, generator=g).to(device)
    ys = torch.randint(0, 2, (pool, batch), generator=g).to(device)
    model.train()

    def step(i):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xs[i % pool]), ys[i % pool])
        loss.backward()
        opt.step()
        return loss

    for i in range(warmup):
        step(i)
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for i in range(steps):
        loss = step(i)
    _sync(device)
    secs = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else None
    return {"batch": batch, "steps": steps, "seconds": round(secs, 3),
            "windows_per_s": round(steps * batch / secs, 1),
            "peak_gpu_mb": round(peak, 1) if peak is not None else None,
            "final_loss": round(loss.item(), 4)}


@torch.no_grad()
def bench_infer(device, batch=512, steps=200, warmup=10, seed=0):
    torch.manual_seed(seed)
    model = MicroStateNet(n_chans=N_CHANS, n_times=N_TIMES, n_classes=2, use_ms=False).to(device).eval()
    x = torch.randn(batch, N_CHANS, N_TIMES, generator=torch.Generator().manual_seed(seed)).to(device)
    for _ in range(warmup):
        model(x)
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        model(x)
    _sync(device)
    secs = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else None
    return {"batch": batch, "steps": steps, "seconds": round(secs, 3),
            "windows_per_s": round(steps * batch / secs, 1),
            "peak_gpu_mb": round(peak, 1) if peak is not None else None}


def bench_cpu_filters(repeats=20, seed=0):
    import mne
    mne.set_log_level("ERROR")
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((N_CHANS, int(CPU_MINUTES * 60 * CPU_SFREQ))) * 20e-6  # volts, EEG-like scale
    times = {"filter_2_20": [], "filter_1_40": [], "resample_128": []}
    for _ in range(repeats):
        t0 = time.perf_counter()
        mne.filter.filter_data(data, CPU_SFREQ, 2.0, 20.0, method="fir", n_jobs=1, verbose=False)
        t1 = time.perf_counter()
        mne.filter.filter_data(data, CPU_SFREQ, 1.0, 40.0, method="fir", n_jobs=1, verbose=False)
        t2 = time.perf_counter()
        mne.filter.resample(data, up=1.0, down=CPU_SFREQ / 128.0, npad="auto", n_jobs=1, verbose=False)
        t3 = time.perf_counter()
        times["filter_2_20"].append(t1 - t0)
        times["filter_1_40"].append(t2 - t1)
        times["resample_128"].append(t3 - t2)
    out = {k: round(float(np.mean(v)), 4) for k, v in times.items()}
    out["seconds_per_recording"] = round(sum(out[k] for k in times), 4)
    out.update(repeats=repeats, channels=N_CHANS, minutes=CPU_MINUTES, sfreq=CPU_SFREQ)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--quick", action="store_true", help="100 training steps and fewer repeats, for a fast check")
    ap.add_argument("--out", default=".", help="folder for bench_<hostname>.json (default: current folder)")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run with --device cpu, or fix the PyTorch / driver install.")
    device = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False  # full 32-bit
    torch.backends.cudnn.allow_tf32 = False

    train_steps = 100 if args.quick else 800
    infer_steps = 50 if args.quick else 200
    cpu_repeats = 3 if args.quick else 20

    res = {"info": system_info(device), "quick": args.quick, "settings": {
        "model": "MicroStateNet(use_ms=False) = EEGNet", "input": [N_CHANS, N_TIMES], "optimizer": "Adam lr 1e-3 wd 1e-3",
        "precision": "float32 (TF32 off)", "cudnn_deterministic": True, "cudnn_benchmark": False}}
    res["train"] = [bench_train(device, b, train_steps) for b in (64, 128)]
    res["infer"] = bench_infer(device, 512, infer_steps)
    res["cpu"] = bench_cpu_filters(cpu_repeats)

    i = res["info"]
    print(f"\nNeuroState benchmark  {i['hostname']}  {i['date']}")
    print(f"{i.get('gpu', 'CPU only')} | driver {i.get('driver')} | torch {i['torch']} CUDA {i['cuda']} | "
          f"Python {i['python']} | {i['cpu']}")
    print(f"{'test':<34}{'windows/s':>12}{'peak MB':>10}{'seconds':>10}")
    for t in res["train"]:
        print(f"{'train EEGNet, batch ' + str(t['batch']) + ', ' + str(t['steps']) + ' steps':<34}"
              f"{t['windows_per_s']:>12.0f}{(t['peak_gpu_mb'] or 0):>10.0f}{t['seconds']:>10.2f}")
    t = res["infer"]
    print(f"{'inference, batch 512':<34}{t['windows_per_s']:>12.0f}{(t['peak_gpu_mb'] or 0):>10.0f}{t['seconds']:>10.2f}")
    c = res["cpu"]
    print(f"CPU per 10-min 19-ch recording (mean of {c['repeats']}): 2-20 Hz {c['filter_2_20']:.3f} s, "
          f"1-40 Hz {c['filter_1_40']:.3f} s, resample {c['resample_128']:.3f} s, total {c['seconds_per_recording']:.3f} s")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"bench_{i['hostname']}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"written: {path}")


if __name__ == "__main__":
    main()
