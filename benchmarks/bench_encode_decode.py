import time
import torch
from bandlet_tf import BandletConfig, BandletTransform


def run_once(size=256, batch=4, levels=2, block_size=8, device='cuda'):
    t = BandletTransform(BandletConfig(levels=levels, block_size=block_size, device=device))
    x = torch.rand(batch, 1, size, size, device=t.device, dtype=t.dtype)
    if t.device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    enc = t.encode(x)
    if t.device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    _ = t.reconstruct(enc)
    if t.device.type == 'cuda':
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    print({
        'device': str(t.device),
        'size': size,
        'batch': batch,
        'levels': levels,
        'block_size': block_size,
        'encode_s': t1 - t0,
        'decode_s': t2 - t1,
    })


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_once(device=dev)
