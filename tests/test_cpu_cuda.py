import torch
from bandlet_tf import BandletConfig, BandletTransform


def test_cpu_cuda_parity():
    if not torch.cuda.is_available():
        return
    x = torch.rand(1, 1, 64, 64)
    t_cpu = BandletTransform(BandletConfig(levels=2, block_size=8, device='cpu'))
    t_gpu = BandletTransform(BandletConfig(levels=2, block_size=8, device='cuda'))
    enc_cpu = t_cpu.encode(x)
    enc_gpu = t_gpu.encode(x.cuda())
    assert torch.allclose(enc_cpu.approx, enc_gpu.approx.cpu(), atol=1e-5, rtol=1e-5)
    for trip_cpu, trip_gpu in zip(enc_cpu.detail_bands, enc_gpu.detail_bands):
        for sub_cpu, sub_gpu in zip(trip_cpu, trip_gpu):
            assert torch.allclose(sub_cpu.packed.coeffs, sub_gpu.packed.coeffs.cpu(), atol=1e-5, rtol=1e-5)
    xrec_cpu = t_cpu.reconstruct(enc_cpu)
    xrec_gpu = t_gpu.reconstruct(enc_gpu).cpu()
    assert torch.allclose(xrec_cpu, xrec_gpu, atol=1e-5, rtol=1e-5)
