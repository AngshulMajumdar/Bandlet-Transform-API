import os
import sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from bandlet_tf import BandletConfig, BandletTransform, BandletPackedLayer


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    T = BandletTransform(BandletConfig(levels=2, block_size=8, device=device))
    x = torch.rand(2, 1, 64, 64, device=T.device, dtype=T.dtype)

    enc = T.encode(x)
    xrec = T.reconstruct(enc)
    rel = (x - xrec).norm() / x.norm()
    print('relative_pr_error', float(rel))

    vec, meta = T.encode_packed(x)
    xrec2 = T.decode_packed(vec, meta)
    rel2 = (x - xrec2).norm() / x.norm()
    print('relative_pack_error', float(rel2))

    packed_layer = BandletPackedLayer(BandletConfig(levels=2, block_size=8, device=device))
    vec3, meta3 = packed_layer(x)
    print('packed_shape', tuple(vec3.shape))
    print('levels', meta3['meta']['levels'])


if __name__ == '__main__':
    main()
