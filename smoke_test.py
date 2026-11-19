"""Check output dimensions, finite losses and backward propagation."""
import torch
from Diff_MT_model import DiffMT, ModelConfig, FocalLoss


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    channels = {'GSR': 1, 'EEG': 4, 'BVP': 1}
    inputs = {'GSR': torch.randn(4,1,37), 'EEG': torch.randn(4,4,93), 'BVP': torch.randn(4,1,64)}
    cfg = ModelConfig(d_model=16, aligned_steps=19, heads=4, window=8, hop=5, strides=(1,2,4))
    for variant in ('full','no_dmfe','no_gce','no_dwf','early'):
        model = DiffMT(channels, 3, cfg, variant)
        logits = model(inputs)
        assert logits.shape == (4,3) and torch.isfinite(logits).all()
        loss = FocalLoss()(logits, torch.tensor([0,1,2,1]))
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        print(variant, 'OK')
    model = DiffMT({'GSR':1}, 3, cfg)
    assert model({'GSR':inputs['GSR']}).shape == (4,3)
    print('unimodal OK')

if __name__ == '__main__':
    main()
