"""Cross-validation training on prepared multimodal NPZ arrays."""
from __future__ import annotations
import argparse
import copy
import csv
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from Diff_MT_model import DiffMT, ModelConfig, FocalLoss

MODALITIES = {
    'wesad': ('GSR', 'BVP', 'EMG', 'ECG', 'RESP'),
    'deap': ('EEG', 'EMG', 'EOG', 'GSR'),
    'mocas': ('EEG_POW', 'EEG', 'BVP', 'EAR', 'GSR'),
    'cogload': ('GSR', 'SKT', 'HR', 'ACC', 'IBI'),
}

class BiosignalDataset(Dataset):
    def __init__(self, path, modalities, label_key='labels'):
        with np.load(path, allow_pickle=False) as data:
            self.labels = np.asarray(data[label_key])
            if self.labels.ndim != 1 or not np.issubdtype(self.labels.dtype, np.integer):
                raise ValueError('Labels must be 1D integer class indices')
            if not np.array_equal(np.unique(self.labels), np.arange(len(np.unique(self.labels)))):
                raise ValueError('Labels must be contiguous class indices beginning at zero')
            self.labels = self.labels.astype(np.int64)
            self.groups = np.array(data['groups']) if 'groups' in data else None
            if self.groups is not None and self.groups.shape != self.labels.shape:
                raise ValueError('groups must contain one ID per sample')
            self.inputs = {}
            for m in modalities:
                x = np.asarray(data[m], dtype=np.float32)
                if x.ndim == 2:
                    x = x[:, None, :]
                if x.ndim != 3 or x.shape[0] != len(self.labels) or min(x.shape[1:]) < 1:
                    raise ValueError(f'{m} must have shape [N,C,T] or [N,T]')
                if not np.isfinite(x).all():
                    raise ValueError(f'{m} contains nonfinite samples')
                self.inputs[m] = torch.from_numpy(x)
        self.channels = {m: x.shape[1] for m,x in self.inputs.items()}
        self.num_classes = len(np.unique(self.labels))
        if self.num_classes < 2:
            raise ValueError('Need at least two classes')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {m: x[i] for m,x in self.inputs.items()}, int(self.labels[i])


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(dataset, indices, batch_size, shuffle=False):
    return DataLoader(Subset(dataset, indices.tolist()), batch_size=batch_size, shuffle=shuffle)


def predict(model, batches, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in batches:
            p = model({m: z.to(device) for m,z in x.items()}).softmax(-1)
            probs.append(p.cpu().numpy()); labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def fit(dataset, tr, va, te, args, cfg, variant, channels, device, checkpoint):
    train_loader = loader(dataset, tr, args.batch_size, True)
    val_loader = loader(dataset, va, args.batch_size)
    test_loader = loader(dataset, te, args.batch_size)
    model = DiffMT(channels, dataset.num_classes, cfg, variant).to(device)
    counts = np.bincount(dataset.labels[tr], minlength=dataset.num_classes)
    if not counts.all():
        raise ValueError('Training split is missing a class; revise splits')
    weights = None
    if args.class_weights == 'balanced':
        weights = torch.tensor(counts.sum() / (len(counts)*counts), dtype=torch.float32, device=device)
    loss_fn = FocalLoss(args.gamma, weights).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.98), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best, state, best_epoch = -1., None, 0
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model({m: z.to(device) for m,z in x.items()})
            loss = loss_fn(logits, y.to(device))
            loss.backward(); opt.step()
        scheduler.step()
        p, y = predict(model, val_loader, device)
        score = f1_score(y, p.argmax(1), labels=np.arange(dataset.num_classes), average='macro', zero_division=0)
        if score > best:
            best, best_epoch = score, epoch+1
            state = copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    torch.save({'model_state': model.state_dict(), 'channels': channels,
                'num_classes': dataset.num_classes, 'variant': variant,
                'config': vars(cfg), 'best_epoch': best_epoch}, checkpoint)
    return predict(model, test_loader, device)


def split_indices(dataset, args):
    y = dataset.labels
    if args.grouped:
        if dataset.groups is None:
            raise ValueError('--grouped requires groups array')
        outer = StratifiedGroupKFold(args.folds, shuffle=True, random_state=args.seed)
        for pool, te in outer.split(y, y, dataset.groups):
            # Group-disjoint validation with up to five inner folds.
            n = min(5, len(np.unique(dataset.groups[pool])))
            if n < 2:
                raise ValueError('Insufficient training groups for validation')
            inner = StratifiedGroupKFold(n, shuffle=True, random_state=args.seed)
            tr, va = next(inner.split(y[pool], y[pool], dataset.groups[pool]))
            yield pool[tr], pool[va], te
    else:
        outer = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
        for pool, te in outer.split(y, y):
            inner = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
            tr, va = next(inner.split(y[pool], y[pool]))
            yield pool[tr], pool[va], te


def run(args):
    mods = tuple(args.modalities) if args.modalities else MODALITIES[args.dataset]
    dataset = BiosignalDataset(args.data, mods, args.label_key)
    args.batch_size = args.batch_size or (1024 if args.dataset in {'deap','mocas'} else 512)
    if args.epochs < 1 or args.folds < 2:
        raise ValueError('Require positive epochs and at least two folds')
    cfg = ModelConfig(d_model=args.dim, aligned_steps=args.steps, heads=args.heads,
                      window=args.window, hop=args.hop, strides=tuple(args.strides))
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps({'training': vars(args), 'model': vars(cfg)}, indent=2))
    rows = []
    for fold, (tr, va, te) in enumerate(split_indices(dataset, args), 1):
        np.savez(out/f'fold_{fold:02d}_indices.npz', train=tr, val=va, test=te)
        for variant in args.variants:
            seed_all(args.seed+fold)
            if variant == 'late':
                # Train unimodal classifiers independently, then average posteriors.
                ps = []
                for m in mods:
                    p,y = fit(dataset,tr,va,te,args,cfg,'full',{m:dataset.channels[m]},device,out/f'fold_{fold:02d}_late_{m}.pt')
                    ps.append(p)
                p = np.mean(ps, axis=0)
            else:
                p,y = fit(dataset,tr,va,te,args,cfg,variant,dataset.channels,device,out/f'fold_{fold:02d}_{variant}.pt')
            pred = p.argmax(1)
            row = {'fold':fold,'variant':variant,'accuracy':100*accuracy_score(y,pred),
                   'macro_f1':100*f1_score(y,pred,labels=np.arange(dataset.num_classes),average='macro',zero_division=0)}
            rows.append(row); print(row, flush=True)
            np.savez(out/f'fold_{fold:02d}_{variant}_predictions.npz', labels=y, probabilities=p, indices=te)
    with (out/'fold_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary=[]
    for variant in args.variants:
        subset=[r for r in rows if r['variant']==variant]
        row={'variant':variant}
        for key in ('accuracy','macro_f1'):
            vals=[r[key] for r in subset]; row[key+'_mean']=float(np.mean(vals)); row[key+'_std']=float(np.std(vals,ddof=1))
        summary.append(row)
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)


def parser(default_variants=('full',)):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True); p.add_argument('--dataset',required=True,choices=MODALITIES)
    p.add_argument('--label-key',default='labels'); p.add_argument('--modalities',nargs='+')
    p.add_argument('--variants',nargs='+',choices=['full','no_dmfe','no_gce','no_dwf','early','late'],default=list(default_variants))
    p.add_argument('--epochs',type=int,default=100); p.add_argument('--folds',type=int,default=10)
    p.add_argument('--batch-size',type=int); p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--gamma',type=float,default=2.); p.add_argument('--class-weights',choices=['balanced','none'],default='balanced')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--grouped',action='store_true')
    p.add_argument('--dim',type=int,default=64); p.add_argument('--steps',type=int,default=64); p.add_argument('--heads',type=int,default=4)
    p.add_argument('--window',type=int,default=16); p.add_argument('--hop',type=int,default=8); p.add_argument('--strides',nargs='+',type=int,default=[1,2,4])
    p.add_argument('--device'); p.add_argument('--output',default='results')
    return p

if __name__ == '__main__':
    run(parser().parse_args())
