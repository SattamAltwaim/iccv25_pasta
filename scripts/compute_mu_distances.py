"""Best-effort reconstruction of the GMM adjacency files required for training.

The PASTA trainers (trainer_adj_predictor.py via the clipasso / npr / prosketch
data loaders) load two files that the authors did not release:

    assets/data/dataset_chair_preprocess/chairs_mu_distances.npy       [N, G, G]
    assets/data/dataset_chair_preprocess/chairs_mu_distances_part.npy  [N, G, G]

Both hold pairwise distances between the means (mu) of the G Gaussians that
SPAGHETTI decodes for each training shape, normalized to [0, 1] per shape
(the trainer uses `1 - dist` as a closeness/adjacency target). The "part"
variant aggregates Gaussians into parts (hierarchical clustering, mirroring
Sketch2Spaghetti's average-linkage grouping) and uses part-centroid distances
broadcast back to the individual Gaussians.

This script recomputes them from the released SPAGHETTI backbone
(occ_gmm_chairs_sym_hard) and the per-shape embeddings zh_0.npy that ship
with the SENS preprocessed chair dataset. It is a reconstruction of
unreleased preprocessing, not the authors' original code — expect it to be
faithful in spirit, but validate training results against the paper.

Usage (from the repo root, with checkpoints/data already in assets/):
    python scripts/compute_mu_distances.py [--num_parts 4] [--batch 64]
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster

import constants
from options import Options
from utils import train_utils


def pairwise_mu_distances(mus: torch.Tensor) -> torch.Tensor:
    # mus: [b, G, 3] -> [b, G, G], normalized to [0, 1] per shape
    dist = torch.cdist(mus, mus)
    max_per_shape = dist.flatten(1).max(dim=1).values.clamp(min=1e-8)
    return dist / max_per_shape[:, None, None]


def part_distances(mus_np: np.ndarray, dist_np: np.ndarray, num_parts: int) -> np.ndarray:
    # Cluster Gaussians into parts by mu position (average linkage, as in
    # Sketch2Spaghetti's grouping), then use part-centroid distances for
    # every pair of member Gaussians.
    out = np.zeros_like(dist_np)
    for i in range(mus_np.shape[0]):
        labels = fcluster(linkage(mus_np[i], 'average'), t=num_parts, criterion='maxclust')
        centroids = np.stack([mus_np[i][labels == c].mean(axis=0) for c in np.unique(labels)])
        cdist = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=-1)
        cmax = cdist.max()
        if cmax > 1e-8:
            cdist = cdist / cmax
        label_index = {c: k for k, c in enumerate(np.unique(labels))}
        idx = np.array([label_index[l] for l in labels])
        out[i] = cdist[idx[:, None], idx[None, :]]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spaghetti_tag', type=str, default='chairs_sym_hard')
    parser.add_argument('--zh_path', type=str, default='',
                        help='defaults to assets/data/dataset_chair_preprocess/<spaghetti_tag>/zh_0.npy')
    parser.add_argument('--out_dir', type=str, default=f'{constants.DATA_ROOT}dataset_chair_preprocess')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--num_parts', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    zh_path = args.zh_path or f'{constants.DATA_ROOT}dataset_chair_preprocess/{args.spaghetti_tag}/zh_0.npy'
    if not os.path.isfile(zh_path):
        raise FileNotFoundError(f'{zh_path} not found — download the SENS dataset_chair_preprocess first')

    opt = Options(tag=args.spaghetti_tag).load()
    opt.device = device
    spaghetti, *_ = train_utils.load_model(opt, device)
    spaghetti.eval()

    zh = torch.from_numpy(np.load(zh_path)).float()
    print(f'zh: {tuple(zh.shape)} from {zh_path}')

    all_dist, all_mus = [], []
    with torch.no_grad():
        for chunk in torch.split(zh, args.batch):
            _, gmms = spaghetti.occ_former.forward_mid([chunk.to(device)])
            mus = gmms[0][0].squeeze(1)  # [b, G, 3]
            all_mus.append(mus.cpu())
            all_dist.append(pairwise_mu_distances(mus).cpu())
    dist = torch.cat(all_dist).numpy()
    mus_np = torch.cat(all_mus).numpy()

    part = part_distances(mus_np, dist, args.num_parts)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(f'{args.out_dir}/chairs_mu_distances.npy', dist)
    np.save(f'{args.out_dir}/chairs_mu_distances_part.npy', part)
    print(f'saved {dist.shape} -> {args.out_dir}/chairs_mu_distances.npy')
    print(f'saved {part.shape} -> {args.out_dir}/chairs_mu_distances_part.npy')


if __name__ == '__main__':
    main()
