"""Evaluate the TAPNext point tracker on sonar image sequences from the
Aracati dataset.

For each window of `n` consecutive frames the script:

  1. selects sparse query points on the first frame (Sobel keypoints, kept
     spread across the image by `GridPointManager` and carried over between
     windows);
  2. tracks those points across the window with TAPNext;
  3. fits a rigid (rotation + translation) transform for every frame pair with
     RANSAC;
  4. converts that transform into an ego-motion estimate (dx, dy, dyaw) and
     compares it against the ground-truth pose delta from the dataset.

One CSV row per evaluated frame pair is appended to a results file under
`--save_path`. The filename encodes every setting that materially changes a
run (model, dataset, seed, window geometry, RANSAC threshold, refinement
options), so runs with different settings never overwrite each other.

Query points
    The tracked set is the persistent Sobel keypoints managed by
    `GridPointManager`: they are re-detected every window, merged into the
    manager, and the survivors are carried into the next window. With
    `--fallback_grid_points`, a regular grid of auxiliary points is appended
    for the current window only whenever fewer than `MIN_PERSISTENT_POINTS`
    persistent points survive selection; those grid points are never stored in
    the manager. Both point sources drop points closer than
    `--min_origin_dist` pixels to the sonar origin (0 disables the filter).

Refinement (--refine)
    Instead of trusting the tracker's raw point estimates, the rigid transform
    can be nudged so that each tracked point's neighborhood best matches its
    frame-0 dense-feature response (features come from a pruned ResNet50 stem).
    Refinement is selective: it only runs on frame pairs whose raw RANSAC
    inlier ratio falls below `--refine_inlier_threshold`, since a high ratio
    already indicates a well-explained fit. By default only the persistent
    Sobel keypoints take part in the optimization; pass
    `--no-optimize_sobel_only` to include the fallback grid points as well.
    The refinement path is GPU-resident: feature maps, correlation patches and
    the rigid-transform optimizer all run on the extractor's device, and the
    only device->host transfer per frame pair is the final (N, 2) refined point
    set handed to skimage's (CPU/numpy) RANSAC.

The torch/numpy seed is set by `--seed` and recorded in the results filename.

Run directly, e.g.:
    python eval/isopot_tracker.py --n 5 --w 0
"""

import os
import queue
import argparse
import warnings
import threading

import cv2
import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import ransac
from skimage.transform import EuclideanTransform

from tqdm import tqdm
from utils.model_wrappers import TAPNextWrapper
from utils.datasets import ISOPoTDataset
from utils.utils import (
    sobel_keypoints,
    find_transformation_between_predictions,
    estimate_motion,
    get_relative_pose,
    get_pose,
)

warnings.filterwarnings("ignore")

# Recorded in the results filename so evaluation runs stay identifiable.
MODEL_NAME = "TAPNext"
DATASET_NAME = "aracati"

# Default dataset locations. These point at the gitignored `local/` folder
# (not published with the repo) — put your own data there, or override on
# the command line with --dataset_path / --mask_path.
DEFAULT_DATASET_PATH = "local/datasets/aracati"
DEFAULT_MASK_PATH = "local/masks/sonar_coverage_mask.png"

# Below this many surviving persistent keypoints, a window is considered too
# sparse to fit a reliable transform from Sobel points alone, and the fallback
# grid kicks in (only with --fallback_grid_points).
MIN_PERSISTENT_POINTS = 10


class WindowPrefetcher:
    """Background-thread prefetcher for `dataset[i]` window loads.

    The main loop iterates windows at known indices, so upcoming windows can
    be loaded from disk while the GPU is busy processing the current one.
    A daemon thread walks the index list, calls `dataset[idx]`, and puts the
    result into a bounded queue (`depth` controls how many windows may be
    loaded ahead; the thread blocks once the queue is full, capping memory
    use at ~depth windows).

    Threading (rather than multiprocessing) is enough here: the load path is
    dominated by cv2.imread / file I/O, which release the GIL, so the loader
    genuinely overlaps with the torch/numpy work on the main thread. It also
    means no pickling of the dataset object and no per-window IPC cost.

    Exceptions raised inside `dataset[idx]` are captured and re-raised on the
    main thread from `__next__`, so failures surface exactly as they would
    have without prefetching.

    Usage:
        prefetcher = WindowPrefetcher(dataset, indices, depth=2)
        for idx, (images, poses, kpts) in prefetcher:
            ...
    """

    _SENTINEL = object()

    def __init__(self, dataset, indices, depth=2):
        self.dataset = dataset
        self.indices = list(indices)
        self.queue = queue.Queue(maxsize=max(1, depth))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        try:
            for idx in self.indices:
                if self._stop.is_set():
                    return
                try:
                    item = self.dataset[idx]
                except BaseException as e:  # re-raised on the main thread
                    self.queue.put((idx, None, e))
                    return
                self.queue.put((idx, item, None))
        finally:
            self.queue.put(self._SENTINEL)

    def __iter__(self):
        return self

    def __next__(self):
        out = self.queue.get()
        if out is self._SENTINEL:
            raise StopIteration
        idx, item, err = out
        if err is not None:
            raise err
        return idx, item

    def close(self):
        """Stop the loader thread and drain the queue so it can exit."""
        self._stop.set()
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="Evaluate the TAPNext point tracker on Aracati sonar sequences.")

    arg_parser.add_argument(
        "--save_path",
        type=str,
        default="eval_runs/",
        help="Directory the evaluation results are written to (created if missing).",
    )
    arg_parser.add_argument(
        "--dataset_path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Root directory of the Aracati sequence.",
    )
    arg_parser.add_argument(
        "--mask_path",
        type=str,
        default=DEFAULT_MASK_PATH,
        help=(
            "Grayscale image marking the valid sonar coverage area. Non-zero "
            "pixels are valid; the mask is eroded before use so points never "
            "sit right on the coverage boundary."
        ),
    )
    arg_parser.add_argument(
        "--fine_tuned_weights",
        type=str,
        default=None,
        help="Path to fine-tuned TAPNext weights (default: the released checkpoint).",
    )
    arg_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Random seed for torch, numpy and the GridPointManager's capacity "
            "drops. Also recorded in the results filename, so runs that differ "
            "only by seed do not overwrite each other."
        ),
    )
    arg_parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of frames per evaluation window.",
    )
    arg_parser.add_argument(
        "--w",
        type=int,
        default=3,
        help=(
            "Index of the warmup frame within the window. Frames up to and "
            "including w only give the tracker time to settle; pairs are "
            "evaluated from w+1 onwards, and every pairwise transform is "
            "composed through this frame as a shared reference."
        ),
    )
    arg_parser.add_argument(
        "--refine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Refine the transformation using feature-correlation optimization. "
            "Refinement is applied selectively, only to frame pairs whose RANSAC "
            "inlier ratio is below --refine_inlier_threshold."
        ),
    )
    arg_parser.add_argument(
        "--refine_inlier_threshold",
        "--rit",
        type=float,
        default=0.9,
        help=(
            "Only refine frame pairs whose RANSAC inlier ratio is strictly below this "
            "value; pairs at or above it are considered well estimated and keep the raw "
            "transform. Use a value above 1 to refine every pair unconditionally."
        ),
    )
    arg_parser.add_argument(
        "--refine_steps",
        type=int,
        default=10,
        help="LBFGS iterations per refinement pass.",
    )
    arg_parser.add_argument(
        "--refine_patch_radius",
        type=int,
        default=10,
        help=(
            "Correlation patch radius for the refinement, in pixels. Also bounds "
            "how far refinement can move a point, since coordinates sampled "
            "outside the patch get no gradient (memory grows as radius^2)."
        ),
    )
    arg_parser.add_argument(
        "--refine_y_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Restrict the refinement optimizer's translation to the Y axis "
            "(rotation about the sonar origin + surge only), matching "
            "utils.BottomCenterRotationTransform exactly. Default keeps full "
            "(tx, ty) translation, still pivoting about the sonar origin."
        ),
    )
    arg_parser.add_argument(
        "--ransac_threshold",
        "--rtr",
        type=float,
        default=3.0,
        help="RANSAC residual threshold (pixels) for transformation estimation.",
    )
    arg_parser.add_argument(
        "--min_origin_dist",
        type=float,
        default=0.0,
        help=(
            "Drop query points (Sobel keypoints and fallback grid points) closer "
            "than this many pixels to the sonar origin at the bottom-center of "
            "the image, where rotation moves pixels the least (0 = keep all points)."
        ),
    )
    arg_parser.add_argument(
        "--fallback_grid_points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            f"Fall back to a regular grid of query points whenever fewer than "
            f"{MIN_PERSISTENT_POINTS} persistent Sobel keypoints survive selection "
            "in a window. The grid is used for that window only and is never "
            "stored in the GridPointManager, so it does not displace the "
            "persistent keypoints in later windows."
        ),
    )
    arg_parser.add_argument(
        "--optimize_sobel_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the --refine feature-correlation optimization only on the "
            "persistent Sobel keypoints, excluding fallback grid points "
            "(use --no-optimize_sobel_only to optimize on all query points)."
        ),
    )
    arg_parser.add_argument(
        "--prefetch",
        type=int,
        default=2,
        help="Number of dataset windows to prefetch on a background thread (0 = disable prefetching).",
    )

    return arg_parser.parse_args()


class ChannelMeanReduce(nn.Module):
    """Reduce channel count by averaging consecutive groups of channels.

    Cheaper and parameter-free compared to a 1x1 convolution, which matters
    because these features are only ever correlated against each other, never
    trained.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        assert in_ch % out_ch == 0, "out_ch must divide in_ch"
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        return x.view(B, self.out_ch, C // self.out_ch, H, W).mean(dim=2)


class PretrainedFeatureExtractor(nn.Module):
    """Shallow ResNet50 stem used to get dense per-pixel features for the
    feature-correlation refinement step (see `--refine`)."""

    def __init__(self, output_channels=256):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Only the stem + first residual block: keeps spatial resolution high,
        # which matters since we sample features at exact pixel locations later.
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.layer1 = resnet.layer1

        self.reduce = ChannelMeanReduce(in_ch=256, out_ch=output_channels)

        # Buffers (not Parameters): move with .to(device), never optimized.
        # These are the statistics the torchvision ResNet50 weights were
        # trained with; feeding raw [0, 1] images instead leaves every
        # activation off-distribution.
        self.register_buffer(
            "imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def prepare_input(self, image, device=None):
        """Grayscale (H, W) uint8 frame -> (1, 3, H, W) tensor ready for
        `forward`.

        The single channel is replicated three times (the ResNet stem expects
        RGB), scaled to [0, 1], and then normalized with the ImageNet mean/std
        the pretrained weights were trained on. `device` defaults to whatever
        device this module currently lives on.
        """
        if device is None:
            device = self.imagenet_mean.device
        x = torch.from_numpy(np.stack([image, image, image], axis=0)).unsqueeze(0)
        x = x.to(device).float() / 255.0
        return (x - self.imagenet_mean.to(device)) / self.imagenet_std.to(device)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.stem(x)        # ~1/2 resolution
        x = self.layer1(x)      # still ~1/2 resolution
        x = self.reduce(x)      # reduce channels

        # Upsample back to the input resolution so features stay pixel-aligned
        # with the query point coordinates they will be sampled at.
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x     # [B, output_channels, H, W]


class EuclideanOptimizer(torch.nn.Module):
    """Learnable 2D rigid transform (rotation + translation), optimized to
    maximize feature-map activation at the transformed query points.

    The rotation is parameterized about `center` -- the sonar origin at the
    bottom-center of the image, `(W // 2, H)` -- matching the convention used
    everywhere else in the pipeline (`estimate_motion`,
    `BottomCenterRotationTransform`, the query-point origin filter). Rotating
    about image pixel (0, 0) instead spans the same rigid-transform family, but
    couples theta and (tx, ty) so strongly (a tiny theta drags every point by
    ~||p|| pixels) that LBFGS ends up letting `trans` absorb what should have
    been rotation about the sonar origin.

    Applied transform:  p' = R(theta) @ (p - center) + t + center
    which is exactly `BottomCenterRotationTransform`'s T3 @ T2 @ R @ T1
    composition, generalized to a full (tx, ty) translation. With
    `y_only=True` the translation is restricted to the Y axis, matching the
    utils class one-to-one (rotation + surge only, no sway)."""

    def __init__(self, center, y_only=False):
        super().__init__()
        self.y_only = y_only
        # Buffer (not Parameter): moves with .to(device), never optimized.
        self.register_buffer(
            "center", torch.as_tensor(center, dtype=torch.float32).view(1, 2)
        )
        self.theta = torch.nn.Parameter(torch.zeros(1))  # radians, about `center`
        if y_only:
            self.trans_y = torch.nn.Parameter(torch.zeros(1))  # ty
        else:
            self.trans = torch.nn.Parameter(torch.zeros(2))  # tx, ty

    def translation(self):
        """Current translation as a (2,) tensor, regardless of `y_only`."""
        if self.y_only:
            zero = torch.zeros_like(self.trans_y)
            return torch.cat([zero, self.trans_y])
        return self.trans

    def transform(self, points):
        """Apply the current rotation-about-center + translation to (N, 2) points."""
        c = torch.cos(self.theta)
        s = torch.sin(self.theta)
        # Build R with cat/stack rather than torch.tensor(...), which would
        # silently detach c/s from the autograd graph and leave theta's
        # gradient at None -- rotation would then never be optimized at all.
        R = torch.stack([torch.cat([c, -s]), torch.cat([s, c])], dim=0)

        centered = points - self.center                      # (N,2)
        return centered @ R.T + self.translation() + self.center


def sample_feature_values(feature_maps, coords):
    """
    Bilinearly sample feature map values at given (x,y) coordinates.

    feature_maps: (N, H, W)
    coords: (N,2) containing (x,y) in pixel units
    """
    N, H, W = feature_maps.shape

    # Normalize coordinates to [-1,1] for grid_sample.
    x = 2 * (coords[:, 0] / (W - 1)) - 1
    y = 2 * (coords[:, 1] / (H - 1)) - 1
    grid = torch.stack([x, y], dim=1).view(N, 1, 1, 2)

    sampled = F.grid_sample(
        feature_maps.unsqueeze(1),  # (N,1,H,W)
        grid,
        mode="bilinear",
        align_corners=True
    )  # -> shape (N,1,1,1)

    return sampled.view(N)


def optimize_euclidean_transform(points, feature_maps, top_left, center,
                                 lr=1, steps=5, y_only=False, trim_frac=0.2):
    """
    Optimize a shared rotation + translation to maximize correlation activation.

    points: (N,2) absolute pixel coords.
    feature_maps: (N,H,W) patch-local correlation maps (see
        `make_optimization_patches`); H, W are the small patch size, not the
        full image.
    top_left: (N,2) pixel coords of each patch's top-left corner, used to
        convert `points` (and their optimized transform) into patch-local
        coordinates before sampling.
    center: (2,) sonar origin (x, y) in absolute pixel coords -- the pivot of
        the rotation, i.e. the bottom-center of the image `(W // 2, H)`,
        matching `BottomCenterRotationTransform` / `estimate_motion` rather
        than a rotation about image pixel (0, 0).
    y_only: restrict translation to the Y axis (surge only), matching
        `BottomCenterRotationTransform` exactly.
    trim_frac: fraction of the worst-scoring points dropped from the objective
        at every evaluation (a robust trimmed mean). A few points with garbage
        correlation patches (occlusion, sonar noise, a wrong track) otherwise
        pull the shared rigid transform toward their noise; trimming lets the
        consensus of the well-matched majority set the transform. 0 disables
        trimming (plain mean over all points). The trimmed set is re-selected
        each evaluation, so a point re-enters the objective as soon as it stops
        being among the worst.

    `points`, `feature_maps` and `top_left` are expected to already live on the
    same device; the optimizer is placed there too so the whole loop stays on
    that device with no per-step host transfers.

    LBFGS is driven the way its API expects: a single `optimizer.step(closure)`
    call where the closure re-evaluates loss and gradients, with `max_iter=steps`
    doing the iterating internally. Passing a stale loss instead would make the
    strong-Wolfe line search re-evaluate the same value repeatedly without ever
    recomputing gradients. Because the line search sets the actual step size,
    `lr` is only the initial trial step (1.0 is the standard choice here).
    """
    model = EuclideanOptimizer(center, y_only=y_only).to(points.device)
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=lr, max_iter=steps, line_search_fn='strong_wolfe'
    )

    # Number of points kept in the trimmed mean (at least 1, at most all).
    n_pts = points.shape[0]
    n_keep = max(1, n_pts - int(n_pts * trim_frac)) if trim_frac > 0 else n_pts

    def closure():
        optimizer.zero_grad()
        transformed = model.transform(points)
        local = transformed - top_left
        scores = sample_feature_values(feature_maps, local)
        if n_keep < n_pts:
            # Robust trimmed mean: only the best n_keep scores contribute, so
            # gradients from the worst-matching points are dropped this step.
            scores = torch.topk(scores, n_keep, largest=True, sorted=False).values
        # Maximize scores -> minimize negative score.
        loss = -scores.mean()
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        final_points = model.transform(points)

    return final_points


class GridPointManager:
    """Keeps a stable, ID-tracked set of query points spread across an image
    grid, so no single region ends up over- or under-represented as points
    are added/dropped across frames."""

    def __init__(self, image_size, grid_size, max_points_per_cell, max_keypoints=500,
                 seed=None):
        """
        image_size: (H, W)
        grid_size: (cell_h, cell_w)
        max_points_per_cell: int, max points allowed per grid cell
        max_keypoints: int, global cap on the total number of tracked points
        seed: int seeding the Generator that `enforce_capacity` draws from.
            `np.random.default_rng` does not read the legacy global state that
            `np.random.seed()` sets, so the capacity drops would otherwise come
            from OS entropy and stay random even in a seeded run. The Generator
            is created once and carried across windows, so its state advances
            deterministically over the whole evaluation.
        """
        self.image_size = image_size
        self.grid_size = grid_size
        self.max_points_per_cell = max_points_per_cell
        self.max_keypoints = max_keypoints

        self.rng = np.random.default_rng(seed)

        self.points = np.empty((0, 2), dtype=float)
        self.ids = np.empty((0,), dtype=int)
        self.next_id = 0  # running ID counter

    # ---------------- Core functionality ----------------

    def _compute_grid_counts(self, points):
        """Compute point counts per grid cell."""
        H, W = self.image_size
        cell_h, cell_w = self.grid_size
        n_rows = int(np.ceil(H / cell_h))
        n_cols = int(np.ceil(W / cell_w))
        grid_counts = np.zeros((n_rows, n_cols), dtype=int)
        if len(points) > 0:
            rows = np.clip((points[:, 1] // cell_h).astype(int), 0, n_rows - 1)
            cols = np.clip((points[:, 0] // cell_w).astype(int), 0, n_cols - 1)
            np.add.at(grid_counts, (rows, cols), 1)
        return grid_counts

    def _distribute_new_points(self, new_points):
        """Keep only the new points that fit in cells with spare capacity.

        Candidates are considered in input order, so when a cell is
        oversubscribed the earlier (stronger-response) keypoints win.
        """
        H, W = self.image_size
        cell_h, cell_w = self.grid_size
        n_rows = int(np.ceil(H / cell_h))
        n_cols = int(np.ceil(W / cell_w))

        existing = self.points
        grid_counts = self._compute_grid_counts(existing)

        new_rows = np.clip((new_points[:, 1] // cell_h).astype(int), 0, n_rows - 1)
        new_cols = np.clip((new_points[:, 0] // cell_w).astype(int), 0, n_cols - 1)
        remaining_capacity = np.maximum(self.max_points_per_cell - grid_counts, 0)
        remaining_capacity_flat = remaining_capacity.ravel()

        flat_idx = new_rows * n_cols + new_cols
        added_counts = np.zeros_like(remaining_capacity_flat)
        keep_mask = np.zeros(len(new_points), dtype=bool)

        for i, idx in enumerate(flat_idx):
            if added_counts[idx] < remaining_capacity_flat[idx]:
                keep_mask[i] = True
                added_counts[idx] += 1

        return new_points[keep_mask]

    # ---------------- Public methods ----------------

    def add_points(self, new_points):
        """Add new points (respecting grid limits) and assign unique IDs."""
        new_points = np.asarray(new_points)
        accepted_points = self._distribute_new_points(new_points)
        n_new = len(accepted_points)

        if n_new > 0:
            new_ids = np.arange(self.next_id, self.next_id + n_new)
            self.next_id += n_new
            self.points = np.vstack([self.points, accepted_points]) if len(self.points) else accepted_points
            self.ids = np.concatenate([self.ids, new_ids]) if len(self.ids) else new_ids

        return accepted_points

    def update_points(self, new_positions, valid_mask):
        """
        Update existing point positions.
        valid_mask: boolean array of same length as current points.
        Points marked False are forgotten.
        """
        new_positions = np.asarray(new_positions)
        assert len(valid_mask) == len(self.points), "Valid mask must match number of tracked points"

        # Keep only valid points, at their new positions.
        self.points = new_positions[valid_mask].copy()
        self.ids = self.ids[valid_mask]

    def enforce_capacity(self, random_state=None):
        """
        Ensure no grid cell exceeds max_points_per_cell, then globally
        subsample down to max_keypoints if the total still exceeds it.
        Extra points are removed at random.

        The drops come from the instance Generator seeded in __init__, so they
        are reproducible across runs with the same seed. Pass `random_state` to
        override it for a single call with a fresh, independently seeded
        Generator (useful in tests); note that doing so leaves the instance
        Generator's state untouched, so later calls stay on the seeded stream.
        """
        if len(self.points) == 0:
            return  # nothing to enforce

        H, W = self.image_size
        cell_h, cell_w = self.grid_size
        n_rows = int(np.ceil(H / cell_h))
        n_cols = int(np.ceil(W / cell_w))

        rng = self.rng if random_state is None else np.random.default_rng(random_state)

        # ---- Per-cell capacity ----
        rows = np.clip((self.points[:, 1] // cell_h).astype(int), 0, n_rows - 1)
        cols = np.clip((self.points[:, 0] // cell_w).astype(int), 0, n_cols - 1)
        flat_idx = rows * n_cols + cols

        keep_mask = np.ones(len(self.points), dtype=bool)
        _, inverse_idx, counts = np.unique(flat_idx, return_inverse=True, return_counts=True)
        inverse_idx = inverse_idx.ravel()  # guard against numpy 2.0 shape change

        for pos, count in enumerate(counts):
            if count > self.max_points_per_cell:
                cell_indices = np.where(inverse_idx == pos)[0]
                drop = rng.choice(cell_indices, size=count - self.max_points_per_cell, replace=False)
                keep_mask[drop] = False

        self.points = self.points[keep_mask]
        self.ids = self.ids[keep_mask]

        # ---- Global keypoint cap ----
        if len(self.points) > self.max_keypoints:
            keep = rng.choice(len(self.points), size=self.max_keypoints, replace=False)
            keep.sort()  # preserve original point/ID ordering
            self.points = self.points[keep]
            self.ids = self.ids[keep]


def make_optimization_patches(query_points, predicted_points, f0, f2, patch_radius=9):
    """Build one local correlation map per query point.

    For every query point, its frame-0 feature vector is correlated against a
    small (2*patch_radius+1)^2 neighborhood of frame-2 features centered on the
    tracker's predicted location, for all points at once. The result is a set
    of small patch-local tensors (N, P, P) plus each patch's top-left pixel,
    rather than one full-image canvas per point, which keeps the allocation
    O(N * P^2) instead of O(N * H * W) per frame pair.

    Runs on whatever device `f0`/`f2` live on: the integer index tensors and
    the returned `top_left` are placed on that device so nothing here forces a
    host transfer.

    f0, f2: feature maps [C, H, W] (same H, W as the source images).
    query_points, predicted_points: (N, 2) pixel coords (x, y).
    Returns: patches (N, P, P) float32 in [0, 1], top_left (N, 2) float32 (x, y).
    """
    C, H, W = f0.shape
    P = 2 * patch_radius + 1
    N = len(query_points)
    device = f0.device

    qx = np.clip(query_points[:, 0].astype(np.int64), 0, W - 1)
    qy = np.clip(query_points[:, 1].astype(np.int64), 0, H - 1)
    ix = np.clip(predicted_points[:, 0].astype(np.int64), 0, W - 1)
    iy = np.clip(predicted_points[:, 1].astype(np.int64), 0, H - 1)

    qy_t = torch.from_numpy(qy).to(device)
    qx_t = torch.from_numpy(qx).to(device)
    query_vec = f0[:, qy_t, qx_t].T  # (N, C)

    # Zero-pad so every patch is the same full PxP size, even near image
    # edges -- this is what makes the gather below batchable across all N
    # points at once, instead of clipping the window separately per point.
    f2_pad = F.pad(f2, (patch_radius, patch_radius, patch_radius, patch_radius))

    offsets = torch.arange(P, device=device)
    rows = torch.from_numpy(iy).to(device)[:, None] + offsets[None, :]  # (N, P)
    cols = torch.from_numpy(ix).to(device)[:, None] + offsets[None, :]  # (N, P)
    patches = f2_pad[:, rows[:, :, None], cols[:, None, :]]   # (C, N, P, P)
    patches = patches.permute(1, 0, 2, 3)                     # (N, C, P, P)

    corr = F.cosine_similarity(query_vec.view(N, C, 1, 1), patches, dim=1)  # (N, P, P)
    # Affine remap [-1, 1] -> [0, 1] rather than clamping negatives to 0: a
    # clamp flattens every anti-correlated region to exactly 0, so a point
    # sitting there receives zero gradient and can never be pulled toward
    # better territory. The remap keeps the same output range while preserving
    # gradient signal (and score ordering) everywhere in the patch.
    corr = (corr + 1.0) * 0.5

    # ...except in the zero-padded (out-of-image) part of each patch, where the
    # cosine similarity against the all-zero pad vector is 0 and the remap
    # would turn it into a middling 0.5 -- *better* than anti-correlated real
    # content, quietly attracting points off the image edge. Force padding back
    # to 0, the worst possible score.
    row_valid = (rows >= patch_radius) & (rows < patch_radius + H)   # (N, P)
    col_valid = (cols >= patch_radius) & (cols < patch_radius + W)   # (N, P)
    valid = row_valid[:, :, None] & col_valid[:, None, :]            # (N, P, P)
    corr = corr * valid

    top_left = np.stack([ix - patch_radius, iy - patch_radius], axis=1).astype(np.float32)
    return corr, torch.from_numpy(top_left).to(device)


def write_results(results_file, estimated_motion, gt_motion, i, j, inlier_count, all_count, visible, poses, inlier_ratio):
    """Append one evaluated frame pair (i -> j) as a CSV row matching the header in main()."""
    with open(results_file, "a") as rf:
        rf.write(f"{i},{j},")
        rf.write(
            f"{estimated_motion[0]:.6f},{estimated_motion[1]:.6f},{estimated_motion[2]:.6f},"
        )
        rf.write(
            f"{gt_motion[0]:.6f},{gt_motion[1]:.6f},{gt_motion[2]:.6f}"
        )

        rf.write(f",{inlier_count},{all_count}")

        n_visible = np.sum(visible)
        rf.write(f",{n_visible}")

        # avrg_error: placeholder column kept so the CSV layout stays compatible
        # with the analysis scripts that read these files.
        rf.write(",0.000000")

        rf.write(f",{inlier_ratio:.6f}")

        from_timestep = poses[0]["metadata"]["timestamp"]
        to_timestep = poses[j]["metadata"]["timestamp"]
        rf.write(f",{from_timestep},{to_timestep}")

        rf.write("\n")


def make_grid_points(image_shape, mask, origin, min_origin_dist=0.0):
    """Regular grid of auxiliary query points, filtered by the valid-area mask
    and by distance from the sonar origin.

    Used as a fallback when too few Sobel keypoints survive selection (see
    --fallback_grid_points). These points are regenerated per window and are
    intentionally *not* stored in the GridPointManager -- only Sobel keypoints
    persist across windows."""
    x = np.linspace(30, image_shape[1] - 30, 40)
    y = np.linspace(30, image_shape[0] - 30, 20)
    xv, yv = np.meshgrid(x, y)
    grid_points = np.stack([xv.flatten(), yv.flatten()], axis=-1)

    valid = mask[grid_points[:, 1].astype(int), grid_points[:, 0].astype(int)]
    grid_points = grid_points[valid]

    dists = np.linalg.norm(grid_points - np.array(origin), axis=1)
    return grid_points[dists > min_origin_dist]


def load_dataset(dataset_path, mask_path, n):
    """Load the Aracati sequence and its valid-area mask.

    The mask is thresholded to a boolean and then eroded, so query points never
    land right on the edge of the sonar coverage area where a small tracking
    error already moves them outside it.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read the dataset mask: {mask_path}")
    mask = mask > 0
    mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=3).astype(bool)

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

    dataset = ISOPoTDataset(
        dataset_path,
        vflip_image=False,
        hflip_image=False,
        n=n,
        superpoint_conf=1,
        generate_kpts=False,
        save_kpts=False,
    )

    return dataset, mask


def select_query_points(image0, mask, grid_point_manager, fallback_grid_points,
                        min_origin_dist=0.0):
    """Pick query points on frame 0 of the window.

    Sobel-edge keypoints are filtered by distance from the sonar origin
    (`min_origin_dist`, 0 = no distance filtering) and by the valid-area mask,
    then merged into the persistent grid manager so points stay spread out
    across frames. If `fallback_grid_points` is set and fewer than
    `MIN_PERSISTENT_POINTS` persistent points survive, a regular grid of points
    is appended for this window only -- those are regenerated every window and
    never stored in the GridPointManager.

    Returns:
        query_points: (N, 2) array; the first `n_persistent` rows are the
            GPM-managed Sobel keypoints, any remaining rows are the
            per-window fallback grid points.
        n_persistent: number of GPM-managed points at the start of the array.
    """
    origin = (image0.shape[1] / 2, image0.shape[0])

    query_points = sobel_keypoints(image0, percentile=0.98, mask=mask, size=7)

    # Distance filtering: drop points too close to the sonar origin
    # (bottom-center), where rotations move pixels the least.
    if min_origin_dist > 0:
        dists = np.linalg.norm(query_points - np.array(origin), axis=1)
        query_points = query_points[dists > min_origin_dist]

    # Mask filtering: keep only points inside the valid sonar coverage area.
    inliers = mask[query_points[:, 1].astype(int), query_points[:, 0].astype(int)]
    query_points = query_points[inliers]

    # Merge the Sobel points into the grid manager (enforce_capacity drops
    # excess points in over-dense cells). Only these points persist across windows.
    grid_point_manager.add_points(query_points)
    grid_point_manager.enforce_capacity()
    query_points = grid_point_manager.points
    n_persistent = len(query_points)

    # Fallback only: too few keypoints survived to fit a reliable transform, so
    # top the set up with a grid for this window. n_persistent is left
    # unchanged, which is what keeps the grid points out of both the
    # carried-over set and (by default) the refinement optimization.
    if fallback_grid_points and n_persistent < MIN_PERSISTENT_POINTS:
        grid_points = make_grid_points(image0.shape, mask, origin,
                                       min_origin_dist=min_origin_dist)
        query_points = np.concatenate([query_points, grid_points], axis=0)

    return query_points, n_persistent


def track_window(model, images, query_points):
    """Track query points across all frames in the window.

    Returns predictions (n_frames, N, 2) and visibility (n_frames, N), i.e.
    the wrapper's point-major output transposed to frame-major.
    """
    with torch.no_grad():
        predictions, visible = model.track(images, query_points)
    predictions = np.transpose(predictions, (1, 0, 2))
    visible = np.transpose(visible, (1, 0))
    return predictions, visible


def find_transformation_between_predictions_fast(pred1, pred2, visible1, visible2,
                                                   threshold=5.0, max_trials=1000):
    """Same as utils.find_transformation_between_predictions(transform_type="euclidean"),
    but with a configurable (lower) max_trials.

    skimage's `ransac` runs its trial loop in pure Python, so cost scales
    directly with max_trials. The two refine-only fits that use this
    (`setup_refinement_features`, `refine_transform`) start from already
    RANSAC-cleaned or feature-refined point sets, so they converge reliably
    with far fewer trials than the 5000 used for raw tracker output in
    `estimate_pairwise_transform`.

    Returns the (2, 3) transform matrix and an inlier mask over *all* N input
    points (not just the jointly visible ones). Falls back to the identity if
    fewer than 3 points are jointly visible or the fit fails.
    """
    N = len(pred1)
    visible_mask = visible1 & visible2
    pts1 = pred1[visible_mask]
    pts2 = pred2[visible_mask]

    if pts1.shape[0] < 3 or pts2.shape[0] < 3:
        return np.eye(2, 3), np.zeros(N, dtype=bool)

    try:
        model, inliers_mask = ransac(
            (pts1, pts2),
            EuclideanTransform,
            min_samples=4,
            residual_threshold=threshold,
            max_trials=max_trials,
        )
        transform_matrix = model.params[:2, :3]
    except Exception:
        transform_matrix = np.eye(2, 3)
        inliers_mask = np.zeros(len(pts1), dtype=bool)

    inliers_full = np.zeros(N, dtype=bool)
    inliers_full[visible_mask] = inliers_mask

    return transform_matrix, inliers_full


def setup_refinement_features(refinement_model, images, w, query_points, predictions,
                              visible, ransac_threshold, device):
    """Extract dense frame-0 features and the initial frame0 -> warmup transform.

    Used by the feature-correlation refinement step (see `--refine`). `f0` is
    the reference feature map for every `refine_transform` call in this window,
    so it is kept resident on `device` rather than downloaded to the host,
    saving one (C, H, W) device->host transfer per frame pair.
    """
    image0 = refinement_model.prepare_input(images[0], device)
    with torch.no_grad():
        f0 = refinement_model(image0).squeeze(0).detach()

    T, inliers = find_transformation_between_predictions_fast(
        predictions[0],
        predictions[w],
        visible[0],
        visible[w],
        threshold=ransac_threshold,
    )
    query_points_transformed = query_points @ T[:2, :2].T + T[:2, 2][None, :]

    return f0, query_points_transformed


def estimate_pairwise_transform(predictions, visible, images, w, j, ransac_threshold,
                                visible_j, T_cache):
    """Estimate the transform from the warmup frame to frame j, and separately
    from the warmup frame to frame j-1, then compose to get the (j-1) -> j
    transform. This is more stable than estimating (j-1) -> j directly,
    since both legs share the same well-tracked reference (warmup) frame.

    `visible_j` is the stuck-point-filtered visibility for frame j; `visible`
    (the full per-window array) is left untouched so it keeps reflecting the
    raw tracker output for every other frame index.

    `T_cache` is a per-window dict {frame_index: 3x3 T_0_index}. The warmup ->
    (j-1) leg was already fitted as the warmup -> j leg of the previous
    iteration, so it is reused instead of re-running the 5000-trial RANSAC.
    Note that the reused fit was made with the stuck-point-filtered visibility
    of its own iteration rather than raw `visible[j-1]`, i.e. a slightly
    cleaner point set."""
    pts2 = predictions[j]
    image2 = images[j]

    T_0_j, inliers = find_transformation_between_predictions(
        predictions[w],
        pts2,
        visible[w],
        visible_j,
        images[w],
        image2,
        threshold=ransac_threshold,
        transform_type="euclidean",
    )

    # Square up to 3x3 so the legs can be composed via matrix inverse/multiply.
    T_0_j = np.vstack([T_0_j, [0, 0, 1]])

    if (j - 1) in T_cache:
        T_0_jm1 = T_cache[j - 1]
    elif j - 1 == w:
        # warmup -> warmup is the identity by construction; no need to fit it.
        T_0_jm1 = np.eye(3)
    else:
        T_0_jm1, _ = find_transformation_between_predictions(
            predictions[w],
            predictions[j - 1],
            visible[w],
            visible[j - 1],
            images[w],
            images[j - 1],
            threshold=ransac_threshold,
            transform_type="euclidean",
        )
        T_0_jm1 = np.vstack([T_0_jm1, [0, 0, 1]])

    T_cache[j] = T_0_j

    T = T_0_j @ np.linalg.inv(T_0_jm1)
    T = T[:2, :]

    return T, inliers


def refine_transform(refinement_model, device, query_points, query_points_transformed, T,
                     f0, image2, pts2, visible_j, opt_mask,
                     steps=10, patch_radius=10, y_only=False):
    """Feature-correlation refinement: nudge the rigid transform so that each
    tracked point's neighborhood best matches its original feature response,
    rather than trusting the tracker's raw point estimate.

    `visible_j` is the stuck-point-filtered visibility for the *current* frame
    j, so each pair refines on the points actually visible in that pair. Only
    points selected by `opt_mask` (by default the persistent Sobel keypoints,
    see --optimize_sobel_only) take part in the optimization and the subsequent
    RANSAC fit; the resulting transform is still applied to the full
    `query_points_transformed` set.

    Effort knobs: `steps` LBFGS iterations on correlation patches of radius
    `patch_radius`. The patch radius also bounds the correction, since a
    coordinate sampled outside its own patch gets zero gradient and so can
    never be pulled further than one radius.

    Everything from the `f2` extraction through the rigid-transform optimizer
    stays on `device`; the only device->host transfer is the refined (N, 2)
    point set, once per frame pair (skimage's RANSAC and the patch builder are
    CPU/numpy).

    Returns the refined (2, 3) transform and the updated carried point set. If
    fewer than 4 points are available to optimize, the raw transform is
    returned unchanged."""
    with torch.no_grad():
        f2 = refinement_model(
            refinement_model.prepare_input(image2, device)
        ).squeeze(0).detach()

    optimizable = visible_j & opt_mask

    temporary_query_points = query_points_transformed @ T[:2, :2].T + T[:2, 2][None, :]

    # Only the optimizable subset is ever used, so build patches for that
    # subset alone -- correlating all N points and discarding most of the
    # result is what would make a large patch radius unaffordable on hard pairs.
    # Anchors come from the original frame, since those are the actual points
    # we want to track (rather than the current prediction).
    anchors = query_points[optimizable]              # frame-0 feature reference
    centers = pts2[optimizable]                      # patch centers: tracker prediction
    current = temporary_query_points[optimizable]    # estimate being optimized

    if len(anchors) < 4:
        # Too few points to fit anything afterwards; keep the raw transform
        # rather than letting the RANSAC fallback collapse it to the identity.
        return T, temporary_query_points

    # Sonar origin at the bottom-center of the image -- the pivot the whole
    # pipeline is expressed around (same as `estimate_motion` and
    # `select_query_points`). The optimizer rotates about this point, not (0,0).
    sonar_origin = (image2.shape[1] // 2, image2.shape[0])

    patches, top_left = make_optimization_patches(
        anchors, centers, f0, f2, patch_radius=patch_radius,
    )

    refined_points = optimize_euclidean_transform(
        torch.from_numpy(current).float().to(device),  # (N,2) on device
        patches,                                       # (N,P,P) on device
        top_left,                                      # (N,2) on device
        sonar_origin,                                  # rotation pivot
        steps=steps,
        y_only=y_only,
    )
    current = refined_points.detach().cpu().numpy()

    # The refined points are already feature-aligned, so this fit converges
    # with far fewer trials (and a tighter threshold) than the raw tracker output.
    T_refined, _ = find_transformation_between_predictions_fast(
        query_points_transformed[optimizable],
        current,
        np.ones(len(current), dtype=bool),
        np.ones(len(current), dtype=bool),
        threshold=1,
        max_trials=1000,
    )

    query_points_transformed = (
        query_points_transformed @ T_refined[:2, :2].T + T_refined[:2, 2][None, :]
    )

    return T_refined, query_points_transformed


def process_frame_pair(args, results_file, i, j, w, ransac_threshold,
                       images, poses, predictions, visible, query_points,
                       refine, refinement_model, device, f0, query_points_transformed,
                       T_cache, opt_mask):
    """Evaluate one (j-1) -> j frame pair: estimate the transform, optionally
    refine it, derive ego-motion, and write the result row.

    Returns the (possibly refined) transform, the updated
    query_points_transformed (only meaningful when `refine` is set), and the
    inlier mask used for grid bookkeeping.

    `opt_mask`: (N,) bool -- which query points may participate in the
    feature-correlation refinement (see --optimize_sobel_only)."""
    origin_points = predictions[j - 1]
    origin_visibility = visible[j - 1]
    origin_pose = get_pose(poses[j - 1])

    pts2 = predictions[j]
    image2 = images[j]
    pose2 = get_pose(poses[j])

    # Drop points that haven't moved at all (likely stuck/false tracks).
    d = np.linalg.norm(pts2 - query_points, axis=1)
    visible2 = visible[j] & (d > 1.0)

    T, inliers = estimate_pairwise_transform(
        predictions, visible, images, w, j, ransac_threshold, visible2, T_cache
    )

    inlier_count = np.sum(inliers)
    point_count = origin_points.shape[0]
    # Denominator is all query points, not just the visible ones, so the ratio
    # also drops when the tracker loses points. Logged as-is in the results file.
    inlier_ratio = inlier_count / point_count if point_count > 0 else 0.0

    if refine:
        # A high inlier ratio means the raw fit already explains nearly every
        # point, so refinement is skipped; only the weak pairs get optimized.
        if inlier_ratio < args.refine_inlier_threshold:
            T, query_points_transformed = refine_transform(
                refinement_model, device, query_points, query_points_transformed, T,
                f0, image2, pts2, visible2, opt_mask,
                steps=args.refine_steps,
                patch_radius=args.refine_patch_radius,
                y_only=args.refine_y_only,
            )
        else:
            # Still march the carried point set forward with the raw transform,
            # so `final_points` (which the --refine path takes from
            # query_points_transformed) stays consistent across the window
            # whether or not any given pair was refined.
            query_points_transformed = (
                query_points_transformed @ T[:2, :2].T + T[:2, 2][None, :]
            )

    resolution = poses[w]["metadata"]["resolution"]
    estimated_motion = list(estimate_motion(T, resolution, images[0].shape[:2]))
    # estimate_motion returns (dy, dx, dyaw) in image axes; reorder to the
    # (dx, dy, dyaw) vehicle convention the ground-truth poses use.
    estimated_motion = [estimated_motion[1], estimated_motion[0], estimated_motion[2]]

    gt_motion = get_relative_pose(origin_pose, pose2)

    write_results(results_file, estimated_motion, gt_motion, i, j, inlier_count,
                  point_count, origin_visibility & visible2, poses, inlier_ratio)

    return T, query_points_transformed, inliers


def update_grid_points(grid_point_manager, mask, images, visible, final_points, inliers,
                       n_persistent):
    """Carry surviving persistent (Sobel) points into the next window.

    A point survives only if it is still visible, still inside the image, still
    inside the valid-area mask, and was a RANSAC inlier on the last evaluated
    pair. Only the first `n_persistent` entries of `final_points` correspond to
    GPM-managed points; any fallback grid points beyond that are discarded here
    since they are regenerated per window."""
    # .copy() so the in-place bitwise ops below don't mutate the caller's
    # `visible` array (visible[-1] is a view into it).
    visibility_mask = visible[-1][:n_persistent].copy()
    final_points = final_points[:n_persistent]
    inliers = inliers[:n_persistent]

    pixel_predictions = np.round(final_points).astype(int)

    height, width = mask.shape
    xs = np.clip(pixel_predictions[:, 0], 0, width - 1)
    ys = np.clip(pixel_predictions[:, 1], 0, height - 1)
    valid_mask = mask[ys, xs]

    # Drop points that fell outside the image bounds (the clip above would
    # otherwise quietly pull them back to the border).
    inside_image_mask = final_points[:, 0] >= 0
    inside_image_mask &= final_points[:, 0] < images[0].shape[1]
    inside_image_mask &= final_points[:, 1] >= 0
    inside_image_mask &= final_points[:, 1] < images[0].shape[0]

    visibility_mask &= valid_mask
    visibility_mask &= inliers
    visibility_mask &= inside_image_mask

    grid_point_manager.update_points(
        new_positions=final_points,
        valid_mask=visibility_mask,
    )


def process_window(args, model, dataset, mask, grid_point_manager, refinement_model, device,
                   results_file, i, n, w, refine, ransac_threshold, window_data=None):
    """Process a single window of `n` frames starting at index `i`.

    `window_data`: optionally the already-loaded result of `dataset[i]` (from
    the background prefetcher). When it is None the window is loaded inline.
    """
    # images/poses: a window of n frames starting at i; the third element is
    # the dataset's superpoint/AKAZE keypoints, unused here.
    images, poses, _ = window_data if window_data is not None else dataset[i]

    query_points, n_persistent = select_query_points(
        images[0], mask, grid_point_manager, args.fallback_grid_points,
        min_origin_dist=args.min_origin_dist,
    )

    predictions, visible = track_window(model, images, query_points)

    # Which points participate in the --refine feature-correlation
    # optimization: by default only the persistent Sobel keypoints; with
    # --no-optimize_sobel_only, all query points (incl. fallback grid points).
    opt_mask = np.zeros(len(query_points), dtype=bool)
    if args.optimize_sobel_only:
        opt_mask[:n_persistent] = True
    else:
        opt_mask[:] = True

    f0 = query_points_transformed = None
    if refine:
        f0, query_points_transformed = setup_refinement_features(
            refinement_model, images, w, query_points, predictions, visible,
            ransac_threshold, device,
        )

    inliers = None
    T_cache = {}  # per-window cache of warmup->frame fits (see estimate_pairwise_transform)
    for j in range(w + 1, len(images)):
        T, query_points_transformed, inliers = process_frame_pair(
            args, results_file, i, j, w, ransac_threshold,
            images, poses, predictions, visible, query_points,
            refine, refinement_model, device, f0, query_points_transformed, T_cache,
            opt_mask,
        )

    # Point set handed to the next window. Windows advance by (n - w - 1)
    # frames, so frame n-w-1 of this window is frame 0 of the next one. With
    # --refine the carried set is the refined one instead, which has been
    # marched forward through every pair in the window.
    final_points = predictions[n - w - 1] if not refine else query_points_transformed

    # Only the persistent Sobel points (the first n_persistent entries) are
    # carried into the next window; fallback grid points are regenerated.
    update_grid_points(
        grid_point_manager, mask, images, visible, final_points, inliers, n_persistent,
    )


def build_results_filename(args):
    """Short key-value parts joined by underscores, so every setting that
    materially changes a run is visible at a glance and runs with different
    settings never overwrite each other. Refinement sub-settings only appear
    with --refine, keeping raw-run names short."""
    def _fmt(v):
        return f"{v:g}"  # 3.0 -> "3", 0.9 -> "0.9"

    parts = [
        MODEL_NAME,
        DATASET_NAME,
        f"seed{args.seed}",
        f"n{args.n}",
        f"w{args.w}",
        f"ransac{_fmt(args.ransac_threshold)}",
        "fallbackgrid" if args.fallback_grid_points else "nogrid",
        f"mindist{_fmt(args.min_origin_dist)}" if args.min_origin_dist > 0 else "nodistfilter",
    ]
    if args.refine:
        parts.append("refine")
        if args.refine_inlier_threshold <= 1.0:
            parts.append(f"ir{_fmt(args.refine_inlier_threshold)}")
        if args.optimize_sobel_only:
            parts.append("sobelonly")
        if args.refine_y_only:
            parts.append("yonly")

    return "_".join(parts) + ".txt"


# One row per evaluated frame pair. `index`/`time_offset` locate the pair
# (window start index and frame offset within the window); p* is the estimated
# motion, g* the ground truth.
RESULTS_HEADER = [
    "index",
    "time_offset",
    "pX",
    "pY",
    "pYAW",
    "gX",
    "gY",
    "gYAW",
    "inlier_count",
    "all_count",
    "n_visible",
    "avrg_error",
    "inlier_ratio",
    "from_timestamp",
    "to_timestamp",
]


def main():
    args = parse_args()

    # Seed torch and numpy so tracking/refinement runs are reproducible across
    # evaluations. GridPointManager needs the seed passed in separately: it
    # uses a Generator, which ignores the legacy global state set here.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n = args.n
    w = args.w
    refine = args.refine
    ransac_threshold = args.ransac_threshold

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    results_filename = build_results_filename(args)
    results_file = os.path.join(save_path, results_filename)

    with open(results_file, "w") as rf:
        rf.write(",".join(RESULTS_HEADER) + "\n")

    model = TAPNextWrapper(fine_tuned_weights=args.fine_tuned_weights)
    dataset, mask = load_dataset(args.dataset_path, args.mask_path, n)

    print(f"Seed: {args.seed} (results -> {results_filename})")
    if args.fallback_grid_points:
        print(
            f"Falling back to grid points in windows with fewer than "
            f"{MIN_PERSISTENT_POINTS} Sobel keypoints."
        )

    grid_point_manager = GridPointManager(
        image_size=mask.shape,
        grid_size=(mask.shape[0] // 16, mask.shape[1] // 16),
        max_points_per_cell=5,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Only built when refining -- constructing it downloads/loads the ResNet50
    # weights, which is pure overhead for a raw run.
    refinement_model = None
    if refine:
        refinement_model = PretrainedFeatureExtractor(output_channels=128).to(device)
        refinement_model.eval()

        if args.optimize_sobel_only:
            print("Refinement optimization restricted to persistent Sobel keypoints.")
        if args.refine_inlier_threshold > 1.0:
            print("Refining every frame pair (inlier gate disabled).")
        else:
            print(
                f"Refining only frame pairs with inlier ratio < {args.refine_inlier_threshold} "
                f"(steps={args.refine_steps}, patch_radius={args.refine_patch_radius})."
            )

    # Windows overlap: each one advances by (n - w - 1) frames, so the last
    # evaluated frame of a window becomes the first frame of the next.
    window_indices = list(range(1, len(dataset), n - w - 1))

    if args.prefetch > 0:
        # Overlap disk I/O with compute: a background thread loads up to
        # `--prefetch` upcoming windows while the current one is processed.
        prefetcher = WindowPrefetcher(dataset, window_indices, depth=args.prefetch)

        def window_iter():
            for idx, data in prefetcher:
                yield idx, data
    else:
        prefetcher = None

        def window_iter():
            for idx in window_indices:
                yield idx, None

    try:
        for i, window_data in tqdm(window_iter(), total=len(window_indices)):
            process_window(
                args, model, dataset, mask, grid_point_manager, refinement_model, device,
                results_file, i, n, w, refine, ransac_threshold,
                window_data=window_data,
            )
    finally:
        if prefetcher is not None:
            prefetcher.close()


if __name__ == "__main__":
    main()