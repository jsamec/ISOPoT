# ISOPoT

**I**maging **S**onar **O**dometry by **P**oint **T**racking — estimate
ego-motion (dx, dy, dyaw) for an underwater vehicle from sonar image
sequences, by tracking sparse keypoints across frames with a point-tracking
model and fitting a rigid 2D transform to the result.

![ISOPoT Boot Sequence video](media/boot_video.gif)

Note: This repo was cleaned up quickly for review and hasn't been re-verified end-to-end. Expect rough edges around reproducibility partially genericized paths, drifted dependency versions, inconsistent renames.

## Install

```bash
conda create -n isopot python=3.12
conda activate isopot
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e .
```

## Local data

`local/` and `weights/` are gitignored and not published with the repo.
The tracker's defaults expect your own copies in place:

- `local/datasets/aracati/` — the Aracati sequence (`--dataset_path` default)
- `local/masks/sonar_coverage_mask.png` — the valid sonar-coverage mask
  (`--mask_path` default)
- `weights/bootstapnext_ckpt.npz` — the base TAPNext checkpoint (~740MB,
  too large to publish in the git repo directly)

Override the dataset/mask paths with `--dataset_path` / `--mask_path`
instead. The base TAPNext checkpoint path itself is not a CLI flag — it
must be placed at `weights/bootstapnext_ckpt.npz` (`--fine_tuned_weights`
optionally layers a fine-tuned state dict on top of it, but doesn't replace
it).

## Running the tracker

The main entry point is `eval/isopot_tracker.py`. For each window of `n`
frames it picks sparse keypoints on the first frame, tracks them across the
window with TAPNext, fits a rigid (rotation + translation) transform per
frame pair, converts that into an estimated ego-motion, and compares it
against the ground-truth pose delta from the dataset. Results are appended
to a CSV under `--save_path` (default `eval_runs/`).

```bash
python eval/isopot_tracker.py --n 5 --w 3
```

See `python eval/isopot_tracker.py --help` for the full set of options
(`--refine` for feature-correlation refinement, `--ransac_threshold`,
`--fallback_grid_points`, etc).

## Layout

- `eval/isopot_tracker.py` — the main tool, described above.
- `utils/` — supporting code: the TAPNext model wrapper
  (`model_wrappers.py`), the `ISOPoTDataset` dataset loader (`datasets.py`),
  and geometry/keypoint helpers (`utils.py`).
- `models/` — point-tracking model implementations used by the wrapper:
  TAPNext (`tapnext/`, `tapnet_utils/`) and SuperPoint keypoint detection
  (`superpoint/`).
- `weights/` — gitignored; holds the base TAPNext checkpoint. See "Local
  data" above.
- `local/` — gitignored local data (dataset, masks); see "Local data" above.
- `eval_runs/` — output CSVs from tracker runs (created on first run).
