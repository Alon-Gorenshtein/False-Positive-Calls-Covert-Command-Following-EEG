"""Morgoth event-level background labels for one streamed BDSP segment, run LOCALLY (MPS, else CPU).
Mirrors the --predict branch of Morgoth's finetune_classification.py (common-average montage, 10-s windows,
1-s step, its own resampling/filtering/clipping/scaling) by calling Morgoth's own utils; the copied Morgoth code
is not modified. Differences from the CUDA script: one process, no DDP, fp32 instead of fp16 autocast, softmax in
float32, and windows Morgoth rejects as invalid are NaN (Morgoth writes 0 / class 0) with a validity flag.
One parquet per segment, no raw signal: ../output/bdsp/morgoth/<stem>.parquet, row = window starting at t_s.
python bdsp_morgoth.py <file.mat> [...]"""
import os
import sys
import time

import numpy as np
import pandas as pd

MORGOTH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output/morgoth"))
OUT = "../output/bdsp/morgoth"
MAX_HOURS = 2
HEADS = {
    "normal": ("NORMAL", "NORMAL.pth", ["p"]),
    "slowing": ("SLOWING", "SLOWING.pth", ["none", "focal", "generalized"]),
    "bs": ("BS", "BS.pth", ["p"]),
    "iiic": ("IIIC", "IIIC_sz_hm13_1.pth", ["other", "seizure", "lpd", "gpd", "lrda", "grda"]),
    "sleep": ("SLEEPPSG", "SLEEPPSG.pth", ["awake", "n1", "n2", "n3", "rem"]),
}
BATCH = 96


def _morgoth():
    if f"{MORGOTH}/code" not in sys.path:
        sys.path.insert(0, f"{MORGOTH}/code")
    import backbone
    import utils
    return utils


def device():
    import torch
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_models(dev=None, heads=HEADS):
    """{head: model}, built with each checkpoint's own training flags and loaded strictly. The CUDA script builds
    with qkv_bias=True and loads with strict=False; these checkpoints were trained without q/v biases, so the
    script's extra biases stay at their zero init and both builds compute the same function."""
    import torch
    from timm.models import create_model
    utils = _morgoth()
    dev = dev or device()
    models = {}
    for head, (key, ckpt, _) in heads.items():
        state = torch.load(f"{MORGOTH}/models/{ckpt}", map_location="cpu", weights_only=False)
        a = vars(state["args"]) if not isinstance(state["args"], dict) else state["args"]
        if a["dataset"] != key or a["nb_classes"] != utils.get_n_classes(key):
            raise RuntimeError(f"{ckpt}: trained for {a['dataset']} with {a['nb_classes']} classes, not {key}")
        m = create_model("morgoth_backbone_base", pretrained=False, num_classes=a["nb_classes"],
                         drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0, drop_block_rate=None,
                         use_mean_pooling=a["use_mean_pooling"], init_scale=0.001,
                         use_rel_pos_bias=a["rel_pos_bias"], use_abs_pos_emb=a["abs_pos_emb"],
                         init_values=a["layer_scale_init_value"], qkv_bias=a["qkv_bias"])
        m.load_state_dict(state["model"], strict=True)
        models[head] = m.float().to(dev).eval()
    return models


def to_rows(pred, valid_index, valid_runs, run_lengths):
    """Per-window probabilities [n_windows x k], NaN where Morgoth rejected the window. Same run-wise resize as
    the script (a no-op when every valid run yields one snippet per window)."""
    utils = _morgoth()
    out = np.full((len(valid_index), pred.shape[1]), np.nan, dtype=np.float32)
    for (s, e), seg in zip(valid_runs, utils.split_nd_to_plus1d(arr=pred, segment_shape=run_lengths)):
        out[s:e + 1] = utils.resize_array_along_axis0(arr=seg, d=2, target_length=e - s + 1)
    return out


def windows(path, key, max_hours=MAX_HOURS):
    """Morgoth's own preprocessing of one recording: (X [valid windows x ch x 10 x 200] scaled as the script feeds
    it, channel names, validity mask, valid runs, run lengths). X is None when every window is rejected. NORMAL,
    SLOWING, BS and IIIC build identical windows (the key only matters for SLEEPPSG and SPIKES), so one call
    serves all four."""
    import torch
    from einops import rearrange
    utils = _morgoth()
    montage = utils.sleep_common_average_montage() if key == "SLEEPPSG" else utils.common_average_montage()
    try:
        ds = utils.ContinuousToSnippetDataset(
            type=key, path_signal=path, given_fs=0, has_avg=False, has_formatted_channel=False,
            allow_missing_channels=False, montage=montage, transform=utils.Compose([utils.Clipping(), utils.Scaling()]),
            step=1, step_in_point=False, polarity=1, leave_one_hemisphere_out=False, channel_symmetric_flip=False,
            max_length_hour=max_hours)
    except utils.SnippetsError as e:
        n = int(e.result_segment_shapes)
        return None, montage.get_channel_names(), np.zeros(n, dtype=bool), [], []
    x = torch.stack([ds[j][0] for j in range(len(ds))]).float() / 100
    valid_index, runs, lengths = ds.get_valid_indices()
    return (rearrange(x, "B N (A T) -> B N A T", T=200), montage.get_channel_names(),
            np.asarray(valid_index, dtype=bool), runs, lengths)


def predict(model, w, dev):
    """Probabilities [n_windows x k] for one head on windows() output, NaN where the window was rejected."""
    import torch
    from scipy.special import softmax
    x, names, valid, runs, lengths = w
    k = model.head.out_features
    if x is None:
        return np.full((len(valid), k), np.nan, dtype=np.float32)
    input_chans = _morgoth().morgoth_get_input_chans(names)
    with torch.no_grad():
        logits = np.concatenate([model(x[i:i + BATCH].to(dev), input_chans=input_chans).float().cpu().numpy()
                                 for i in range(0, len(x), BATCH)])
    pred = 1 / (1 + np.exp(-logits)) if k == 1 else softmax(logits, axis=1)
    return to_rows(pred, valid, runs, lengths)


def run(path, models, dev=None, out_dir=OUT):
    """Label one segment with every head and write <stem>.parquet. Returns (frame, seconds per step)."""
    dev = dev or device()
    cols, secs = {}, {}
    t0 = time.time()
    w19, wsleep = windows(path, "NORMAL"), windows(path, "SLEEPPSG")
    secs["preprocess"] = time.time() - t0
    for head, model in models.items():
        t0 = time.time()
        prob = predict(model, wsleep if head == "sleep" else w19, dev)
        secs[head] = time.time() - t0
        for j, name in enumerate(HEADS[head][2]):
            cols[f"{head}_{name}"] = prob[:, j]
    n = len(w19[2])
    if len(wsleep[2]) != n:
        raise RuntimeError(f"{path}: {n} windows vs {len(wsleep[2])} for the sleep head")
    df = pd.DataFrame({"t_s": np.arange(n, dtype=np.int32), "valid": w19[2], "valid_sleep": wsleep[2], **cols})
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.basename(path).rsplit(".", 1)[0]
    df.to_parquet(f"{out_dir}/{stem}.parquet.part", index=False)
    os.replace(f"{out_dir}/{stem}.parquet.part", f"{out_dir}/{stem}.parquet")
    return df, secs


def main(paths):
    dev = device()
    models = load_models(dev)
    for p in paths:
        df, secs = run(p, models, dev)
        print(os.path.basename(p), len(df), "windows", {k: round(v, 1) for k, v in secs.items()}, flush=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
