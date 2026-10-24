"""Output tags for declared reruns (e.g. "_v2labels", the Kojima B label correction). A tagged run writes
<stem><tag><ext> next to the original and never overwrites it; untagged runs never read tagged files."""
import glob
import os
import re

TAGGED = re.compile(r"_v\d+[a-z_]*$")


def name(path, tag=""):
    stem, ext = os.path.splitext(path)
    return f"{stem}{tag}{ext}"


def untagged(pattern):
    """glob(pattern), sorted, without tagged files or macOS ._ resource forks."""
    return sorted(f for f in glob.glob(pattern) if not os.path.basename(f).startswith("._")
                  and not TAGGED.search(os.path.splitext(os.path.basename(f))[0]))


def pick(path, tag=""):
    """The tagged version of path when it exists (a rerun dataset), else path (a dataset the rerun left alone)."""
    t = name(path, tag)
    return t if tag and os.path.exists(t) else path
