"""Download unchanged, pinned upstream inputs; no inference or credentials."""
import hashlib
import argparse
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
COMMIT = "a65cdbec3f6060906407cf391d2c69446e363d83"
DATA_COMMIT = "13861f99b2b268dbdffad7a95281e4ea27fe172d"
FILES = ["code/eval.py", "code/prompt.txt", "data/falvzixun.json", "data/sifakaoshi.json", "LICENSE", "README.md"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("lexrubric-sources"),
                        help="Destination for pinned sources; installed package files are never overwritten")
    output_root = parser.parse_args().output_dir
    records = {}
    for name in FILES:
        url = f"https://raw.githubusercontent.com/foggpoy/LexRubric/{COMMIT}/{name}"
        path = output_root / "upstream" / name
        data = urlopen(url, timeout=90).read()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != data:
            raise RuntimeError(f"Refusing to replace changed source: {path}")
        path.write_bytes(data)
        records[name] = {"url": url, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        print(f"Saved {name}: {len(data)} bytes", flush=True)
    manifest = {"repository": "https://github.com/foggpoy/LexRubric", "commit": COMMIT,
                "paper": "https://arxiv.org/html/2606.09389v1", "files": records}
    (output_root / "upstream_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    original = {}
    for name in ["falvzixun.json", "sifakaoshi.json"]:
        url = f"https://raw.githubusercontent.com/foggpoy/LexRubric/{DATA_COMMIT}/data/{name}"
        data = urlopen(url, timeout=90).read()
        path = output_root / "data" / name
        path.parent.mkdir(exist_ok=True)
        if path.exists() and path.read_bytes() != data:
            raise RuntimeError(f"Refusing to replace changed original data: {path}")
        path.write_bytes(data)
        original[name] = {"url": url, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        print(f"Saved original data/{name}: {len(data)} bytes", flush=True)
    (output_root / "data_manifest.json").write_text(json.dumps({"commit": DATA_COMMIT, "files": original}, indent=2) + "\n")


if __name__ == "__main__":
    main()
