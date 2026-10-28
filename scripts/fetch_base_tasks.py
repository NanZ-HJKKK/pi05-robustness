"""Download small, pinned official task definitions; never downloads weights/assets."""
import ast
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "evidence/upstream"


def get(url):
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "pi05-research-setup"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1)


def main():
    manifest = json.loads((DEST / "manifest.json").read_text(encoding="utf-8"))
    sha = manifest["openpi"]["commit"]
    meta = json.loads(get("https://api.github.com/repos/Physical-Intelligence/openpi/contents/third_party/libero?ref=" + sha))
    repo, commit = "Lifelong-Robot-Learning/LIBERO", meta["sha"]
    map_path = "libero/libero/benchmark/libero_suite_task_map.py"
    raw = get(f"https://raw.githubusercontent.com/{repo}/{commit}/{map_path}")
    tree = ast.parse(raw.decode())
    tasks = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign) and
                                  any(isinstance(t, ast.Name) and t.id == "libero_task_map" for t in n.targets)))
    paths = [map_path] + [f"libero/libero/bddl_files/{suite}/{name}.bddl"
                         for suite, names in tasks.items() if suite != "libero_90" for name in names]

    def fetch(path):
        data = get(f"https://raw.githubusercontent.com/{repo}/{commit}/{path}")
        name = "libero__" + path.replace("/", "__")
        (DEST / name).write_bytes(data)
        return path, {"file": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        files = dict(pool.map(fetch, paths))
    manifest["libero"] = {"repo": repo, "commit": commit, "files": files}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"libero_commit": commit, "downloaded_text_files": len(files)}))


if __name__ == "__main__":
    main()
