"""Metadata-only resource inventory; never download model/dataset contents."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse

sys.path.insert(0,str(Path(__file__).resolve().parent))
from fetch_base_tasks import get

ROOT = Path(__file__).resolve().parents[1]


def hf_inventory(repo, filename=None):
    base = "https://huggingface.co/api/datasets/"+repo
    info = json.loads(get(base+"?blobs=true"))
    revision = info["sha"]
    # blobs=true provides sizes for the entire fixed-revision siblings listing.
    files = [f for f in info["siblings"] if filename is None or f["rfilename"] == filename]
    if filename and not files:
        raise ValueError("Requested asset missing: "+filename)
    return {"repo":repo,"revision":revision,"files":[{"path":f["rfilename"],"size":f["size"],
        "oid":f.get("blobId"),"lfs":f.get("lfs")} for f in files], "bytes":sum(f["size"] for f in files)}


def main():
    manifest_path = ROOT/"evidence/upstream/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    more = ["packages/openpi-client/src/openpi_client/websocket_client_policy.py",
            "src/openpi/serving/websocket_policy_server.py", "src/openpi/training/checkpoints.py",
            "examples/libero/requirements.txt"]
    for path in more:
        data = get(f"https://raw.githubusercontent.com/Physical-Intelligence/openpi/{manifest['openpi']['commit']}/{path}")
        name = "openpi__"+path.replace("/","__")
        (ROOT/"evidence/upstream"/name).write_bytes(data)
        manifest["openpi"]["files"][path] = {"file":name,"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    prefix = "checkpoints/pi05_libero/"
    query = urllib.parse.urlencode({"prefix":prefix, "fields":"items(name,size,md5Hash,crc32c),nextPageToken"})
    page = json.loads(get("https://storage.googleapis.com/storage/v1/b/openpi-assets/o?"+query))
    if page.get("nextPageToken"):
        raise ValueError("GCS listing needs pagination")
    items = page["items"]
    checkpoint = {"prefix":"gs://openpi-assets/"+prefix,"files":items,"bytes":sum(int(i["size"]) for i in items)}
    report = {"checkpoint":checkpoint,"dataset":hf_inventory("physical-intelligence/libero"),
              "plus_assets":hf_inventory("Sylvest/LIBERO-plus", "assets.zip"),
              "note":"Metadata only. Compressed Plus size excludes extraction; Python environments, cache and output checkpoints require additional space."}
    (ROOT/"evidence/resource_inventory.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:{"bytes":v["bytes"],"GiB":round(v["bytes"]/2**30,2),"files":len(v["files"])}
                      for k,v in report.items() if isinstance(v,dict)},indent=2))


if __name__ == "__main__":
    main()
