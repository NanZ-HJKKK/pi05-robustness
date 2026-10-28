import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

TEST_DIR = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("hf_transfer_script", Path(__file__).resolve().parents[1] / "scripts/fetch_hf_resources.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Response(io.BytesIO):
    status = 206

    def __init__(self, data, start, end, total):
        super().__init__(data)
        self.headers = {"Content-Range": "bytes %d-%d/%d" % (start, end, total)}


class HFTransferTests(unittest.TestCase):
    data = b"verified payload, including partial prefixes"

    def item(self, lfs=True):
        return {"path": "data/test.parquet", "size": len(self.data),
                "oid": hashlib.sha1(("blob %d\0" % len(self.data)).encode() + self.data).hexdigest(),
                "lfs": {"sha256": hashlib.sha256(self.data).hexdigest()} if lfs else None}

    def response(self, request, timeout):
        start, end = map(int, request.get_header("Range").split("=")[1].split("-"))
        return Response(self.data[start:end + 1], start, end, len(self.data))

    def test_parallel_ranges_and_lfs_hash(self):
        with tempfile.TemporaryDirectory(dir=TEST_DIR) as tmp, patch.object(module.urllib.request, "urlopen", self.response):
            self.assertTrue(Path(tmp).resolve().is_relative_to(TEST_DIR))
            target = Path(tmp) / "file"
            record = module.download("https://example.test/file", target, self.item(), lambda: None, workers=3, stride=7)
            self.assertEqual(target.read_bytes(), self.data)
            self.assertEqual(record["sha256"], hashlib.sha256(self.data).hexdigest())
            self.assertFalse(target.with_name("file.chunks").exists())

    def test_resume_partial_prefix_and_git_blob_hash(self):
        starts = []
        def respond(request, timeout):
            starts.append(int(request.get_header("Range").split("=")[1].split("-")[0]))
            return self.response(request, timeout)
        with tempfile.TemporaryDirectory(dir=TEST_DIR) as tmp, patch.object(module.urllib.request, "urlopen", respond):
            self.assertTrue(Path(tmp).resolve().is_relative_to(TEST_DIR))
            target = Path(tmp) / "file"
            chunks = target.with_name("file.chunks")
            chunks.mkdir()
            (chunks / "0.partial").write_bytes(self.data[:3])
            module.download("https://example.test/file", target, self.item(False), lambda: None, stride=9)
            self.assertEqual(starts[0], 3)
            self.assertEqual(target.read_bytes(), self.data)

    def test_truncated_response_preserves_prefix(self):
        starts = []
        def respond(request, timeout):
            start, end = map(int, request.get_header("Range").split("=")[1].split("-"))
            starts.append(start)
            body = self.data[start:end + 1] if len(starts) > 1 else self.data[:4]
            return Response(body, start, end, len(self.data))
        with tempfile.TemporaryDirectory(dir=TEST_DIR) as tmp, patch.object(module.urllib.request, "urlopen", respond), patch.object(module.time, "sleep"):
            self.assertTrue(Path(tmp).resolve().is_relative_to(TEST_DIR))
            target = Path(tmp) / "file"
            module.download("https://example.test/file", target, self.item(), lambda: None)
            self.assertEqual(starts, [0, 4])
            self.assertEqual(target.read_bytes(), self.data)

    def test_reject_wrong_range_or_corrupt_body(self):
        for corrupt_body in [False, True]:
            def respond(request, timeout):
                if corrupt_body:
                    return Response(b"x" * len(self.data), 0, len(self.data) - 1, len(self.data))
                return Response(self.data, 1, len(self.data), len(self.data))
            with self.subTest(corrupt_body=corrupt_body), tempfile.TemporaryDirectory(dir=TEST_DIR) as tmp, patch.object(module.urllib.request, "urlopen", respond):
                self.assertTrue(Path(tmp).resolve().is_relative_to(TEST_DIR))
                target = Path(tmp) / "file"
                with self.assertRaises(ValueError):
                    module.download("https://example.test/file", target, self.item(), lambda: None)
                self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
