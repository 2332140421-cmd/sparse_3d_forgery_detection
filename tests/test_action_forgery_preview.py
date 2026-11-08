"""Research contracts for this media-only review, without external assets."""
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.prepare_action_forgery_preview import choose, extract_member, render, row


class PreviewTests(unittest.TestCase):
    def test_selection_independent_of_listing_order(self):
        items = [{"id": str(i)} for i in range(20)]
        a = choose(items, 4, lambda x: x["id"])
        b = choose(list(reversed(items)), 4, lambda x: x["id"])
        self.assertEqual(a, b)
        self.assertEqual(len({x["id"] for x in a}), 4)

    def test_missing_mask_is_not_empty_ground_truth(self):
        r = row("FakeParts", "rev", "x", "fake", "x.mp4", "fakeparts/x.mp4", "inpainting")
        self.assertEqual(r["mask_path"], "")
        self.assertEqual(r["alignment"], "not available")
        self.assertIn("无公开", r["mask_type"])

    def test_reused_member_must_match_archive_not_only_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(root / "test.zip", "w") as z:
                z.writestr("source.mp4", b"official member")
            with zipfile.ZipFile(root / "test.zip") as z:
                extract_member(root, z, "source.mp4", "videos/source.mp4")
                (root / "videos/source.mp4").write_bytes(b"wrong existing")
                with self.assertRaisesRegex(ValueError, "ARCHIVE_MEMBER_MISMATCH"):
                    extract_member(root, z, "source.mp4", "videos/source.mp4")

    def test_html_uses_relative_paths_and_no_guessed_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "preview").mkdir()
            r = row("HFV", "rev", "a<&", "fake", "a.mp4", "hfv/a.mp4", "0: spatial")
            r.update(decode="COMPLETE", frames=10, width=32, height=32, fps="10", duration_s=1, pictures=[])
            render(root, [r])
            text = (root / "preview/index.html").read_text()
            self.assertIn("../hfv/a.mp4", text)
            self.assertIn("preload='none'", text)
            self.assertIn("a&lt;&amp;", text)
            self.assertNotIn(str(root), text)
            self.assertIn("不显示mask或伪造叠加", text)


if __name__ == "__main__":
    unittest.main()
