import pathlib
import unittest


class NoMergeMarkersTests(unittest.TestCase):
    def test_python_files_have_no_merge_markers(self):
        repo = pathlib.Path(__file__).resolve().parent
        markers = ("<<<<<<<", "=======", ">>>>>>>")
        offenders = []

        for py_file in repo.glob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            for marker in markers:
                if f"\n{marker} " in text or text.startswith(marker):
                    offenders.append(f"{py_file.name}: {marker}")

        self.assertEqual(offenders, [], msg=f"Merge markers found: {offenders}")


if __name__ == "__main__":
    unittest.main()
