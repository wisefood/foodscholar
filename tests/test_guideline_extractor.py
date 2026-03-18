import tempfile
import unittest
from pathlib import Path


class GuidelineExtractorServiceTests(unittest.TestCase):
    def test_dedupe_guidelines_collapses_near_duplicates(self):
        from services.guideline_extractor import dedupe_guidelines

        items = [
            "- Offer vegetables daily.",
            "Offer vegetables daily",
            "Offer vegetables daily!",
            "Serve pulses regularly.",
        ]

        self.assertEqual(
            dedupe_guidelines(items),
            ["Offer vegetables daily.", "Serve pulses regularly."],
        )

    def test_artifact_workspace_uses_uuid_folder_and_source_pdf(self):
        from services.guideline_extractor import GuidelineExtractorService

        artifact_uuid = "123e4567-e89b-12d3-a456-426614174000"

        with tempfile.TemporaryDirectory() as tmpdir:
            service = GuidelineExtractorService(workspace_root=tmpdir)
            info = service.get_artifact_workspace(artifact_uuid)

            self.assertTrue(Path(info.artifact_dir).is_dir())
            self.assertEqual(info.workspace_root, str(Path(tmpdir).resolve()))
            self.assertEqual(
                info.pdf_path,
                str(Path(tmpdir).resolve() / artifact_uuid / "source.pdf"),
            )
            self.assertFalse(info.pdf_exists)

    def test_invalid_artifact_uuid_is_rejected(self):
        from services.guideline_extractor import (
            GuidelineExtractionError,
            GuidelineExtractorService,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            service = GuidelineExtractorService(workspace_root=tmpdir)
            with self.assertRaises(GuidelineExtractionError):
                service.get_artifact_workspace("not-a-uuid")


if __name__ == "__main__":
    unittest.main()
