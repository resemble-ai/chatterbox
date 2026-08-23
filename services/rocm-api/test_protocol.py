"""Pure tests for phrase chunking and the streaming protocol implementation."""

import ast
import sys
import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name("app.py")
sys.path.insert(0, str(SOURCE.parent))
from chunking import narration_chunks


def function_source(name: str) -> str:
    tree = ast.parse(SOURCE.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    return ast.get_source_segment(SOURCE.read_text(), node) or ""


class StreamImplementationTests(unittest.TestCase):
    def test_stream_closes_cleanly(self) -> None:
        source = SOURCE.read_text()
        self.assertIn('await websocket.send_json({"type": "end"})', source)
        self.assertIn('await websocket.close(code=1000, reason="complete")', source)

    def test_model_generation_is_off_the_event_loop(self) -> None:
        source = SOURCE.read_text()
        self.assertIn("await asyncio.to_thread(generate_audio, chunk.text, voice)", source)
        self.assertIn("with model_lock:", function_source("generate_audio"))

    def test_long_sentences_split_on_clause_punctuation(self) -> None:
        text = (
            "The old gate begins to open with a long metallic complaint, "
            "and something hidden behind it answers with a lower scraping sound, "
            "before the torchlight finally reaches the chamber beyond."
        )
        chunks = narration_chunks(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 140 for chunk in chunks))
        self.assertEqual(chunks[0].boundary, "clause")
        self.assertEqual(chunks[-1].boundary, "final")

    def test_paragraph_boundaries_are_protocol_metadata(self) -> None:
        chunks = narration_chunks("The first paragraph ends.\n\nThe second begins.")
        self.assertEqual([chunk.text for chunk in chunks], ["The first paragraph ends.", "The second begins."])
        self.assertEqual([chunk.boundary for chunk in chunks], ["paragraph", "final"])

    def test_short_sentences_are_not_fragmented(self) -> None:
        chunks = narration_chunks("The bell rings. The ravens scatter.")
        self.assertEqual([chunk.text for chunk in chunks], ["The bell rings.", "The ravens scatter."])

    def test_lone_word_joins_the_following_sentence(self) -> None:
        chunks = narration_chunks("You're not the tanner. Good. The tanner had the wrong key.")
        texts = [chunk.text for chunk in chunks]
        self.assertNotIn("Good.", texts)
        self.assertEqual(texts[-1], "Good. The tanner had the wrong key.")

    def test_trailing_lone_word_joins_the_previous_sentence(self) -> None:
        text = (
            "The key was not on the table, and it had clearly been taken in haste, "
            "so the search moved to the next chamber, where the door hung open and "
            "a single word echoed from the dark. Key."
        )
        chunks = narration_chunks(text)
        self.assertNotIn("Key.", [chunk.text for chunk in chunks])
        self.assertTrue(chunks[-1].text.endswith("Key."))

    def test_leading_lone_word_joins_the_next_sentence(self) -> None:
        chunks = narration_chunks("Yes. The king has arrived.")
        self.assertEqual([chunk.text for chunk in chunks], ["Yes. The king has arrived."])


if __name__ == "__main__":
    unittest.main()
