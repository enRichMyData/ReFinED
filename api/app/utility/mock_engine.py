import time
import random

class MockModel:
    """A mock model that simulates processing time and generates random results.
    """
    def __init__(self, *args, **kwargs):
        print("MOCK MODE initialized with simulated lag")

    def process_text(self, text: str):
        print(f"DEBUG processing text: {text}")
        time.sleep(2.0)
        return [MockSpan(text)]

class MockSpan:
    def __init__(self, text):
        self.text = text
        self.predicted_entity = MockEntity(text)
        self.candidate_entities = [
            (self.predicted_entity, 0.98),               # top candidate with high confidence
            (MockEntity(f"Alternative {text}"), 0.45)    # second candidate with lower confidence
        ]
        self.coarse_mention_type = "MOCK_TYPE"
        self.coarse_type = "MOCK_TYPE"

class MockEntity:
    def __init__(self, title):
        self.wikidata_entity_id = f"Q{random.randint(100, 99999)}"
        self.wikipedia_entity_title = f"Mock {title}"
        self.description = "Simulated entity for MOCK MODE"