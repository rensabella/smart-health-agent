"""Explicit deterministic embedding fixture; never offered by runtime environment settings."""

import numpy as np


class ScriptedEmbedding:
    embedding_id = "test_only/scripted_signal_vectors_v1"
    dimensions = 8

    def token_count(self, text, *, prefix):
        return len(text) + len(prefix)

    def _encode(self, texts):
        rows = []
        for text in texts:
            lowered = text.casefold()
            row = [
                float(any(word in lowered for word in words))
                for words in (
                    ("eda", "皮电"),
                    ("heart", "心率"),
                    ("eeg", "脑电"),
                    ("sleep", "睡眠"),
                    ("activity", "活动"),
                    ("ppg", "光"),
                    ("wellness", "健康生活"),
                )
            ]
            row.append(float(not any(row)))
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)

    encode_queries = _encode
    encode_passages = _encode

    def close(self):
        pass
