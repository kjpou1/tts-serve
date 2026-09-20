"""Stub for the ``voxcpm`` package (test machines only).

The server imports only the ``VoxCPM`` class (no module-level engine
constants), so this placeholder keeps the import surface identical without
pulling in torch.  Model loading is never exercised in the tests.
"""


class _TtsModelStub:
    """Placeholder for the inner TTS model."""

    sample_rate = 48000
    device = "cuda"
    last_successful_seed = None


class VoxCPM:
    """Placeholder — real model loading is never exercised in the tests."""

    tts_model = _TtsModelStub()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError(
            "voxcpm stub: from_pretrained() is not available in tests"
        )

    def generate(self, *args, **kwargs):
        raise NotImplementedError(
            "voxcpm stub: generate() is not available in tests"
        )
