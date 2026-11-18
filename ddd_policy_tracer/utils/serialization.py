"""Serialization methods for various data types."""

import struct


def serialize_f32(vector: list[float]) -> bytes:
    """Serializes a list of floats into a compact "raw bytes" format."""
    return struct.pack("%sf" % len(vector), *vector)


def deserialize_f32(blob: bytes) -> list[float]:
    """Deserializes a blob of raw bytes back into a list of floats."""
    num_floats = len(blob) // 4
    return list(struct.unpack("%sf" % num_floats, blob))
