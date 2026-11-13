Enrichment layer for DocumentChunk objects

```python
layers = [
    FilterByLengthLayer(min_length=500),
    EmbeddingLayer(embedding_service=EmbedService()),
    TopicClassificationLayer(),
    EntityExtractionLayer(),
    LanguageFingerprintLayer(),
]

pipeline = Pipeline(layers=layers, doc_store=LocalChunkStore(), batch_size=16)
pipeline.run(chunks)

class Pipeline:
    def run(self):
        pass
```