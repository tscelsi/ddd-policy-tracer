## Graph Schema

### Entities

| Entity | Source Object |
| --- | --- |
| Person | DocumentChunk |
| Chunk | DocumentChunk |
| Document | SourceDocument |

### Relations

| Relation | Source Entity | Target Entity |
| --- | --- | --- |
| made by | Document | Person |
| belongs to | Chunk | Document |