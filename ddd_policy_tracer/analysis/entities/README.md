EntityCatalog - a catalog of known entities that we trust and want to track. Each CatalogEntity has a unique ID, canonical name, type, and metadata. We resolve EntityMentions to CatalogEntities when possible.

CatalogEntity - is a person, organization, policy that pertains to something in the world. It's a unique thing that we want to track across documents and over time. Examples: "Clean Air Act", "EPA", "California". The CatalogEntity is the canonical representation.

EntityMention - is a specific mention of an entity in a document. It has a span, text, type, and may be linked to a CatalogEntity if we can resolve it.

