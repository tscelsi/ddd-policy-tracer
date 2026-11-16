# Completion Chat Frontend

## Goal

Add a simple, responsive chat interface for asking questions about parliamentary speeches and bills through the existing completion API.

## User experience

- Build a Next.js app with TypeScript under `frontend/`, styled with the dark charcoal/purple palette, compact borders, and restrained serif typography of the supplied reference image.
- Let a user submit one question at a time and see a visible transcript until the page is refreshed.
- Treat every submitted question as an independent search. Do not send prior turns as context or persist conversations.
- Show a loading state while the synchronous request is running, explicit API errors, and a distinct insufficient-evidence state when the API abstains.
- Render the answer and its inline citation references. Opening a citation reveals its source passage and available speaker, date, house, and chamber metadata.
- Keep the retrieval limit at the API default of 10; do not add a control for it.
- Do not imply that a source excerpt links to a full speech. The current completion API does not provide a speech-detail URL or lookup route.

## API integration

- Use the existing `POST /api/completion` request contract: `{ "question": string }`, with the optional `limit` omitted.
- Proxy the request through the Next.js server to the Python API, configured by an environment variable and defaulting to `http://127.0.0.1:5000`.
- Keep API transport server-side so browser requests remain same-origin and do not require CORS changes.

## Domain language

- A **Citation** is the answer's reference to retrieved evidence and its provenance.
- A **Source Excerpt** is supporting passage text from a document chunk; it is not the complete speech or source document, and does not imply a navigable link.
- Add these terms to the existing `UBIQUITOUS_LANGUAGE.md` vocabulary.

## Acceptance criteria

- The frontend can be run independently from the Python API and successfully proxies a question to it.
- Users can submit follow-up questions, but each request is independent and only the in-memory transcript is retained.
- Returned answer text, inline citations, expandable excerpt details, abstention, loading, and error states are presented clearly.
- The frontend passes its type check and production build, with targeted tests for the chat/API boundary.

## Out of scope

- Conversation-aware retrieval, user accounts, or persisted conversation history.
- Full-speech pages or citation URLs; these require a separate backend contract.
- User-configurable retrieval limits.
