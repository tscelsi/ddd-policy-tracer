# Completion Chat Frontend

This Next.js app provides a chat interface for the Python completion API.

## Run locally

1. Start the API using the repository's normal setup (by default, it listens on `http://127.0.0.1:5000`).
2. From this directory, install the frontend dependencies and start the dev server:

   ```sh
   npm install
   npm run dev
   ```

3. Open `http://localhost:3000`.

The browser sends `POST /api/completion` to Next.js on port 3000. The Next.js server then proxies it to the Python API, which defaults to `http://127.0.0.1:5000`; set `COMPLETION_API_BASE_URL` in the frontend environment to use a different API origin. See `.env.example`.

Each submitted question is an independent search. The visible conversation is kept in memory only and clears on refresh.

## Checks

```sh
npm test
npm run typecheck
npm run build
```
