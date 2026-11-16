import { strict as assert } from "node:assert";
import { afterEach, describe, it } from "node:test";
import { POST } from "../app/api/completion/route";

describe("POST /api/completion proxy", () => {
  const originalFetch = globalThis.fetch;
  const originalConsoleError = console.error;
  const originalApiBaseUrl = process.env.COMPLETION_API_BASE_URL;

  afterEach(() => {
    globalThis.fetch = originalFetch;
    console.error = originalConsoleError;
    if (originalApiBaseUrl === undefined) {
      delete process.env.COMPLETION_API_BASE_URL;
    } else {
      process.env.COMPLETION_API_BASE_URL = originalApiBaseUrl;
    }
  });

  it("forwards the request to the configured API and preserves the response", async () => {
    process.env.COMPLETION_API_BASE_URL = "http://api.example.test/";
    let fetchedUrl: RequestInfo | URL | undefined;
    let fetchedOptions: RequestInit | undefined;
    globalThis.fetch = async (input, init) => {
      fetchedUrl = input;
      fetchedOptions = init;
      return new Response('{"answer":"ok"}', {
        status: 200,
        headers: {
          "content-type": "application/json",
          "server-timing": "completion;dur=20",
        },
      });
    };
    const request = new Request("http://localhost:3000/api/completion", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: '{"question":"What was discussed?"}',
    });
    const response = await POST(request);

    assert.equal(String(fetchedUrl), "http://api.example.test/api/completion");
    assert.equal(fetchedOptions?.method, "POST");
    assert.equal(fetchedOptions?.body, '{"question":"What was discussed?"}');
    assert.equal(fetchedOptions?.cache, "no-store");
    assert.equal(response.status, 200);
    assert.equal(response.headers.get("server-timing"), "completion;dur=20");
    assert.deepEqual(await response.json(), { answer: "ok" });
  });

  it("uses the documented local API by default", async () => {
    delete process.env.COMPLETION_API_BASE_URL;
    let fetchedUrl: RequestInfo | URL | undefined;
    globalThis.fetch = async (input) => {
      fetchedUrl = input;
      return new Response("{}", { status: 200 });
    };
    const request = new Request("http://localhost:3000/api/completion", {
      method: "POST",
      body: '{"question":"What was discussed?"}',
    });

    await POST(request);

    assert.equal(String(fetchedUrl), "http://127.0.0.1:5000/api/completion");
  });

  it("returns an explicit gateway error when the API cannot be reached", async () => {
    globalThis.fetch = async () => {
      throw new Error("offline");
    };
    console.error = () => {};
    const request = new Request("http://localhost:3000/api/completion", {
      method: "POST",
      body: '{"question":"What was discussed?"}',
    });

    const response = await POST(request);

    assert.equal(response.status, 502);
    assert.deepEqual(await response.json(), {
      detail: "The completion service is unavailable. Please try again.",
    });
  });
});
