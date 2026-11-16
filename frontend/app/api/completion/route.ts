const DEFAULT_API_BASE_URL = "http://127.0.0.1:5000";

export async function POST(request: Request): Promise<Response> {
  const apiBaseUrl =
    process.env.COMPLETION_API_BASE_URL ?? DEFAULT_API_BASE_URL;

  try {
    const apiUrl = new URL("/api/completion", apiBaseUrl);
    const apiResponse = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "content-type": request.headers.get("content-type") ?? "application/json",
      },
      body: await request.text(),
      cache: "no-store",
    });
    const responseHeaders = new Headers({
      "content-type":
        apiResponse.headers.get("content-type") ?? "application/json",
    });
    const serverTiming = apiResponse.headers.get("server-timing");
    if (serverTiming) {
      responseHeaders.set("server-timing", serverTiming);
    }

    return new Response(await apiResponse.text(), {
      status: apiResponse.status,
      headers: responseHeaders,
    });
  } catch (error) {
    console.error("Unable to reach the completion API.", error);
    return Response.json(
      { detail: "The completion service is unavailable. Please try again." },
      { status: 502 },
    );
  }
}
