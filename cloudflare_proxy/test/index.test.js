import assert from "node:assert/strict";
import test from "node:test";

import worker from "../src/index.js";


test("proxy caps completions at 4000 and forwards response_format", async () => {
  let forwarded;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (_url, options) => {
    forwarded = JSON.parse(options.body);
    return new Response(JSON.stringify({ choices: [{ message: { content: "ok" } }] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  try {
    const responseFormat = {
      type: "json_schema",
      json_schema: { name: "response", schema: { type: "object" } },
    };
    const request = new Request("https://proxy.example", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "accounts/fireworks/models/kimi-k2p6",
        messages: [{ role: "user", content: "select" }],
        max_tokens: 9000,
        response_format: responseFormat,
      }),
    });
    const response = await worker.fetch(request, { FIREWORKS_API_KEY: "test-key" });

    assert.equal(response.status, 200);
    assert.equal(forwarded.max_tokens, 4000);
    assert.deepEqual(forwarded.response_format, responseFormat);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
