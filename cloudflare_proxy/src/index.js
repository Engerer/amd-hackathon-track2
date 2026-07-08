const MAX_BODY_CHARS = 28_000_000;
const MAX_TOKENS = 1000;
const MAX_MESSAGES = 8;
const MAX_TEMPERATURE = 0.9;

function numberOrDefault(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function positiveNumberOrDefault(value, fallback) {
  const number = numberOrDefault(value, fallback);
  return number > 0 ? number : fallback;
}

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json",
    },
  });
}

function validatePayload(body) {
  if (!body || typeof body !== "object") {
    return "Request body must be a JSON object.";
  }
  if (typeof body.model !== "string" || !body.model.trim()) {
    return "Missing model.";
  }
  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return "Missing messages.";
  }
  if (body.messages.length > MAX_MESSAGES) {
    return `Too many messages. Maximum is ${MAX_MESSAGES}.`;
  }
  if (JSON.stringify(body).length > MAX_BODY_CHARS) {
    return "Request is too large.";
  }
  return "";
}

export default {
  async fetch(request, env) {
    if (request.method !== "POST") {
      return jsonResponse({ error: "Use POST." }, 405);
    }

    if (env.MODEL_PROXY_TOKEN && request.headers.get("X-Proxy-Token") !== env.MODEL_PROXY_TOKEN) {
      return jsonResponse({ error: "Unauthorized." }, 401);
    }

    if (!env.FIREWORKS_API_KEY) {
      return jsonResponse({ error: "FIREWORKS_API_KEY secret is not configured." }, 500);
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return jsonResponse({ error: "Invalid JSON body." }, 400);
    }

    const validationError = validatePayload(body);
    if (validationError) {
      return jsonResponse({ error: validationError }, 400);
    }

    const maxTokens = positiveNumberOrDefault(env.MAX_TOKENS, MAX_TOKENS);
    const maxTemperature = positiveNumberOrDefault(env.MAX_TEMPERATURE, MAX_TEMPERATURE);
    const requestedTemperature = numberOrDefault(body.temperature, 0.2);
    const payload = {
      model: body.model,
      messages: body.messages,
      temperature: Math.min(Math.max(requestedTemperature, 0), maxTemperature),
      max_tokens: Math.min(positiveNumberOrDefault(body.max_tokens, 700), maxTokens),
      reasoning_effort: body.reasoning_effort || "none",
    };
    if (body.response_format && typeof body.response_format === "object") {
      payload.response_format = body.response_format;
    }

    const baseUrl = (env.FIREWORKS_BASE_URL || "https://api.fireworks.ai/inference/v1").replace(/\/$/, "");
    const fireworksResponse = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${env.FIREWORKS_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!fireworksResponse.ok) {
      const text = await fireworksResponse.text();
      console.error("Fireworks error", fireworksResponse.status, text.slice(0, 1000));
      return jsonResponse(
        {
          error: "Fireworks request failed.",
          status: fireworksResponse.status,
          detail: text.slice(0, 1000),
        },
        fireworksResponse.status
      );
    }

    const data = await fireworksResponse.json();
    return jsonResponse(data);
  },
};
