"use strict";

const { onRequest } = require("firebase-functions/v2/https");
const { defineSecret } = require("firebase-functions/params");

const fireworksApiKey = defineSecret("FIREWORKS_API_KEY");

const DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1";
const MAX_BODY_CHARS = 28_000_000;
const MAX_TOKENS = 1000;
const MAX_MESSAGES = 8;

function reject(res, status, message) {
  res.status(status).json({ error: message });
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
  const bodySize = JSON.stringify(body).length;
  if (bodySize > MAX_BODY_CHARS) {
    return "Request is too large.";
  }
  return "";
}

exports.fireworksChat = onRequest(
  {
    region: "us-central1",
    timeoutSeconds: 180,
    memory: "1GiB",
    cors: false,
    secrets: [fireworksApiKey],
  },
  async (req, res) => {
    if (req.method !== "POST") {
      return reject(res, 405, "Use POST.");
    }

    const proxyToken = process.env.MODEL_PROXY_TOKEN || "";
    if (proxyToken && req.get("X-Proxy-Token") !== proxyToken) {
      return reject(res, 401, "Unauthorized.");
    }

    const validationError = validatePayload(req.body);
    if (validationError) {
      return reject(res, 400, validationError);
    }

    const requestedMaxTokens = Number(req.body.max_tokens || 700);
    const payload = {
      model: req.body.model,
      messages: req.body.messages,
      temperature: Math.min(Number(req.body.temperature || 0.2), 0.4),
      max_tokens: Math.min(requestedMaxTokens, MAX_TOKENS),
    };

    try {
      const baseUrl = process.env.FIREWORKS_BASE_URL || DEFAULT_BASE_URL;
      const response = await fetch(`${baseUrl.replace(/\/$/, "")}/chat/completions`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${fireworksApiKey.value()}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const text = await response.text();
      if (!response.ok) {
        console.error("Fireworks error", response.status, text.slice(0, 1000));
        return reject(res, response.status, "Fireworks request failed.");
      }

      const data = JSON.parse(text);
      res.status(200).json(data);
    } catch (error) {
      console.error(error);
      reject(res, 500, "Proxy request failed.");
    }
  }
);
