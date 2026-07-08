# Cloudflare Fireworks Proxy

Cloudflare Worker proxy for Track 2 inference. This keeps the Fireworks API key out of the public Docker image.

Flow:

```text
Track 2 Docker container
 -> Cloudflare Worker
 -> Fireworks API
 -> Cloudflare Worker
 -> Track 2 Docker container
```

The Fireworks API key is stored as a Cloudflare Worker secret, not in GitHub or Docker.

## Setup

1. Create or sign in to a Cloudflare account.

2. Install dependencies:

```bash
npm install
```

3. Log in:

```bash
npx wrangler login
```

4. Store the Fireworks key as a Worker secret:

```bash
npx wrangler secret put FIREWORKS_API_KEY
```

5. Deploy:

```bash
npx wrangler deploy
```

6. Copy the Worker URL into the main `.env`:

```text
MODEL_PROXY_URL=https://track2-fireworks-proxy.<your-subdomain>.workers.dev
FIREWORKS_API_KEY=
FIREWORKS_MODEL=accounts/fireworks/models/qwen3p7-plus
FIREWORKS_CAPTION_MODEL=accounts/fireworks/models/glm-5p2
FIREWORKS_RERANK_MODEL=accounts/fireworks/models/glm-5p2
```

## Optional Proxy Token

You can also add a lightweight proxy token:

```bash
npx wrangler secret put MODEL_PROXY_TOKEN
```

Then set the same value as `MODEL_PROXY_TOKEN` in the Docker image or local `.env`.

Do not treat this token as a perfect secret if it is baked into a public Docker image. It mainly protects against casual misuse.

## Shutdown

After judging, delete the Worker or remove the `FIREWORKS_API_KEY` secret, then rotate the Fireworks API key.
