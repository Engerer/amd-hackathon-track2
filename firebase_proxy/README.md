# Firebase Fireworks Proxy

This proxy keeps `FIREWORKS_API_KEY` out of the public Docker image.

Flow:

```text
Track 2 Docker container
 -> Firebase HTTPS function
 -> Fireworks API
 -> Firebase HTTPS function
 -> Track 2 Docker container
```

## Deploy

1. Install the Firebase CLI:

```bash
npm install -g firebase-tools
```

2. Log in:

```bash
firebase login
```

3. Set your project:

```bash
cd firebase_proxy
firebase use --add
```

4. Store the Fireworks API key as a Firebase secret:

```bash
firebase functions:secrets:set FIREWORKS_API_KEY
```

5. Install dependencies and deploy:

```bash
cd functions
npm install
cd ..
firebase deploy --only functions:fireworksChat
```

6. Copy the deployed function URL into the main project's `.env`:

```text
MODEL_PROXY_URL=https://us-central1-your-project.cloudfunctions.net/fireworksChat
FIREWORKS_API_KEY=
FIREWORKS_MODEL=accounts/fireworks/models/gemma-4-26b-a4b-it
```

## Optional Proxy Token

You can set `MODEL_PROXY_TOKEN` as a Cloud Run environment variable after deployment and send the same value from the container as `MODEL_PROXY_TOKEN`.

This protects casual misuse, but do not treat a token baked into a public Docker image as a secret.

## Shutdown

After judging, disable or delete the function and rotate the Fireworks API key.
