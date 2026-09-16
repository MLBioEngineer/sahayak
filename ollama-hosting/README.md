# Self-Hosted Ollama Hosting Guide for সহায়ক (Sahayak)

This guide provides 3 budget-conscious ways to host Ollama without relying on third-party AI APIs (OpenAI, Anthropic, etc.).

---

## Option 1: Hugging Face Spaces (Free Tier — 16 GB RAM) [Recommended]

Render's free tier provides 512 MB RAM, which is insufficient for 3B models. Hugging Face Spaces provides **16 GB RAM and 2 vCPUs completely free**.

### Steps to Deploy:
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Name your space (e.g. `sahayak-ollama`).
3. Select License: **MIT** or **Apache 2.0**.
4. Select Space SDK: **Docker** -> **Blank**.
5. Select Hardware: **CPU Basic (2 vCPU, 16 GB RAM) - Free**.
6. Set Visibility: **Public** (or Private with Hugging Face Token).
7. Push the files in `ollama-hosting/` (`Dockerfile` and `entrypoint.sh`) to the Space:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/sahayak-ollama
   cp ollama-hosting/* sahayak-ollama/
   cd sahayak-ollama
   git add .
   git commit -m "Deploy Ollama with Llama 3.2 3B"
   git push
   ```
8. Once built and running, your Ollama public URL will be:
   `https://<your-username>-sahayak-ollama.hf.space`
9. In Render.com Dashboard -> **sahayak-backend** -> **Environment**:
   Set `OLLAMA_BASE_URL` = `https://<your-username>-sahayak-ollama.hf.space`

---

## Option 2: Cloudflare Tunnel to Local Machine (100% Free, Instant)

If your local computer runs Ollama smoothly, you can expose it securely via a free HTTPS Cloudflare Tunnel without any port forwarding:

1. Download `cloudflared` from [Cloudflare](https://github.com/cloudflare/cloudflared/releases).
2. Start Ollama locally:
   ```bash
   ollama serve
   ollama pull llama3.2:3b
   ```
3. In another terminal, run:
   ```bash
   cloudflared tunnel --url http://localhost:11434
   ```
4. Cloudflare will output an HTTPS URL like:
   `https://random-words.trycloudflare.com`
5. In Render.com Dashboard -> **Environment**:
   Set `OLLAMA_BASE_URL` = `https://random-words.trycloudflare.com`

---

## Option 3: Lightweight VPS (Hetzner / DigitalOcean / Oracle Free Tier)

1. On a Linux server (Ubuntu 22.04 / 24.04 with at least 4GB RAM):
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama run llama3.2:3b
   ```
2. Configure systemd to listen on `0.0.0.0:11434`:
   Edit `/etc/systemd/system/ollama.service.d/override.conf`:
   ```ini
   [Service]
   Environment="OLLAMA_HOST=0.0.0.0:11434"
   Environment="OLLAMA_ORIGINS=*"
   ```
3. Restart Ollama:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart ollama
   ```
4. Point Render backend's `OLLAMA_BASE_URL` to `http://<your-vps-ip>:11434`.
