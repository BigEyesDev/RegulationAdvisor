# Week 5 Plan — Docker + Deployment (RegulationAdvisor v0.5.0)

**Based on:** RegulationAdvisor_MasterPlan.md — Week 5  
**Start date:** 2026-07-15  
**Target version:** v0.5.0  
**Daily time:** 3–4 hours  

**Deliverable:** Containerised app deployed to HuggingFace Spaces AND AWS ECS.  
**Branch convention:** `feat/w5-d<N>-<slug>` → merge to `dev` → merge `dev` to `main` at end of week.

---

## State Entering Week 5

| Component | Status |
|-----------|--------|
| FastAPI REST API (`/api/chat`, `/api/metrics`, `/api/evaluate`) | ✅ Done — v0.4.0 |
| Streaming SSE chat endpoint | ✅ Done |
| ChromaDB swap via `build_vector_store()` | ✅ Done |
| Gradio Evaluation Dashboard tab | ✅ Done |
| Integration test suite | ✅ Done |
| Dockerfile | ⬜ This week |
| Docker Compose (app + ChromaDB) | ⬜ This week |
| HuggingFace Spaces deploy | ⬜ This week |
| AWS ECS deployment | ⬜ This week |

---

## Day 1 — Dockerfile (branch: `feat/w5-d1-dockerfile`)

**Goal:** Single-stage Dockerfile that builds and runs the FastAPI app.

### Tasks

1. **Write `Dockerfile`** (update the skeleton already in repo):
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   # Install uv for fast dependency install
   RUN pip install uv

   COPY pyproject.toml uv.lock ./
   RUN uv sync --frozen --no-dev

   COPY src/ src/
   COPY data/ data/
   COPY evals/ evals/
   COPY scripts/ scripts/

   # Pre-build the FAISS index at image build time
   RUN uv run python scripts/ingest.py

   EXPOSE 8000
   CMD ["uv", "run", "uvicorn", "regulation_advisor.api.app:app", \
        "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Test locally:**
   ```bash
   docker build -t regulation-advisor:latest .
   docker run -p 8000:8000 --env-file .env regulation-advisor:latest
   curl http://localhost:8000/health
   ```

3. **`.dockerignore`** — exclude `data/index/`, `.venv/`, `*.pyc`, `.git`, `notebooks/`, `learning/`

4. **Gate check:** `docker run` → `GET /health` returns `{"status":"ok"}`. Gradio UI loads at `/`.

### Files touched
- `Dockerfile` (rewrite)
- `.dockerignore` (new)

---

## Day 2 — Docker Compose (branch: `feat/w5-d2-docker-compose`)

**Goal:** `docker compose up` starts the app + ChromaDB together with persistent vector storage.

### Tasks

1. **Rewrite `docker-compose.yml`:**
   ```yaml
   services:
     app:
       build: .
       ports:
         - "8000:8000"
       depends_on:
         chromadb:
           condition: service_healthy
       environment:
         - VECTOR_STORE_BACKEND=chromadb
         - CHROMA_HOST=chromadb
         - CHROMA_PORT=8000
       env_file: .env

     chromadb:
       image: chromadb/chroma:latest
       ports:
         - "8001:8000"
       volumes:
         - chroma_data:/chroma/chroma
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
         interval: 10s
         timeout: 5s
         retries: 5

   volumes:
     chroma_data:
   ```

2. **Ingest into ChromaDB on first start:**  
   Add an `entrypoint.sh` that runs `python scripts/ingest.py` if ChromaDB collection is empty,
   then starts uvicorn. This avoids re-ingesting on every restart.

3. **Test the persistence guarantee:**
   ```bash
   docker compose up -d
   # ask a question via curl
   docker compose down
   docker compose up -d
   # same question — answer should be instant (no re-embed)
   curl http://localhost:8000/health
   ```

4. **Gate check:** `docker compose down && docker compose up` — vectors still there, no re-ingestion.

### Files touched
- `docker-compose.yml` (rewrite)
- `scripts/entrypoint.sh` (new)

---

## Day 3 — HuggingFace Spaces Deploy (branch: `feat/w5-d3-hf-spaces`)

**Goal:** Live public URL on HuggingFace Spaces in under 30 minutes.

### Tasks

1. **Update `README.md` YAML front-matter** for HF Spaces:
   ```yaml
   ---
   title: RegulationAdvisor
   emoji: ⚖️
   colorFrom: blue
   colorTo: indigo
   sdk: docker
   app_port: 8000
   pinned: false
   ---
   ```
   Use `sdk: docker` — lets HF build from `Dockerfile` directly (no separate `app_file`).

2. **Secrets in HF Spaces** — add via the Spaces Settings UI (never commit to repo):
   - `OPENROUTER_API_KEY` (or whichever provider is active)
   - `TAVILY_API_KEY`
   - `VECTOR_STORE_BACKEND=faiss` (HF Spaces has no ChromaDB sidecar — use FAISS)

3. **Upload:**
   ```bash
   huggingface-cli login
   huggingface-cli upload <YOUR_USERNAME>/regulation-advisor . --repo-type=space
   ```

4. **Smoke test the live URL:**
   - `GET https://<username>-regulation-advisor.hf.space/health` → `{"status":"ok"}`
   - Open the Gradio UI, ask "What AI practices are prohibited?" — verify Article 5 cited.

5. **Add live URL to `README.md`** under a "Live Demo" heading.

6. **Gate check:** Public URL responds. Gradio chat returns cited answer within 30 seconds.

### Files touched
- `README.md` (front-matter + live URL section)
- `src/regulation_advisor/ui/app_runner.py` (ensure `VECTOR_STORE_BACKEND` env var respected)

---

## Day 4 — AWS ECR + ECS Setup (branch: `feat/w5-d4-aws-ecr-ecs`)

**Goal:** Docker image in AWS ECR, ECS task definition ready.

### Prerequisite
```bash
aws configure   # set region to eu-central-1, output json
```

### Tasks

1. **Push image to ECR:**
   ```bash
   aws ecr create-repository --repository-name regulation-advisor --region eu-central-1
   
   AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
   aws ecr get-login-password --region eu-central-1 | \
     docker login --username AWS --password-stdin \
     ${AWS_ACCOUNT}.dkr.ecr.eu-central-1.amazonaws.com
   
   docker tag regulation-advisor:latest \
     ${AWS_ACCOUNT}.dkr.ecr.eu-central-1.amazonaws.com/regulation-advisor:v0.5.0
   docker push \
     ${AWS_ACCOUNT}.dkr.ecr.eu-central-1.amazonaws.com/regulation-advisor:v0.5.0
   ```

2. **Store secrets in AWS Secrets Manager:**
   ```bash
   aws secretsmanager create-secret \
     --name regulation-advisor/openrouter-api-key \
     --secret-string "${OPENROUTER_API_KEY}"
   ```

3. **ECS Task Definition** (`infra/task-definition.json`):
   - CPU: 512 (0.5 vCPU), Memory: 1024 MB
   - Port mapping: `8000:8000`
   - Secrets: reference Secrets Manager ARNs (not plaintext env vars)
   - Log driver: `awslogs` → CloudWatch log group `/ecs/regulation-advisor`

4. **Register task definition:**
   ```bash
   aws ecs register-task-definition --cli-input-json file://infra/task-definition.json
   ```

5. **Gate check:** Task definition appears in ECS console. Image pulled successfully in a
   test `aws ecs run-task` one-shot run.

### Files touched
- `infra/task-definition.json` (new)
- `infra/README.md` (new — deployment commands reference)

---

## Day 5 — ECS Service + ALB + Final Gate Check (branch: `feat/w5-d5-ecs-service`)

**Goal:** Public HTTPS URL via Application Load Balancer. Version bump to v0.5.0.

### Tasks

1. **Create ECS Cluster:**
   ```bash
   aws ecs create-cluster --cluster-name regulation-advisor
   ```

2. **Application Load Balancer** (via AWS Console or CLI):
   - VPC: default
   - Scheme: internet-facing
   - Target group: HTTP, port 8000, health check path `/health`

3. **ECS Service:**
   ```bash
   aws ecs create-service \
     --cluster regulation-advisor \
     --service-name regulation-advisor-svc \
     --task-definition regulation-advisor \
     --desired-count 1 \
     --launch-type FARGATE \
     --network-configuration "..." \
     --load-balancers "..."
   ```

4. **Smoke test public URL:**
   ```bash
   curl https://<ALB_DNS>/health
   curl -X POST https://<ALB_DNS>/api/chat/sync \
     -H "Content-Type: application/json" \
     -d '{"message": "What is Article 5?"}'
   ```

5. **Add ALB URL to `README.md`** under "API Endpoint".

6. **Version bump:**
   - `pyproject.toml`: `0.4.0 → 0.5.0`
   - `CHANGELOG.md`: add `[0.5.0]` entry

7. **Merge dev → main** (same flow as end of Week 4).

8. **Gate check (master plan):** `https://<ALB_DNS>/api/metrics` returns RAGAS scores JSON.

### Files touched
- `infra/task-definition.json` (update with service config)
- `README.md` (ALB URL)
- `pyproject.toml` (version bump)
- `CHANGELOG.md` (v0.5.0 entry)

---

## Week 5 Deliverables Summary

| # | Deliverable | Branch |
|---|-------------|--------|
| 1 | Working `Dockerfile` — builds + runs locally | `feat/w5-d1-dockerfile` |
| 2 | `docker compose up` — app + ChromaDB with persistent volume | `feat/w5-d2-docker-compose` |
| 3 | Live HuggingFace Spaces URL | `feat/w5-d3-hf-spaces` |
| 4 | Docker image in AWS ECR + ECS task definition | `feat/w5-d4-aws-ecr-ecs` |
| 5 | Public ALB URL with `/health` and `/api/metrics` working | `feat/w5-d5-ecs-service` |
| 6 | `v0.5.0` tagged + dev merged into main | end of day 5 |

---

## AWS Skills You Gain This Week

| Service | What You Do | Why It Matters |
|---------|------------|----------------|
| **ECR** | Push Docker image | Every ML deploy on AWS starts here |
| **ECS Fargate** | Serverless container hosting | No EC2 management; standard for ML APIs |
| **ALB** | Public HTTPS URL | Load balancing, health checks, auto-restart |
| **Secrets Manager** | Store API keys | Appears in every ML engineering JD |
| **CloudWatch Logs** | `/ecs/regulation-advisor` log group | Production observability baseline |
| **IAM** | Task execution role | Required to pull from ECR and read Secrets |

---

## Cost Estimate

| Resource | Approx cost |
|----------|-------------|
| ECS Fargate (0.5 vCPU, 1 GB, 1 task) | ~$12–18/month |
| ALB | ~$18/month |
| ECR storage | ~$0.10/month |
| Secrets Manager | ~$0.40/month |
| **Total** | **~$30–40/month** |

Run it for the interview period (3–4 weeks), then `aws ecs update-service --desired-count 0`
to stop billing. Delete ALB when done — it's the biggest cost.

---

## Carry-Overs into Week 6

- `notebooks/week5_notes.md` — write after Day 5:
  - What surprised you about Docker multi-stage builds?
  - Did ChromaDB persistence work on first try?
  - What IAM permission took the longest to debug?
- `docs/smolagents_comparison.md` — still needs the "production decision" section filled in
  with Week 5 deployment context (LangGraph + FastAPI + ECS vs smolagents limitations)
