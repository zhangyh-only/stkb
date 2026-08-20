.PHONY: install infra dev check clean-infra

install:
	test -f .env || cp .env.example .env
	cd services && uv sync --dev
	cd web && pnpm install

infra:
	docker compose --env-file .env -f infra/compose.yaml up -d --wait

dev: infra
	@set -a; . ./.env; set +a; \
	(cd services && uv run uvicorn app.main:app --reload --host "$${STKB_API_HOST:-0.0.0.0}" --port "$${STKB_API_PORT:-8000}") & \
	api_pid=$$!; \
	(cd web && pnpm dev --host 0.0.0.0 --port "$${STKB_WEB_PORT:-5173}") & \
	web_pid=$$!; \
	trap 'kill $$api_pid $$web_pid 2>/dev/null || true' INT TERM EXIT; \
	wait

check:
	cd services && uv run ruff check .
	cd services && uv run pytest
	cd web && pnpm build

clean-infra:
	docker compose --env-file .env -f infra/compose.yaml down
