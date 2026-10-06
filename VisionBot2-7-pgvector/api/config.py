"""App configuration from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    llm_model: str = "gpt-4o"
    llm_mini_model: str = "gpt-4o-mini"
    llm_hyde_model: str = "gpt-5.2"   # HyDE generation — stronger model improves retrieval quality
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072

    # Zendesk
    zendesk_api_key: str
    zendesk_email: str
    zendesk_subdomain: str

    # Qdrant (local file mode — no server/Docker needed)
    qdrant_path: str = "./qdrant_data"
    qdrant_problems_collection: str = "ticket_problems"
    qdrant_resolutions_collection: str = "ticket_resolutions"

    # Postgres / pgvector (v7-pgvector branch)
    postgres_url: str = ""  # e.g. postgresql://user@localhost:5432/supportiq

    # Slack
    slack_bot_token: str = ""
    slack_app_token: str = ""

    # Internal log analysis tools
    pai_api_key: str = ""
    dobby_api_key: str = ""

    # Atlassian (Jira + Confluence)
    atlassian_base_url: str = ""
    atlassian_email: str = ""
    atlassian_api_key: str = ""

    # API
    api_secret_token: str
    frontend_url: str = "http://localhost:5173"

    # Retrieval tuning
    top_k_retrieve: int = 20
    top_k_rerank: int = 8
    confidence_threshold: float = 0.35
    resolution_boost: float = 1.3
    rrf_k: int = 60

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
