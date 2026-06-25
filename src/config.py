"""
Centralized configuration management for Pair AI
Consolidates all environment variables and default values
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Azure OpenAI Configuration
# ---------------------------------------------------------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZUREOPENAIENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZUREOPENAIKEY")
DEPLOYMENT_NAME_GPT41 = os.getenv("DEPLOYMENTNAMEGPT")
API_VERSION = os.getenv("APIVERSION")

# ---------------------------------------------------------------------------
# Azure Storage Configuration
# ---------------------------------------------------------------------------
AZURE_DOC_INTEL_ENDPOINT = os.getenv("DOC_INTEL_ENDPOINT")
AZURE_DOC_INTEL_CLIENT_KEY = os.getenv("DOC_INTEL_CLIENT_KEY")

# ---------------------------------------------------------------------------
# Azure Storage Configuration
# ---------------------------------------------------------------------------
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT", "sashareddev001")
PAIRAIDEV_CONTAINER_NAME= os.getenv("PAIRAIDEV_CONTAINER_NAME", "pair-ai-dev-pair-ai-dev")

# ---------------------------------------------------------------------------
# Azure Authentication Configuration
# ---------------------------------------------------------------------------
AI_PLATFORM_SCOPE = os.getenv("AI_PLATFORM_SCOPE")
AZURE_FEDERATED_TOKEN_FILE = os.getenv("AZURE_FEDERATED_TOKEN_FILE")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")  # Optional: Required for some blob storage scenarios

# ---------------------------------------------------------------------------
# Application Configuration
# ---------------------------------------------------------------------------
# Default LLM parameters (for general chat, simple Q&A)
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.5

# GPT-4.1 specific limits
GPT41_MAX_INPUT_TOKENS = 1_000_000  # 1 million input token context window
GPT41_MAX_OUTPUT_TOKENS = 16_384    # Max completion tokens

# AI Validator settings (for document validation with full SWPPP context)
# Context: Full SWPPP documents (50-200 pages) + rule prompts fit within 1M input
# Response: Structured JSON with section findings, page numbers, reasoning
AI_VALIDATOR_MAX_TOKENS = 1000      # More tokens needed for detailed validation responses
AI_VALIDATOR_TEMPERATURE = 0.0      # Deterministic validation (same as DEFAULT)

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
