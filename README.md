# MCP Multi-Server Router

A lightweight LLM-based router that dispatches user queries to the right MCP (Model Context Protocol) server — Atlassian (Jira/Confluence), DuckDuckGo web search, or a golden dataset manager for RAG evaluation — using Groq's Llama 3.3 for both routing decisions and final answers.

## How It Works

1. **User submits a query** via CLI
2. **Groq LLM routes it** — decides whether the query belongs to Atlassian, DuckDuckGo, or the Golden Dataset server
3. **Selected server fetches data** — the appropriate MCP server (or web search) is called to retrieve relevant context
4. **Groq LLM answers** — the final response is generated using the retrieved context, with the last 3 conversation turns included for follow-up support

```
User Query
    │
    ▼
┌─────────────┐
│  Router LLM  │  (decides: atlassian / duckduckgo / golden-dataset)
└─────────────┘
    │
    ├──► mcp-atlassian (Jira/Confluence via MCP)
    ├──► DuckDuckGo Search (free web search)
    └──► golden-dataset-mcp (dataset versioning + RAG eval)
    │
    ▼
┌─────────────┐
│  Answer LLM  │  (synthesizes final answer from context)
└─────────────┘
```

## Servers

| Server | Purpose |
|---|---|
| `mcp-atlassian` | Handles Jira ticket/issue queries and Confluence page search via the MCP protocol |
| `duckduckgo-search` | Free, no-API-key web search for general queries |
| `golden-dataset-mcp` | Manages version-controlled golden datasets — add/list/commit entries, diff versions, evaluate answers against a committed dataset for RAG evaluation |

## Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com)
- Atlassian (Jira/Confluence) credentials, if using the Atlassian routing path
- `golden-dataset-mcp` installed and accessible on your `PATH` (or update the path in the script), if using the Golden Dataset routing path

## Installation

```bash
pip install python-dotenv mcp langchain-groq duckduckgo-search mcp-atlassian golden-dataset-mcp
```

## Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
JIRA_BASE_URL=https://yourcompany.atlassian.net
JIRA_EMAIL=your_email@company.com
JIRA_API_TOKEN=your_jira_api_token
JIRA_PROJECT=ADPE
GOLDEN_DATASET_PATH=/path/to/your/golden_dataset   # optional — has a fallback default
```

> ⚠️ **Never commit your `.env` file.** Use the included `.env.example` as a template and keep real credentials local only.

## Usage

```bash
python mcp_router.py
```

```
MCP Router — Active Servers:
  [1] mcp-atlassian      → Jira + Confluence
  [2] duckduckgo         → Web Search (free)
  [3] golden-dataset-mcp → Golden Datasets & RAG Evaluation

Type your query, 'exit' to quit.

Query: what jira tickets were created this week?
[Router] → atlassian
[Atlassian MCP] Calling: jira_search
[Context] 1204 chars fetched, asking Groq...

  Answer: ...
```

### Golden Dataset Commands

When the router selects the golden-dataset path, it drops into an interactive CLI:

| Command | Description |
|---|---|
| `init` | Initialize a new dataset |
| `add` | Add a question/answer entry |
| `list` | List all working-tree entries |
| `commit` | Commit the working tree as a new version |
| `status` | Show dataset status |
| `diff <v1> <v2>` | Diff two committed versions |
| `delete <entry_id>` | Delete an entry by ID |
| `evaluate` | Evaluate actual answers against a committed version |
| `back` | Return to the main router |

## Project Structure

```
.
├── mcp_router.py       # Main router script
├── .env.example        # Template for required environment variables
├── .env                # Your local credentials (gitignored, not committed)
└── README.md
```

## Notes

- Conversation history is limited to the last 3 turns to keep prompts within token limits.
- Context passed to the answer LLM is truncated to 8,000 characters.
- The Golden Dataset path is fully interactive and does not use the answer LLM — results are printed directly from the MCP server.

## License

_Add your license here (e.g. MIT)._
