"""
MCP Router — Simplified
========================
How it works:
1. User types a query
2. Groq LLM decides which server to use (Atlassian, DuckDuckGo, or Golden Dataset)
3. The selected server fetches the data
4. Groq LLM answers the query using that data

Servers available:
- mcp-atlassian      : handles Jira and Confluence queries via MCP protocol
- duckduckgo         : handles general web search queries (free, no API key)
- golden-dataset-mcp : manages version-controlled golden datasets and RAG evaluation
"""

import sys
import os
import asyncio
import re
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Load credentials from .env file
load_dotenv()

# ── Groq LLM setup ────────────────────────────────────────────────────────────
# This is the LLM we use for both routing decisions and final answers
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0  # 0 = more factual, less creative
)

# ── Atlassian MCP Server config ───────────────────────────────────────────────
# StdioServerParameters tells the MCP client how to launch the mcp-atlassian server
# It runs as a subprocess and communicates via stdin/stdout (stdio transport)
ATLASSIAN_MCP = StdioServerParameters(
    command="mcp-atlassian",        # the CLI command to start the server
    args=["--transport", "stdio"],  # use stdio (stdin/stdout) to communicate
    env={
        **os.environ,               # pass all current env vars
        "JIRA_URL": os.getenv("JIRA_BASE_URL"),
        "JIRA_USERNAME": os.getenv("JIRA_EMAIL"),
        "JIRA_API_TOKEN": os.getenv("JIRA_API_TOKEN"),
        "CONFLUENCE_URL": f"{os.getenv('JIRA_BASE_URL')}/wiki",
        "CONFLUENCE_USERNAME": os.getenv("JIRA_EMAIL"),
        "CONFLUENCE_API_TOKEN": os.getenv("JIRA_API_TOKEN"),
    },
)

# ── Golden Dataset MCP Server config ────────────────────────────────────────
GOLDEN_DATASET_PATH = os.getenv(
    "GOLDEN_DATASET_PATH",
    r"c:\Users\Nipun\Documents\llm-eval-data-cetric-dataset-curation\golden_dataset"
)

GOLDEN_DATASET_MCP = StdioServerParameters(
    command=r"C:\Users\Nipun\AppData\Local\Programs\Python\Python312\Scripts\golden-dataset-mcp.exe",
    args=[],
    env={**os.environ},
)

# ── Conversation history ──────────────────────────────────────────────────────
# Stores last few Q&A turns so follow-up queries like "what about that?" work
history = []


# ── Step 1: Route — decide which server to use ────────────────────────────────
def select_server(query: str) -> str:
    """Ask Groq which server should handle this query."""
    decision = llm.invoke(
        f"""You are a router. Decide which server should handle the user query.

Servers:
- atlassian      : Jira tickets, issues, bugs, sprints, Confluence pages, docs
- duckduckgo     : general web search, news, facts, anything outside Jira/Confluence
- golden-dataset : golden dataset operations — list entries, evaluate answers, check dataset status, commit versions, add/update entries

Reply with ONLY one word: atlassian, duckduckgo, or golden-dataset

Query: {query}"""
    ).content.strip().lower()

    if "golden" in decision or "dataset" in decision:
        server = "golden-dataset"
    elif "duckduckgo" in decision:
        server = "duckduckgo"
    else:
        server = "atlassian"
    print(f"[Router] → {server}")
    return server


# ── Step 2a: Fetch data from DuckDuckGo ──────────────────────────────────────
def search_web(query: str) -> str:
    """Search the web using DuckDuckGo. Free, no API key needed."""
    results = DDGS().text(query, max_results=5)
    if not results:
        return "No results found."
    # format results as plain text for the LLM
    return "\n\n".join(
        f"Title: {r['title']}\nURL: {r['href']}\nSummary: {r['body']}"
        for r in results
    )


# ── Step 2b: Fetch data from Atlassian MCP ───────────────────────────────────
def pick_atlassian_tool(query: str, tool_names: list) -> tuple:
    """
    Decide which Atlassian MCP tool to call and with what arguments.
    Returns (tool_name, arguments_dict).
    """
    q = query.lower()

    # if query has a quoted title e.g. "My Page Title" → search Confluence for it
    quoted = re.findall(r'["\u201c\u201d](.*?)["\u201c\u201d]', query)
    if quoted and "confluence_search" in tool_names:
        return "confluence_search", {"query": quoted[0]}

    # if query mentions confluence/page/doc → search all Confluence pages
    confluence_words = {"confluence", "page", "pages", "wiki", "doc", "document", "space"}
    if any(w in q for w in confluence_words) and "confluence_search" in tool_names:
        return "confluence_search", {"query": "type=page"}

    # if query wants to create a ticket → use jira_create_issue
    if any(w in q.split() for w in {"create", "add", "new", "make"}) and "jira_create_issue" in tool_names:
        # strip common words to extract the ticket summary
        skip = {"create", "add", "new", "make", "a", "jira", "ticket", "issue", "for", "the"}
        summary = " ".join(w for w in query.split() if w.lower() not in skip) or query
        return "jira_create_issue", {
            "project_key": os.getenv("JIRA_PROJECT", "ADPE"),
            "summary": summary,
            "issue_type": "Task"
        }

    # default → search all Jira tickets in the project
    return "jira_search", {
        "jql": f"project = {os.getenv('JIRA_PROJECT', 'ADPE')} ORDER BY created DESC",
        "limit": 50
    }


# ── Step 2c: Golden Dataset interactive CLI ──────────────────────────────────
GOLDEN_COMMANDS = """
  Commands:
    init                  — initialise a new dataset
    add                   — add a question/answer entry
    list                  — list all working-tree entries
    commit                — commit working tree as a new version
    status                — show dataset status
    diff  <v1> <v2>       — diff two committed versions
    delete <entry_id>     — delete an entry by id
    evaluate              — evaluate actual answers against committed version
    back                  — return to main router
"""


async def run_golden_dataset_cli():
    """Interactive CLI for all golden-dataset operations, backed by the MCP server."""
    print(GOLDEN_COMMANDS)
    async with stdio_client(GOLDEN_DATASET_MCP) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            while True:
                cmd = input("  golden> ").strip().lower()
                if not cmd:
                    continue
                if cmd == "back":
                    break

                elif cmd == "init":
                    name = input("    Dataset name: ").strip() or "golden-dataset"
                    desc = input("    Description : ").strip()
                    result = await session.call_tool("init_dataset", {"input": {
                        "dataset_path": GOLDEN_DATASET_PATH, "name": name, "description": desc
                    }})

                elif cmd == "add":
                    while True:
                        question = input("    Question (or 'done'): ").strip()
                        if question.lower() == "done":
                            break
                        answer = input("    Answer              : ").strip()
                        tags = input("    Tags (comma-sep, optional): ").strip()
                        args = {"dataset_path": GOLDEN_DATASET_PATH, "question": question, "answer": answer}
                        if tags:
                            args["tags"] = [t.strip() for t in tags.split(",")]
                        result = await session.call_tool("add_entry", {"input": args})
                        print(f"    ✓ {' '.join(block.text for block in result.content if hasattr(block, 'text'))}")
                    continue

                elif cmd == "list":
                    version = input("    Version (leave blank for working tree): ").strip() or None
                    args = {"dataset_path": GOLDEN_DATASET_PATH}
                    if version:
                        args["version"] = version
                    result = await session.call_tool("list_entries", {"input": args})

                elif cmd == "commit":
                    desc = input("    Commit message: ").strip()
                    result = await session.call_tool("commit_version", {"input": {
                        "dataset_path": GOLDEN_DATASET_PATH, "description": desc
                    }})

                elif cmd == "status":
                    result = await session.call_tool("dataset_status", {"input": {
                        "dataset_path": GOLDEN_DATASET_PATH
                    }})

                elif cmd.startswith("diff"):
                    parts = cmd.split()
                    v1 = parts[1] if len(parts) > 1 else input("    v1: ").strip()
                    v2 = parts[2] if len(parts) > 2 else input("    v2: ").strip()
                    result = await session.call_tool("diff_versions", {"input": {
                        "dataset_path": GOLDEN_DATASET_PATH, "v1": v1, "v2": v2
                    }})

                elif cmd.startswith("delete"):
                    parts = cmd.split()
                    entry_id = parts[1] if len(parts) > 1 else input("    Entry ID: ").strip()
                    result = await session.call_tool("delete_entry", {"input": {
                        "dataset_path": GOLDEN_DATASET_PATH, "entry_id": entry_id
                    }})

                elif cmd == "evaluate":
                    version = input("    Version to evaluate (leave blank for latest): ").strip() or None
                    print("    Enter actual answers (one per line, 'done' to finish):")
                    actual_answers = []
                    while True:
                        ans = input("      > ").strip()
                        if ans.lower() == "done":
                            break
                        actual_answers.append(ans)
                    args = {"dataset_path": GOLDEN_DATASET_PATH, "actual_answers": actual_answers}
                    if version:
                        args["version"] = version
                    result = await session.call_tool("evaluate_answers", {"input": args})

                else:
                    print(f"    Unknown command: '{cmd}'. Type 'back' to exit.")
                    continue

                output = "\n".join(block.text for block in result.content if hasattr(block, "text"))
                print(f"\n  {output}\n")


async def fetch_from_golden_dataset(query: str) -> str:
    """Entry point — always launches the interactive golden dataset CLI."""
    await run_golden_dataset_cli()
    return None


async def fetch_from_atlassian(query: str) -> str:
    """Launch the mcp-atlassian server, call the right tool, return the result."""
    # stdio_client launches mcp-atlassian as a subprocess
    async with stdio_client(ATLASSIAN_MCP) as (read, write):
        # ClientSession handles the MCP protocol handshake
        async with ClientSession(read, write) as session:
            await session.initialize()  # MCP handshake

            # get the list of all tools the server exposes
            tools = (await session.list_tools()).tools
            tool_names = [t.name for t in tools]

            # pick the right tool for this query
            tool_name, tool_args = pick_atlassian_tool(query, tool_names)
            print(f"[Atlassian MCP] Calling: {tool_name}")

            # call the tool and extract text from the response
            result = await session.call_tool(tool_name, tool_args)
            return "\n".join(
                block.text for block in result.content if hasattr(block, "text")
            )


# ── Step 3: Answer — send context + query to Groq ────────────────────────────
def answer_with_groq(query: str, context: str) -> str:
    """Build a prompt with conversation history and get an answer from Groq."""
    # include last 3 turns of conversation so follow-ups work
    history_text = "".join(f"Q: {t['query']}\nA: {t['answer']}\n\n" for t in history[-3:])

    prompt = (
        f"{history_text}"
        f"Context:\n{context[:8000]}\n\n"  # limit to 8000 chars to avoid token limit
        f"Question: {query}\nAnswer:"
    )
    return llm.invoke(prompt).content


# ── Main runner ───────────────────────────────────────────────────────────────
async def run_query_async(query: str) -> str:
    # Step 1: route
    server = select_server(query)

    # Step 2: fetch data from the right server
    if server == "duckduckgo":
        context = search_web(query)
    elif server == "golden-dataset":
        context = await fetch_from_golden_dataset(query)
    else:
        context = await fetch_from_atlassian(query)

    print(f"[Context] {len(context) if context else 0} chars fetched{', asking Groq...' if context else ', done.'}")

    # Step 3: answer (skip Groq for golden-dataset — result already printed)
    if context is None:
        return ""
    answer = answer_with_groq(query, context)

    # save to history for follow-up queries
    history.append({"query": query, "answer": answer})
    return answer


def run_query(query: str) -> str:
    # asyncio.run() lets us call the async function from normal (sync) code
    return asyncio.run(run_query_async(query))


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("MCP Router — Active Servers:")
    print("  [1] mcp-atlassian      → Jira + Confluence")
    print("  [2] duckduckgo         → Web Search (free)")
    print("  [3] golden-dataset-mcp → Golden Datasets & RAG Evaluation")
    print("\nType your query, 'exit' to quit.\n")

    while True:
        query = input("Query: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break
        result = run_query(query)
        if result:
            print(f"\n  Answer: {result}\n")
