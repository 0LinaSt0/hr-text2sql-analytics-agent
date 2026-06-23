# hr-text2sql-analytics-agent

HR Enterprise Agentic Reporting via Text-to-SQL

An AI-powered agent for querying HR employee data in natural Russian language. Built with **LangGraph**, **GigaChat**, and **ClickHouse**, this system translates user questions into SQL queries and retrieves relevant information from an employee database.

## Overview

The agent follows a **graph-based pipeline** that processes a natural language question through several stages:

1. **Verification & Preparation** — checks if the database contains enough attributes to answer the question, identifies self-referential queries.
2. **Attribute Processing** — resolves organizational structures (OSS), employee names, categorical features, skills, and methodologies (e.g., Agile).
3. **Decomposition** — evaluates query complexity, optionally breaks complex questions into sub-questions, and generates sub-SQL queries.
4. **SQL Generation & Execution** — builds the final SQL query (ClickHouse syntax), executes it, handles errors, and returns an explanation.

All LLM calls use **GigaChat-2-Max** for inference and **EmbeddingsGigaR** for semantic retrieval.

## Project Structure

```
src/hrp_agent_text2sql/
├── nodes/                              # LangGraph pipeline nodes
│   ├── preparations_inspections.py     # Feature & self-info checks
│   ├── attributes_processing.py        # Names, OSS, skills, categorical features
│   ├── decompose.py                    # Complexity detection & decomposition
│   └── generation_execution.py         # SQL generation, execution, error handling
├── promts/                             # LLM prompts (Russian)
│   ├── main.py                         # Core SQL generation prompt
│   ├── add.py                          # Additional (auxiliary) prompts
│   ├── decompose.py                    # Decomposition prompts
│   └── tasks.py                        # Task-specific prompts
├── schemas/                            # Data models & state
│   ├── agent_context.py                # DB connection, custom retriever, context
│   ├── agent_state.py                  # LangGraph state definition
│   ├── db_info.py                      # Column metadata for the target table
│   └── structured_output.py            # Pydantic models for structured LLM output
├── utils/
│   ├── oss.py                          # Organizational structure search & resolution
│   ├── sql.py                          # SQL parsing & validation
│   ├── person.py                       # Employee name matching & FIO resolution
│   ├── parser_processing.py            # Robust JSON/pydantic parsing
│   └── node_log.py                     # Node execution tracing
├── custom_retrievers/                  # Precomputed embedding bases
│   ├── oss_ebase_embs.json             # OSS structure embeddings
│   └── skills_ebase_embs.json          # Skill name embeddings
├── data/
│   ├── hr_sample.sqlite                # Syntetic dataset for example
│   ├── oss_ebase.json                  # Organizational structure hierarchy
│   └── skills.json                     # Skill reference list
├── agent.py                            # AnAgent class — graph builder & entry point
├── config.py                           # API key & feature flags
├── errors.py                           # Custom error types
└── gigamodels.py                       # GigaChat model initialization
```

Also includes syntetic dataset for example

## Technologies

| Component       | Technology                                     |
|----------------|-------------------------------------------------|
| LLM            | GigaChat-2-Max, EmbeddingsGigaR                 |
| Framework      | LangChain, LangGraph                            |
| Database       | ClickHouse (via clickhouse-connect)             |
| Vector Search  | FAISS, cosine similarity + BM25 ensemble        |
| Language       | Python ≥ 3.9                                    |


## Key Features

- **Natural language → SQL** — ask questions in plain Russian about employees
- **Organizational structure resolution** — fuzzy matching of department/branch names via vector search
- **Skill lookup** — semantic search over the skill reference base
- **Complex query decomposition** — breaks multi-part questions into individual SQL queries
- **Self-info detection** — handles questions about the current user's data
- **Error recovery** — automatic SQL regeneration on ClickHouse errors
- **Row-Level Security (RLS)** — restricts data access based on user context