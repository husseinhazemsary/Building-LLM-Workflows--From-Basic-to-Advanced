# CSAI 422 – Lab Assignment 5
## 🔗 Building LLM Workflows: From Basic to Advanced

This repository contains the full implementation of Building LLM Workflows.
---

## 📌 Overview

The project demonstrates three main types of workflows using a blog post about AI in healthcare:

1. **Basic Pipeline Workflow** (with Reflexion-based self-correction)
2. **Agent-Driven Workflow** (dynamic tool usage)
3. **Comparative Evaluation System** (bonus challenge)

The workflows are implemented using a modular system of LLM tool functions, OpenAI-compatible APIs, and Python orchestration logic.

---
🧠 Challenges Faced
Getting the agent to correctly call finish required careful formatting of responses and tool call logs.

Tool schemas had to be designed with strict parameter definitions to enable function-calling via OpenAI-compatible APIs.

Reflexion required balancing self-correction logic with iteration limits to avoid infinite loops or wasted tokens.
