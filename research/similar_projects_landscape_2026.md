# Similar Projects & Landscape Research (March 2026)

> **Note:** This research was originally compiled for the PIRDLY project and predates FeMind's creation as a standalone crate. References to PIRDLY throughout reflect the original research context.

This document catalogs projects with philosophies, patterns, and architectures relevant to autonomous AI coding orchestrator design. Projects are grouped by the design insight they offer.

---

## Table of Contents

1. [Multi-Agent Orchestration Platforms](#1-multi-agent-orchestration-platforms)
2. [AI Agent Memory & Context Systems](#2-ai-agent-memory--context-systems)
3. [Task Planning & Execution Frameworks](#3-task-planning--execution-frameworks)
4. [Rust AI Agent Frameworks](#4-rust-ai-agent-frameworks)
5. [Context Management & Session Patterns](#5-context-management--session-patterns)
6. [Key Design Insights for PIRDLY](#6-key-design-insights-for-pirdly)

---

## 1. Multi-Agent Orchestration Platforms

### Ruflo (formerly Claude Flow)
- **GitHub:** https://github.com/ruvnet/ruflo
- **Stars:** ~19,000+
- **Description:** The leading multi-agent orchestration platform for Claude Code. Deploys 60+ agents in coordinated swarms with shared memory, persistent workflows, and RAG across codebases.
- **Key Architecture:**
  - Orchestrator assigns tasks and monitors agents
  - Memory bank with CRDT-based shared knowledge
  - Terminal manager for shell sessions
  - Task scheduler with prioritized queues and dependency tracking
  - Self-learning neural capabilities that learn from every task execution
  - 215 MCP tools, 8 AgentDB controllers
- **PIRDLY Insight:** Ruflo's evolution from "Claude Flow" to a 250K+ line TypeScript/WASM codebase shows the scale these orchestrators can reach. Their CRDT-based shared memory and AgentDB controller patterns are relevant to PIRDLY's two-tier memory design. However, Ruflo is deeply Claude-specific — PIRDLY's multi-CLI approach (Claude, Codex, Gemini) is a differentiator. Ruflo's self-learning from task execution patterns directly parallels PIRDLY's learning/memory system.

### OpenHands
- **GitHub:** https://github.com/All-Hands-AI/OpenHands
- **Stars:** 65,000+
- **Description:** Open-source platform for cloud coding agents. Model-agnostic, enterprise-ready. Scale from one to thousands of agents with open SDKs, APIs, and micro-agents.
- **Key Philosophy:** Agents should be composable micro-units that can scale horizontally. Model-agnostic design lets you swap LLM backends.
- **PIRDLY Insight:** OpenHands proves the market for model-agnostic agent orchestration. Their micro-agent architecture — small, focused agents composed into larger workflows — validates PIRDLY's approach of using specialized CLI tools rather than a monolithic agent. Their enterprise-scale thinking (thousands of agents) shows where the industry is heading.

### BMAD (Breakthrough Method for Agile AI-Driven Development)
- **Description:** A complete framework orchestrating 12+ specialized AI agents for different roles — Product Manager, Architect, Scrum Master, Developer, QA.
- **Key Philosophy:** Model the software development lifecycle as a team of role-specific agents, each with its own persona and toolset.
- **PIRDLY Insight:** BMAD's role-based agent design is interesting but may be over-engineered for PIRDLY's use case. However, the concept of specialized personas for different phases (discovery, planning, execution) aligns with PIRDLY's phase-based approach. The key takeaway: don't just orchestrate code generation — orchestrate the entire SDLC.

### Claude Squad
- **GitHub:** https://github.com/smtilson/claude-squad
- **Description:** Terminal app that multiplexes Claude Code instances — spawn multiple agents working concurrently in separate tmux panes, each on a different task.
- **Key Philosophy:** Simple multiplexing over complex orchestration. Let each agent work independently with minimal coordination overhead.
- **PIRDLY Insight:** Claude Squad is the simplest possible multi-agent approach — just parallel tmux sessions. PIRDLY adds structured coordination, memory, and verification on top of this basic pattern. Claude Squad validates that developers want multi-instance workflows but shows the gap: no shared memory, no learning, no structured planning.

### Gas Town (Steve Yegge) & OpenClaw (Peter Steinberger)
- **Description:** Multi-agent orchestration systems for Claude Code with persistent work tracking. Made "AI agent orchestrators" feel mainstream.
- **Key Architecture:** Under the hood: LLM API call + state loop + tools + memory + orchestration — a small set of primitives wired together.
- **PIRDLY Insight:** The core insight that agent orchestrators are fundamentally simple primitives composed together validates PIRDLY's architecture. The differentiator is how well you compose them, not how complex you make them.

### Continuous-Claude-v2
- **GitHub:** https://github.com/parcadei/Continuous-Claude-v2
- **Description:** Context management for Claude Code. Hooks maintain state via ledgers and handoffs. MCP execution without context pollution. Agent orchestration with isolated context windows.
- **Key Pattern:** Agents spawn with fresh context for complex tasks that would degrade in a compacted context. They return a summary and optionally create handoffs.
- **PIRDLY Insight:** Directly implements the "fresh context window" pattern that PIRDLY's research on Ralph Loop identified. The ledger/handoff system for state persistence between fresh contexts is exactly what PIRDLY's file-based state management achieves. Worth studying their hook-based approach vs. PIRDLY's external orchestration.

### ComposioHQ Agent Orchestrator
- **GitHub:** https://github.com/ComposioHQ/agent-orchestrator
- **Description:** Manages fleets of AI coding agents in parallel. Each agent gets its own git worktree, branch, and PR. Agent-agnostic (Claude Code, Codex, Aider), runtime-agnostic (tmux, Docker), tracker-agnostic (GitHub, Linear). Has 8 swappable plugin slots.
- **Key Architecture:** Plugin-based with swappable slots for agent runtimes, SCM platforms, and task trackers.
- **PIRDLY Insight:** Their plugin architecture for swapping agent runtimes, SCM platforms, and task trackers is a strong pattern. PIRDLY's CLI profile system maps well to this. The multi-tool agnosticism (Claude, Codex, Aider) validates PIRDLY's multi-CLI approach.

### Overstory
- **GitHub:** https://github.com/jayminwest/overstory
- **Description:** Multi-agent orchestration via git worktrees and tmux. Coordinates agents through a custom SQLite mail system. Pluggable `AgentRuntime` interface supports Claude Code, Pi, Gemini CLI.
- **Key Innovation:** SQLite-based inter-agent messaging system for coordination between isolated agent sessions.
- **PIRDLY Insight:** The SQLite-based inter-agent messaging is clever — agents communicate through database records rather than shared memory. Also explicitly warns about compounding error rates in agent swarms, a useful caution for PIRDLY's orchestrator design.

---

## 2. AI Agent Memory & Context Systems

### MemOS (Memory Operating System)
- **GitHub:** https://github.com/MemTensor/MemOS
- **Description:** A Memory Operating System for LLMs and AI agents. Unifies store/retrieve/manage for long-term memory with context-aware interactions.
- **Key Architecture:**
  - Unified Memory API (add, retrieve, edit, delete)
  - Memory structured as an inspectable, editable graph
  - Multi-modal memory (text, images, tool traces, personas)
  - Local plugin with on-device SQLite, hybrid search (FTS5 + vector)
  - Task summarization and skill evolution
- **PIRDLY Insight:** MemOS's use of SQLite with FTS5 + vector hybrid search directly validates PIRDLY's decision to start with FTS5 and add embeddings later. Their "skill evolution" concept — agents learning reusable patterns from completed tasks — maps to PIRDLY's learnings system. The graph-structured memory is more sophisticated than PIRDLY's current flat design and could be a future enhancement.

### MemoRAG
- **GitHub:** https://github.com/qhjqhj00/MemoRAG
- **Description:** RAG framework built on a super-long memory model. Achieves global understanding of entire databases by recalling query-specific clues from memory. Accepted at TheWebConf 2025.
- **Key Innovation:** Instead of standard RAG (query → retrieve → answer), MemoRAG builds a global memory model first, then uses it to generate better retrieval clues.
- **PIRDLY Insight:** MemoRAG's two-phase approach (build global understanding, then targeted retrieval) is relevant to PIRDLY's project discovery phase. After discovery, PIRDLY could build a holistic project model that improves subsequent context injection quality rather than just doing keyword-based FTS5 lookups.

### memU
- **GitHub:** https://github.com/NevaMind-AI/memU
- **Description:** Memory framework for 24/7 proactive agents. Reduces LLM token cost for always-on agents. Continuously captures user intent — agents can anticipate what you're about to do.
- **Key Philosophy:** Memory should enable proactive agent behavior, not just reactive recall. Agents should understand ongoing user intent.
- **PIRDLY Insight:** PIRDLY's learning system is reactive (store what happened, recall when relevant). memU's proactive intent detection could inspire PIRDLY to analyze patterns of what users typically need next in a project phase and pre-fetch relevant context.

### LightRAG
- **GitHub:** https://github.com/HKUDS/LightRAG
- **Stars:** Large (EMNLP 2025 paper)
- **Description:** Simple and fast RAG with knowledge graph management. Supports creating, editing, and deleting entities and relationships. Comes with Web UI and API.
- **PIRDLY Insight:** LightRAG's knowledge graph approach could enhance PIRDLY's memory system — instead of flat learnings, structure them as a graph of concepts, errors, patterns, and relationships. Their "simple and fast" philosophy aligns with PIRDLY's pragmatic FTS5-first approach.

### OMEGA Memory
- **GitHub:** https://github.com/omega-memory/core
- **Description:** Persistent memory for AI coding agents via MCP server with 25 tools. LongMemEval: 76.8% verified (marketing claims 95.4%, unverified best-of-8). Stores decisions, lessons, error patterns with semantic search. SQLite + CPU-only ONNX embeddings, fully local. Features memory decay, graph relationships, and encryption at rest.
- **Key Innovation:** "Forgetting intelligence" — memory decay with exemptions for error patterns. ~8ms embedding time and <50ms query latency set a performance benchmark.
- **PIRDLY Insight:** Architecture is remarkably close to PIRDLY's memory design — SQLite-backed, local-first, with error pattern tracking and cross-session learning. The memory decay concept (less-used learnings fade while validated patterns persist) is something PIRDLY should consider. The performance benchmarks (<50ms queries) set a target.

### Mem0
- **GitHub:** https://github.com/mem0ai/mem0
- **Stars:** ~37,000+
- **Description:** The most popular open-source memory layer for LLM applications. Provides persistent, contextual memory that evolves with each user interaction.
- **PIRDLY Insight:** The most popular project in this space. Its cloud-dependent model contrasts with PIRDLY's local-first approach, but its API design for memory storage/retrieval is worth studying as a reference for ergonomic memory interfaces.

### Engram
- **GitHub:** https://github.com/Gentleman-Programming/engram
- **Description:** Persistent memory system for AI coding agents. Agent-agnostic Go binary with SQLite + FTS5, MCP server, HTTP API, CLI, and TUI.
- **PIRDLY Insight:** Technically very close to PIRDLY's planned memory system (SQLite + FTS5, CLI interface, agent-agnostic). Written in Go, but the feature set overlap is significant — a direct competitor in the memory space. PIRDLY's Rust implementation and two-tier global/project split are differentiators.

### Agent Memory Paper List
- **GitHub:** https://github.com/Shichun-Liu/Agent-Memory-Paper-List
- **Description:** Curated academic survey distinguishing Agent Memory from RAG and Context Engineering. Covers memory forms, functions, and dynamics.
- **PIRDLY Insight:** Useful reference for understanding the theoretical landscape. The distinction between memory forms (episodic, semantic, procedural) could help PIRDLY categorize its learnings more effectively — e.g., "error patterns" are procedural memory, "project context" is semantic, "session history" is episodic.

---

## 3. Task Planning & Execution Frameworks

### Claude Task Master
- **GitHub:** https://github.com/eyaltoledano/claude-task-master
- **Description:** Task management system for AI-driven development. Without task structure, Claude tries to solve everything at once and loses the thread. With TaskMaster, the agent works on one clearly defined task at a time. Reports ~90% fewer errors.
- **Key Philosophy:** Structured task decomposition prevents context thrashing. One task at a time with clear boundaries.
- **PIRDLY Insight:** Validates PIRDLY's task-by-task execution approach. The 90% error reduction stat is a strong signal that structured task management is essential. PIRDLY's acceptance criteria verification before task completion adds an extra layer that Task Master doesn't have.

### Claude Task Manager
- **GitHub:** https://github.com/vibehat/claude-task-manager
- **Description:** AI agent orchestration coordinating multiple AI models across workflows, with personal project memory so agents never forget decisions, context, or progress. Integrates with Claude Code, Cursor, VS Code.
- **Key Innovation:** Built on Claude Task Master as a core engine but adds cross-tool orchestration and persistent project memory.
- **PIRDLY Insight:** Shows the natural evolution from task management to full orchestration with memory — exactly PIRDLY's trajectory. Their multi-tool integration (Claude Code, Cursor, VS Code) parallels PIRDLY's multi-CLI profile system.

### GitHub Copilot Coding Agent
- **Description:** GitHub's autonomous coding agent. Assign a GitHub issue to Copilot, and it spins up an ephemeral dev environment, checks out the repo, creates a branch, codes, runs tests/linters, and opens a PR.
- **Key Philosophy:** Ephemeral, isolated environments per task. Full CI/CD integration. Issue-driven workflow.
- **PIRDLY Insight:** Copilot's ephemeral environment pattern parallels PIRDLY's researched worktree-per-task isolation from Auto-Claude. The issue-driven workflow is a different entry point than PIRDLY's discovery-based approach but could be supported as a project type.

### Plandex
- **GitHub:** https://github.com/plandex-ai/plandex
- **Stars:** ~15,000
- **Description:** Terminal-based AI coding agent for large projects. Plans and executes multi-step tasks across dozens of files. Features a diff review sandbox, version control for plans, automated debugging, and configurable autonomy (full auto to fine-grained control). Written in Go.
- **Key Innovation:** Plan versioning with branches — explore alternative approaches without losing previous plans. Diff review sandbox keeps AI changes separate until approved.
- **PIRDLY Insight:** Plandex's plan branching is excellent — PIRDLY could adopt the idea of versioned plans to explore alternatives. The diff sandbox (staging AI changes for review before applying) is a strong UX pattern. Their chat-to-tell mode transition mirrors PIRDLY's discovery-to-execution flow.

### MetaGPT
- **GitHub:** https://github.com/FoundationAgents/MetaGPT
- **Stars:** ~50,000+
- **Description:** Multi-agent framework where "Code = SOP(Team)". Assigns roles (product manager, architect, engineer) to GPTs. Returns PRD, Design, Tasks, or Repo from a one-line requirement.
- **Key Philosophy:** Standard operating procedures (SOPs) drive agent coordination — formalize how agents interact rather than letting them figure it out.
- **PIRDLY Insight:** MetaGPT's SOP-driven approach is the most mature "idea to code" pipeline. Their role-based agent specialization could inform PIRDLY's planning phase. The one-line-requirement-to-repo pipeline is the ultimate simplification of what PIRDLY aims to do.

### Devika
- **GitHub:** https://github.com/stitionai/devika
- **Stars:** ~18,000+
- **Description:** Open-source autonomous software engineer (Devin alternative). Breaks down high-level instructions into steps, researches information, and writes code. Supports Claude, GPT-4, Gemini, Mistral, and local LLMs.
- **PIRDLY Insight:** Devika uses direct API calls while PIRDLY uses external CLI orchestration — a fundamental architectural difference. PIRDLY's zero-token orchestration overhead is a structural advantage. However, Devika's dynamic agent state tracking and step decomposition UX are worth studying.

### Agentic Project Management (APM)
- **GitHub:** https://github.com/sdi2200262/agentic-project-management
- **Description:** AI workflow framework bringing project management principles to AI-assisted workflows. Follows: Setup Phase (Discovery & Planning) → Task Loop Phase (Plan Execution). Supports Cursor, Copilot, Claude Code.
- **PIRDLY Insight:** The discovery-then-task-loop architecture is essentially PIRDLY's planned workflow. Their approach to context retention across long sessions addresses the same problem PIRDLY's memory system targets.

### GitHub Spec Kit
- **Description:** GitHub's open-source toolkit for spec-driven development. Start with a spec (contract), then use AI agents (Copilot, Claude Code, Gemini CLI) to generate, test, and validate code against it.
- **PIRDLY Insight:** Spec-driven development is conceptually similar to PIRDLY's PRD-based planning. The idea that the spec is the source of truth that agents validate against reinforces PIRDLY's acceptance criteria verification pattern.

---

## 4. Rust AI Agent Frameworks

### AutoAgents (Liquidos AI)
- **GitHub:** https://github.com/liquidos-ai/AutoAgents
- **Description:** Modular multi-agent framework in Rust. Type-safe agent model with structured tool calling, configurable memory, and pluggable LLM backends. Built on Ractor (actor framework).
- **Key Architecture:**
  - Actor-based agent model (via Ractor)
  - Type-safe tool calling
  - Configurable memory per agent
  - Pluggable LLM backends
  - WASM support for browser deployment
- **PIRDLY Insight:** Most directly relevant Rust framework. The actor model (Ractor) for agent management is worth evaluating — it provides natural isolation, message passing, and supervision trees. PIRDLY could benefit from an actor-based architecture for managing concurrent CLI tool executions. Their type-safe tool calling pattern aligns with PIRDLY's Rust philosophy.

### Kowalski
- **GitHub:** https://github.com/yarenty/kowalski
- **Description:** Rust-native agentic AI framework. Zero Python dependencies, compiles to standalone binaries. Federation crate enables lightweight multi-agent workflows — pipeline automations, task delegation, agent-to-agent communication.
- **Key Philosophy:** Pure Rust, no Python dependency, modular submodules (federation, orchestration, pipelines).
- **PIRDLY Insight:** Kowalski's zero-Python philosophy matches PIRDLY's pure-Rust approach. Their federation crate concept — a dedicated module for multi-agent coordination — could inform PIRDLY's crate structure. The pipeline automation pattern is relevant to PIRDLY's phase-based execution.

### rs-agent (Lattice AI)
- **GitHub:** https://github.com/Protocol-Lattice/rs-agent
- **Description:** Production-ready Rust agent orchestrator. Pluggable LLMs, tool calling (UTCP), retrieval-capable memory, CodeMode execution, multi-agent coordination. Feature-flagged adapters for Gemini, Ollama, Anthropic, OpenAI.
- **Key Innovation:** Feature-flagged LLM adapters behind a common trait — compile with only the backends you need.
- **PIRDLY Insight:** rs-agent's feature-flag approach for LLM backends is elegant — PIRDLY could use similar feature flags for CLI tool support (claude, codex, gemini). Their common LLM trait design is a good pattern for PIRDLY's executor profiles.

### AutoGPT (Rust)
- **GitHub:** https://github.com/kevin-rs/autogpt
- **Description:** Pure Rust framework for building AGI. 8 built-in specialized autonomous agents. No-code agent configs via declarative YAML.
- **PIRDLY Insight:** The YAML-based agent configuration is interesting for PIRDLY's CLI profile system — letting users define custom tool profiles declaratively rather than through code.

### Anda Framework
- **GitHub:** https://github.com/ldclabs/anda
- **Description:** Rust AI agent framework for decentralized, autonomous agents with perpetual memory. Built for ICP and TEEs.
- **PIRDLY Insight:** While the decentralized/TEE aspects aren't relevant, Anda's "perpetually memorizing" design philosophy aligns with PIRDLY's persistent learning system. Their approach to agent autonomy — agents that can operate independently with their own memory — mirrors PIRDLY's goal of autonomous project execution.

### Rig
- **GitHub:** https://github.com/0xPlaygrounds/rig
- **Stars:** ~1,700+
- **Description:** The leading Rust framework for building LLM-powered applications. Modular abstractions over LLM providers and vector stores. Growing ecosystem including terminal coding agents.
- **PIRDLY Insight:** The most mature Rust LLM library. PIRDLY might use it for any direct LLM integration (e.g., plan generation), though PIRDLY's external orchestration model may not need it. Worth knowing as a dependency option.

### graniet/llm
- **GitHub:** https://github.com/graniet/llm
- **Description:** Rust library and CLI to unify multiple LLM backends (OpenAI, Claude, Gemini, Ollama) with a single API. Supports reactive agents that cooperate via shared memory.
- **PIRDLY Insight:** The unified API across multiple LLM providers and shared-memory agent cooperation model are relevant. Could serve as a reference for PIRDLY's CLI profile abstraction layer.

### Swarm (fcn06)
- **GitHub:** https://github.com/fcn06/swarm
- **Description:** Rust SDK for building agent networks using MCP and Agent-to-Agent (A2A) protocols. Can execute predefined plans or generate them dynamically.
- **PIRDLY Insight:** Built around open standards (MCP, A2A) for interoperability. PIRDLY's planned MCP server could benefit from studying this project's protocol implementation.

### rs-graph-llm
- **GitHub:** https://github.com/a-agmon/rs-graph-llm
- **Description:** High-performance framework for multi-agent workflow systems in Rust. LangGraph-inspired but Rust-native, using Rig for LLM integration.
- **PIRDLY Insight:** Graph-based workflow execution could inform PIRDLY's task dependency management in the orchestrator phase.

---

## 5. Context Management & Session Patterns

### The Fresh Context Pattern (Industry-Wide)
Multiple projects and workflows now explicitly manage context window degradation:
- **Ralph Loop:** External controller restarts sessions to get fresh context windows
- **GSD:** Sub-agents with fresh context to avoid accumulation
- **Continuous-Claude-v2:** Hook-based ledger system for context handoffs
- **Ruflo:** Real-time memory management that archives, optimizes, and restores context

**PIRDLY Insight:** Fresh context management is now a recognized industry pattern. PIRDLY's external orchestration approach (spawning CLI processes) gives it natural context freshness — each CLI invocation gets a fresh window. This is a structural advantage over in-process orchestrators that must manage context manually.

### Orchestration Topology Patterns
Documented patterns from Claude Code's agent teams and community projects:

| Pattern | Description | PIRDLY Relevance |
|---------|-------------|-----------------|
| **Leader-Worker** | One orchestrator, multiple specialists | PIRDLY's primary model |
| **Task Pool** | Workers self-assign from queue | Could work for parallel task phases |
| **Pipeline** | Sequential processing with handoffs | Maps to PIRDLY's phase transitions |
| **Consensus** | Multiple agents propose, leader picks best | Useful for plan generation |
| **Worker-Watcher** | Worker does task, watcher monitors for safety | Maps to PIRDLY's verification step |

### VoltAgent
- **GitHub:** https://github.com/VoltAgent/voltagent
- **Description:** Open-source TypeScript AI agent engineering platform. Memory adapters for cross-run persistence. Retriever agents for RAG grounding. Guardrails, workflows, MCP, voice support.
- **PIRDLY Insight:** VoltAgent's "memory adapter" pattern — pluggable storage backends for agent memory — is a clean abstraction PIRDLY could adopt. Their guardrails concept (validation before agent actions) parallels PIRDLY's acceptance criteria verification.

---

## 6. Key Design Insights for PIRDLY

### What the Landscape Validates

1. **External orchestration is the right approach.** Projects like Ralph Loop, GSD, and Continuous-Claude-v2 all converge on the pattern of external control with fresh context windows — exactly what PIRDLY does by spawning CLI processes.

2. **Persistent memory is a differentiator.** Most orchestrators (Claude Squad, basic multi-agent setups) lack cross-session learning. PIRDLY's two-tier memory system (global + project) is more sophisticated than most competitors.

3. **FTS5 is a valid starting point.** MemOS uses SQLite FTS5 + vector hybrid search, validating PIRDLY's decision to start with FTS5 and add embeddings later.

4. **Task-by-task execution prevents errors.** Claude Task Master's 90% error reduction with structured task management confirms PIRDLY's approach.

5. **Multi-CLI support is a gap in the market.** Most orchestrators are Claude-specific (Ruflo, Claude Squad) or model-agnostic at the API level (OpenHands). PIRDLY's approach of orchestrating CLI tools (Claude Code, Codex CLI, Gemini CLI) is unique.

### What PIRDLY Could Learn

1. **Actor-based architecture** (from AutoAgents/Ractor): Consider using Rust actors for managing concurrent CLI executions — provides natural isolation, supervision, and message passing.

2. **Knowledge graph memory** (from LightRAG, MemOS): Structure learnings as a graph of concepts, errors, and relationships rather than flat records. This enables richer queries and pattern discovery.

3. **Proactive memory** (from memU): Don't just recall on demand — analyze patterns to anticipate what context will be needed next in a given project phase.

4. **CRDT-based shared state** (from Ruflo): For future multi-agent parallelism, CRDTs enable conflict-free state merging without coordination overhead.

5. **Feature-flagged backends** (from rs-agent): Use Rust feature flags for CLI tool support — compile with only the backends you need.

6. **YAML-based profiles** (from AutoGPT Rust): Let users define custom CLI tool profiles declaratively.

7. **Skill evolution** (from MemOS): Track not just what was learned but how learnings evolve over time — a learning that's been validated 10 times is more reliable than one seen once.

8. **Worker-Watcher pattern** (from Claude Code teams): PIRDLY's verification step could be formalized as a separate watcher agent that monitors task execution for quality and safety.

9. **Memory decay** (from OMEGA Memory): Implement "forgetting intelligence" — learnings that haven't been accessed or validated decay over time, while frequently-confirmed error patterns are preserved. Prevents memory bloat.

10. **Plan versioning/branching** (from Plandex): Allow users to branch plans and explore alternative approaches without losing previous iterations. Version control for the plan itself, not just the code.

11. **SOP-driven coordination** (from MetaGPT): Formalize agent interaction patterns as standard operating procedures rather than ad-hoc coordination. Makes orchestration predictable and debuggable.

12. **SQLite inter-agent messaging** (from Overstory): Use SQLite as a message bus between isolated agent sessions — fits naturally with PIRDLY's existing SQLite-based architecture.

### What PIRDLY Already Does Better

1. **Start-to-finish journey:** Most orchestrators focus on execution only. PIRDLY covers idea → discovery → plan → execution → completion.

2. **Zero-token orchestration overhead:** External CLI orchestration costs zero LLM tokens for coordination, unlike in-process orchestrators.

3. **Multi-CLI agnosticism:** Supporting Claude, Codex, and Gemini CLIs through profiles is unique in the landscape.

4. **Two-tier memory with error classification:** Separating global patterns from project-specific context, with transient/quota/permanent error categorization, is more sophisticated than competitors.

5. **Verification before completion:** The acceptance criteria check before marking tasks complete is a quality gate most orchestrators lack.

---

## Competitive Positioning

PIRDLY occupies a unique position by combining traits that no single competitor fully covers:

| Trait | PIRDLY | Closest Competitor |
|-------|--------|-------------------|
| Rust-native orchestrator | Yes | Loom (Rust + Claude Code) |
| Zero-token orchestration | Yes | GSD (same philosophy) |
| Two-tier memory (global + project) | Yes | OMEGA (single-tier, but close) |
| Idea-to-completion lifecycle | Yes | Plandex (plan-to-execute only) |
| Multi-CLI-tool agnostic | Yes | Overstory, agent-orchestrator (but no memory) |
| Verification before completion | Yes | Spec Kit (spec-driven validation) |
| Error classification (transient/quota/permanent) | Yes | None found |

---

## Projects to Monitor

| Project | URL | Watch For |
|---------|-----|-----------|
| Ruflo | https://github.com/ruvnet/ruflo | Memory architecture, MCP patterns |
| OpenHands | https://github.com/All-Hands-AI/OpenHands | Scale patterns, micro-agent design |
| MemOS | https://github.com/MemTensor/MemOS | Memory graph design, FTS5+vector |
| OMEGA Memory | https://github.com/omega-memory/core | Memory decay, performance benchmarks |
| Mem0 | https://github.com/mem0ai/mem0 | Memory API design |
| Plandex | https://github.com/plandex-ai/plandex | Plan versioning, diff sandbox UX |
| MetaGPT | https://github.com/FoundationAgents/MetaGPT | SOP-driven agent coordination |
| AutoAgents | https://github.com/liquidos-ai/AutoAgents | Rust actor patterns, type-safe tools |
| Kowalski | https://github.com/yarenty/kowalski | Rust federation, pipeline patterns |
| rs-agent | https://github.com/Protocol-Lattice/rs-agent | Feature-flagged LLM adapters |
| Rig | https://github.com/0xPlaygrounds/rig | Rust LLM abstractions |
| Overstory | https://github.com/jayminwest/overstory | SQLite inter-agent messaging |
| Continuous-Claude-v2 | https://github.com/parcadei/Continuous-Claude-v2 | Context handoff patterns |
| Engram | https://github.com/Gentleman-Programming/engram | SQLite+FTS5 memory patterns |
| LightRAG | https://github.com/HKUDS/LightRAG | Knowledge graph RAG |
| Claude Task Master | https://github.com/eyaltoledano/claude-task-master | Task decomposition UX |

### Curated Lists for Ongoing Discovery

| List | URL | Covers |
|------|-----|--------|
| awesome-claude-code | https://github.com/hesreallyhim/awesome-claude-code | Claude Code ecosystem |
| awesome-agent-orchestrators | https://github.com/andyrewlee/awesome-agent-orchestrators | Agent orchestration tools |
| awesome-ai-agents | https://github.com/e2b-dev/awesome-ai-agents | Autonomous AI agents |
| Awesome-Agent-Memory | https://github.com/TeleAI-UAGI/Awesome-Agent-Memory | Memory systems & papers |

---

*Research compiled March 2026. The existing research in this directory (Ralph Loop, GSD, Auto-Claude, Loom) remains the foundational inspiration. This document extends that research with the broader 2026 landscape covering 30+ projects across 5 categories.*
