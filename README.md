# Research Assistant Agent

**Agentic AI (26ECAC401) · Team 7 · KLE Technological University · CSE (AI)**

Built under the name ResearchGenie, this is a local first agentic system that turns a research question into an evidence grounded, citation verified paper draft, on a laptop GPU by default, with no data leaving the machine.

---

## 1. Team

| Team name / number | Team 7 |
|---|---|
| Division | B |

| # | Name | USN | Division |
|---|------|-----|----------|
| 1 | Ayush Kabbur | 01FE23BCI033 | B |
| 2 | Owais Aminbhavi | 01FE23BCI378 | B |
| 3 | Mohammed Sadiq Lakkundi | 01FE23BCI057 | B |
| 4 | Md Abdul Raqeeb | 01FE23BCI019 | B |

---

## 2. Quick links

| What | Where |
|---|---|
| Assignments, all 14 labs | [`Assignments/`](./Assignments) |
| Course project code | [`Course-Project/code`](./Course-Project/code) |
| Problem statement document | [`Course-Project/Problem_Statement/`](./Course-Project/Problem_Statement) |
| Final course project report | [`Course-Project/Final_Report/`](./Course-Project/Final_Report) |
| Project documentation, worklet canvas, review sheet, technology report | [`Course-Project/docs/`](./Course-Project/docs) |
| Evidence, live run artifacts, raw test output | [`Course-Project/docs/evidence/`](./Course-Project/docs/evidence) |

---

## 3. Repository structure

```
ResearchGenie/
├── README.md
├── requirements.txt
├── Assignments/                 # Lab 1 to Lab 14, each with the real code embedded and a fresh captured test run
└── Course-Project/
    ├── Problem_Statement/       # the proposal document shared at the start of the course
    ├── Final_Report/            # the final course project report, as a document
    ├── code/                    # the real ResearchGenie codebase
    └── docs/                    # worklet canvas, lab review sheet, technology report, evidence/
```

---

## 4. Overview of the assignments

Every lab's folder includes the actual real source code it is built on, pasted in full, not only linked, plus a freshly captured real test run executed for this submission. Where a concept needed a full live model run rather than a unit test, the lab links to the real live run evidence already produced by the working product.

| Lab | Topic | Folder | What it demonstrates |
|---|---|---|---|
| 1 | Agent vs Chatbot | [`LAB_01`](./Assignments/LAB_01_Agent_vs_Chatbot) | Clarify, plan, revise loop in Discovery, not a one shot answer |
| 2 | Tool Using Agent | [`LAB_02`](./Assignments/LAB_02_Tool_Using_Agent) | Native tool calling for database selection, 12 of 12 valid, 11 of 12 sensible, live |
| 3 | Skill Creation | [`LAB_03`](./Assignments/LAB_03_Skill_Creation) | Typed contracts, a headless CLI, one Agent Skill for two runtimes |
| 4 | Memory and Retrieval | [`LAB_04`](./Assignments/LAB_04_Memory_and_Retrieval) | A SQLite library plus cache; seeded papers still pass the relevance gate |
| 5 | MCP Style Connector | [`LAB_05`](./Assignments/LAB_05_MCP_Connector) | 6 read only MCP tools, live verified in Claude Desktop |
| 6 | OpenCLA / OpenClaw Runtime | [`LAB_06`](./Assignments/LAB_06_OpenClaw_Runtime) | Event ledger plus two live dashboards; a real chat bot authenticated and connected end to end |
| 7 | Agentic Node Graph | [`LAB_07`](./Assignments/LAB_07_Agentic_Node_Graph) | A 9 node writing graph and a second graph for the Dev Agent |
| 8 | Parallel Swarm | [`LAB_08`](./Assignments/LAB_08_Parallel_Swarm) | Parallel workers plus a recurrence threshold, 34 real gap statements, live |
| 9 | Deep Research Agent | [`LAB_09`](./Assignments/LAB_09_Deep_Research_Agent) | 4 fresh live runs, 0 invalid references, full evidence tables |
| 10 | Cloud Operations Agent | [`LAB_10`](./Assignments/LAB_10_Cloud_Operations_Agent) | The Doctor, 12 rules, proven live: kill Ollama, it recovers |
| 11 | Agentic SDLC Factory | [`LAB_11`](./Assignments/LAB_11_Agentic_SDLC_Factory) | Dev Agent proven on a real ticket, a human reviewed fix applied through the same pipeline |
| 12 | Safety and Governance | [`LAB_12`](./Assignments/LAB_12_Safety_and_Governance) | Approval gates, versioned resume, a deterministic decision brief |
| 13 | Discipline Specific Agents | [`LAB_13`](./Assignments/LAB_13_Discipline_Specific_Agents) | Trend Advisor across 6 engineering fields, ranked by measured growth |
| 14 | Capstone Project Studio | [`LAB_14`](./Assignments/LAB_14_Capstone) | The full product, the full real test suite, run fresh for this submission |

---

## 5. Course project, ResearchGenie

### 5.1 Problem statement

Faculty research work spans literature discovery, gap identification, drafting, citation and formatting compliance, and tracking deadlines, largely done by hand and repeated for every paper. Full document: [`Course-Project/Problem_Statement/Research_Assistant_Agent_Proposal.docx`](./Course-Project/Problem_Statement/Research_Assistant_Agent_Proposal.docx).

### 5.2 Objectives

1. Turn a research question into a real, citation verified literature review draft, not a plausible sounding one.
2. Keep every citation checkable: verified, invalid, or unverifiable against real scholarly databases, never silently repaired.
3. Grade the result with a checker independent of the model that wrote it.
4. Run entirely on a local model by default, at zero cloud cost, with no data leaving the machine.
5. Self heal from real failures, such as a dead model mid run, with human confirmed fixes, not blind restarts.

### 5.3 Approach

Four fixed stages, always in order, so any run is auditable step by step. Discovery plans, searches and screens. A human approves the corpus. Writing runs a 9 node LangGraph on evidence tagged cards only. Verification checks every DOI against two databases. Quality Assurance grades the result deterministically, with no LLM in that loop. Around that core sit a Trend Advisor for what to research, a Doctor for fixing a failed run, and a Dev Agent for fixing the product's own code under test.

### 5.4 Implementation

| Layer | Where | Labs |
|---|---|---|
| Surfaces | `researchgenie/`, `frontend/`, `skills/`, two live dashboards | 3, 5, 6 |
| Orchestrator | `orchestrator/` | 6, 10, 12 |
| Agents | `services/research-discovery`, `research-writing`, `verification`, `quality-assurance`, `trend-advisor`, `dev_agent/` | 1, 2, 7, 8, 9, 11, 13 |
| Shared | `shared/contracts`, `shared/utilities` | 2, 3, 4 |

### 5.5 Results

- **559 tests passing** (see [`Assignments/LAB_14_Capstone`](./Assignments/LAB_14_Capstone) for the exact fresh run captured for this submission), CI green on every push.
- **0 invalid references** across every fresh live run; one draft honestly reported as partial, not upgraded.
- **12 of 12 valid tool plans**, 11 of 12 sensible, from the 9B local model.
- **The Doctor proven live**: a model killed mid run is diagnosed, fixed, and the run resumes to completion.
- **The Dev Agent's safety architecture proven live**: across 8 real attempts on a real ticket, it refused to overwrite tests and required red before green every single time, and the resulting fix went through the same human reviewed pipeline it is designed around.
- **OpenClaw's chat runtime proven live**: a real Telegram bot created, paired, and authenticated end to end, with two live dashboards built to show the real event flow between Telegram, the gateway, and the agent as it happens.

---

## 6. Documentation

- Worklet canvas: [`Course-Project/docs/Agentic_AI_Worklet_Canvas_ResearchGenie.pptx`](./Course-Project/docs/Agentic_AI_Worklet_Canvas_ResearchGenie.pptx)
- Lab worklet review sheet: [`Course-Project/docs/AAI_26ECAC401_Lab_Worklet_Review_Sheet.docx`](./Course-Project/docs/AAI_26ECAC401_Lab_Worklet_Review_Sheet.docx)
- Full technology report: [`Course-Project/docs/technology-report.html`](./Course-Project/docs/technology-report.html)
- Live demo runbook: [`Course-Project/docs/demo-runbook.md`](./Course-Project/docs/demo-runbook.md)
