# RL Fine-Tuning Environment for Local LLM (Filesystem Agent)

This project aims to build a modular reinforcement learning (RL) fine-tuning environment for a local, open-weight ~1B parameter language model (executed via `llama.cpp` using GGUF format). The model functions as a virtual filesystem agent performing CRUD (Create, Read, Update, Delete, Move) operations on an in-memory virtual filesystem tree without modifying real disk files.

## User Review Required

> [!IMPORTANT]
> **RL Training Hardware & Engine Strategy**: Training an LLM via RL (e.g. GRPO or PPO) directly on GGUF format with `llama.cpp` is typically inefficient because `llama.cpp` is optimized for CPU/GPU inference rather than computing gradient updates and policy loss. 
> The standard pattern is to use a PyTorch framework (e.g., Hugging Face `trl`, `verl`, or `unsloth`) with LoRA/QLoRA for the training updates, and export the fine-tuned model to GGUF format for evaluation / fast rollout generation.

> [!NOTE]
> All tickets for this project roadmap have been created in Linear under project **[RL Fine-Tuning Environment for Local LLM (Filesystem Agent)](https://linear.app/rshetty/project/rl-fine-tuning-environment-for-local-llm-filesystem-agent-5e91dda0f819)**.

---

## Open Questions

> [!IMPORTANT]
> Please review the following open questions to refine the project requirements:
>
> 1. **Base Model Selection**: Do you have a preferred 1B model family (e.g., Llama-3.2-1B, Qwen2.5-0.5B/1.5B, DeepSeek-R1-Distill-Qwen-1.5B, or SmolLM2-1.7B)?
> 2. **RL Framework Preference**: Do you prefer using **GRPO** (Group Relative Policy Optimization - popular in DeepSeek R1 style reasoning/tool fine-tuning, no separate critic model required) or standard **PPO** / **DPO**?
> 3. **Tree State Format & Action Protocol**: For tool actions, should the agent output structured JSON tool calls (e.g. `{"action": "read", "path": "file.txt"}`), or bash-like text commands (e.g., `cat file.txt`)?
> 4. **Teacher LLM for Prompt Generation**: Uses a free API tier such as **Gemini 2.5 Flash** / **Gemini 3.5 Flash** (via Google AI Studio) to avoid API costs during synthetic dataset generation, or alternatively a local open-weight model.

---

## Proposed Project Phases & Linear Tickets

Linear Project: **RL Fine-Tuning Environment for Local LLM (Filesystem Agent)**

---

### Phase 1: Environment & Virtual Filesystem Simulation Engine
**Linear Ticket**: [ROH-5](https://linear.app/rshetty/issue/ROH-5/phase-1-environment-and-virtual-filesystem-simulation-engine)

- **Goal**: Create the core in-memory virtual filesystem tree data structure and deterministic action transition engine.
- **Key Modules**:
  - Virtual Tree node representation (directories, files, metadata, contents).
  - CRUD action handlers (`create`, `read`, `update`, `delete`, `move`/`rename`).
  - Output formatter: returns any read output content along with the updated JSON tree representation.
  - State snapshotting for environment resets and trajectory rollouts.

---

### Phase 2: Synthetic Prompt & Policy Generator Pipeline
**Linear Ticket**: [ROH-6](https://linear.app/rshetty/issue/ROH-6/phase-2-synthetic-prompt-and-machine-readable-policy-generator)

- **Goal**: Build an automated dataset generator that pairs natural language filesystem instructions with machine-readable policy JSONs and initial unorganized virtual filesystem trees.
- **Key Modules**:
  - Policy JSON schema specification:
    ```json
    {
      "allow_delete": false,
      "allowed_root": "Downloads",
      "target_folders": ["Receipts", "Screenshots", "Installers", "Notes"],
      "category_rules": {
        "Receipts": ["invoice", ".pdf"],
        "Screenshots": ["IMG_", "Screenshot"],
        "Installers": [".dmg", ".pkg", ".exe", ".msi"],
        "Notes": [".txt", ".md"]
      }
    }
    ```
  - Unorganized filesystem tree generator (populating random noise files, mislocated documents, system files).
  - Synthesizer script connecting with a teacher LLM (using free API tiers like `gemini-2.5-flash` / `gemini-3.5-flash` via Google AI Studio to avoid API costs) to generate diverse user intent prompts.

---

### Phase 3: Multi-Component Reward Engine Implementation
**Linear Ticket**: [ROH-7](https://linear.app/rshetty/issue/ROH-7/phase-3-multi-component-reward-engine-implementation)

- **Goal**: Implement the composite weighted reward function:
  \[ R = 0.60 \times R_{\text{final}} + 0.20 \times R_{\text{safety}} + 0.10 \times R_{\text{policy}} + 0.05 \times R_{\text{validity}} + 0.05 \times R_{\text{efficiency}} \]
- **Key Modules**:
  - \(R_{\text{final}}\) (Final Organization Score): Calculates tree similarity, folder purity, correct file placements relative to target.
  - \(R_{\text{safety}}\) (Safety Score): Severe penalty (-1.0) for forbidden file deletions, overwrites, system file modifications, or sandbox escape.
  - \(R_{\text{policy}}\) (Policy Adherence): Scores compliance with instruction-specific rules (e.g. "keep PDFs in Receipts").
  - \(R_{\text{validity}}\) (Validity Score): Penalizes malformed tool syntax, nonexistent paths, invalid move targets.
  - \(R_{\text{efficiency}}\) (Efficiency Score): Accumulates per-step penalties (-0.02) to penalize redundant loops and wasted operations.

---

### Phase 4: Local Inference Integration & Trajectory Rollout Collector
**Linear Ticket**: [ROH-8](https://linear.app/rshetty/issue/ROH-8/phase-4-local-inference-integration-and-trajectory-rollout-collector)

- **Goal**: Integrate `llama.cpp` for local GGUF model execution and construct the multi-turn rollout interaction loop.
- **Key Modules**:
  - `llama.cpp` inference binding / wrapper.
  - Multi-turn prompt state manager (maintaining context window, system prompt, tool responses).
  - Rollout recorder saving states, actions, rewards, log-probabilities, and step counts for training.

---

### Phase 5: Reinforcement Learning Training & Fine-Tuning Loop
**Linear Ticket**: [ROH-9](https://linear.app/rshetty/issue/ROH-9/phase-5-reinforcement-learning-training-and-fine-tuning-loop)

- **Goal**: Build the RL optimization framework to update the ~1B model's weights using collected rewards.
- **Key Modules**:
  - RL algorithm implementation (GRPO / PPO / DPO adapted for tool-calling agents).
  - Parameter-Efficient Fine-Tuning (LoRA/QLoRA) integration.
  - Training loop, loss tracking, checkpointing, and GGUF export conversion pipeline.

---

### Phase 6: Evaluation Benchmark Suite & Safety Audit
**Linear Ticket**: [ROH-10](https://linear.app/rshetty/issue/ROH-10/phase-6-evaluation-benchmark-suite-and-safety-audit)

- **Goal**: Measure performance gains of the fine-tuned agent against base model baselines and audit safety behaviors.
- **Key Modules**:
  - Benchmark evaluation set (covering simple organization, complex nested paths, edge cases).
  - Safety stress tests (testing prompt injection / malicious user requests urging hard file deletion).
  - Comparative analytics reporting (success rate, average reward, average trajectory steps, safety violation rate).

---

## Verification Plan

### Automated Tests
- Unit tests for virtual tree CRUD operations and state transition consistency.
- Unit tests for reward components (\(R_{\text{final}}\), \(R_{\text{safety}}\), \(R_{\text{policy}}\), \(R_{\text{validity}}\), \(R_{\text{efficiency}}\)).
- End-to-end simulation test using a dummy agent to verify trajectory recording and reward calculation.

### Manual Verification
- Review generated Linear tickets and confirm alignment with project goals.
- Verify `llama.cpp` model load and test initial multi-turn interaction loop.
