# Robotics Foundation Model Paper Library

A curated collection of research papers, code analysis notes, and learning roadmaps covering the path from CS (NLP/CV) foundation models to robotics control.

## Structure

```
paper/
├── foundation_model/          # CS -> Robotics foundation model pipeline
│   ├── LLM/                   # LLM foundations (Transformer, GPT, Qwen, Kimi-K2)
│   ├── methods/               # 17 core method papers with code + notes
│   └── surveys/               # 10 survey papers
├── humanoid/                  # Humanoid whole-body control (12 papers)
├── manip/                     # Dexterous manipulation (20+ papers)
├── html2aitext_convert/       # arxiv HTML -> Markdown tool
├── papers.yaml                # Manifest: all paper metadata, arxiv IDs, repo URLs
├── scripts/setup.sh           # One-click: clone all repos + fetch papers
└── CLAUDE.md                  # Paper acquisition workflow + full index
```

## What's Tracked in Git

| Content | Examples |
|---------|---------|
| Analysis notes (`*_notes.md`) | 50+ paper analysis notes (Chinese + English terms) |
| Index files (`CLAUDE.md`) | Paper index, workflow documentation |
| Learning roadmap | `CS2Robotics_Roadmap.md` - 7-level progressive reading guide |
| Manifest | `papers.yaml` - all arxiv IDs + 60+ repo URLs |
| Arxiv markdown | Converted paper text via `html2aitext_convert` |
| Tool source | `html2aitext_convert/src/`, `scripts/` |

## What's NOT Tracked (Regenerable)

| Content | How to Regenerate |
|---------|------------------|
| 60+ cloned code repos | `./scripts/setup.sh --repos-only` |
| PDF files | `curl -sL https://arxiv.org/pdf/<id>` |
| HTML cache | `./html2aitext_convert/arxiv2md.sh <id>` |
| Large datasets (DexCanvas parquet etc.) | See original repo instructions |

## Quick Start

```bash
# Clone this repo
git clone <url> paper && cd paper

# Install dependencies
pip install pyyaml

# Clone all 60+ code repos (shallow) + fetch available arxiv papers
./scripts/setup.sh

# Or separately:
./scripts/setup.sh --repos-only
./scripts/setup.sh --papers-only
```

## Learning Roadmap

See [`foundation_model/CS2Robotics_Roadmap.md`](foundation_model/CS2Robotics_Roadmap.md) for a 7-level progressive reading guide:

```
Level 0: Representation & Encoding (Bengio, Transformer)
Level 1: LLM (GPT series, Chinchilla, Qwen, Kimi-K2)
Level 2: Vision-Language + Generative (ViT, CLIP, DDPM, Flow Matching, DiT)
Level 3: RL/Robotics meets Transformer (Decision Transformer, DreamerV3, ACT, Diffusion Policy)
Level 4: VLA unified models (RT-1, RT-2, Octo, OpenVLA, PaliGemma, pi_0, GR00T N1)
Level 5: Surveys (8 survey papers for global perspective)
```

## Notes Format

Each `*_notes.md` follows a standard 8-section format:

1. Core Problem
2. Method Overview (architecture, pipeline, key formulas)
3. Key Designs (2-3 most important contributions)
4. Experiments (main results, ablations)
5. Related Work Analysis
6. Limitations & Future Directions
7. **Paper vs Code Discrepancies** (critical: what the paper didn't say)
8. Cross-Paper Comparison

Written in Chinese with English technical terms. No emoji.

## Fetch a New Paper

```bash
# 1. Convert arxiv paper to markdown
bash html2aitext_convert/arxiv2md.sh <arxiv_id>

# 2. Create folder: <category>/<YY>_<ShortName>/
mkdir -p foundation_model/methods/25_NewPaper

# 3. Copy markdown + clone code
cp html2aitext_convert/output/<title>.md foundation_model/methods/25_NewPaper/
git clone --depth 1 <repo_url> foundation_model/methods/25_NewPaper/<repo_name>

# 4. Add entry to papers.yaml
# 5. Write <ShortName>_notes.md
# 6. Update CLAUDE.md index
```
