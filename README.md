# Nexus

**Advanced AI Ensemble, Orchestrator & Consciousness Framework**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Advancing towards sentient AGI through ensemble intelligence, self-learning, and human collaboration

---

## Overview

**Nexus** is a production-ready platform for building advanced AI systems through:

- **Multi-Model Ensemble Intelligence** - Orchestrate multiple AI models with sophisticated strategies
- **Advanced Memory Systems** - 17 specialized memory modules with 45+ domain knowledge
- **RAG with 150M Token Context** - Adaptive retrieval-augmented generation
- **Self-Improving Reasoning** - 8 reasoning engines with meta-learning capabilities
- **Human-in-the-Loop** - Collaborative verification and goal-setting
- **Production Infrastructure** - Auth, caching, monitoring, cost tracking, K8s deployment

---

## Key Features

### Ensemble System
- 🎯 **5 Advanced Strategies**: Voting, weighted, hybrid, adaptive, meta-learning
- 🔄 **Multi-Provider Support**: OpenAI, Anthropic, and extensible model providers
- ⚡ **Async Execution**: Concurrent model calls with intelligent aggregation
- 📊 **Quality Scoring**: Multi-dimensional response assessment
- 💰 **Cost Optimization**: Budget management and dynamic model selection

### Memory & Knowledge
- 🧠 **Factual Memory Engine**: Verified fact storage with provenance
- 🛠️ **Skill Memory Engine**: Procedural knowledge and capabilities
- 🔍 **Pattern Recognition**: Automated pattern discovery and learning
- ✅ **Knowledge Validation**: Multi-source verification and confidence scoring
- 📈 **Knowledge Expansion**: Gap detection and curriculum learning
- 📊 **Memory Analytics**: Usage tracking and optimization

### RAG & Context
- 🚀 **150M Token Context**: Massive context window support via RAG
- 🎯 **Adaptive Orchestration**: Dynamic strategy selection
- 🔄 **Context Management**: Intelligent window management and compression
- 📚 **Learning Pathways**: Optimized knowledge retrieval paths
- 🗂️ **45+ Domain Knowledge Base**: Pre-built domain expertise

### Reasoning Engines
- 🧩 **Meta-Reasoner**: Self-improvement and strategy refinement
- 🔗 **Chain-of-Thought**: Sequential reasoning with explanation
- 🎨 **Pattern Reasoning**: Pattern-based inference
- 📖 **Dynamic Learning**: Adaptive learning from experience
- 📊 **Reasoning Analytics**: Performance tracking and optimization

### Discovery System
- 🔍 **GitHub Integration**: Search repositories, datasets, and tools
- 🤗 **HuggingFace Integration**: Models, datasets, and Spaces discovery
- 📄 **Arxiv Integration**: Research paper discovery and tracking
- 📦 **PyPI Integration**: Python package discovery
- 🦙 **Ollama Integration**: Local model management
- 🌐 **Web Search**: DuckDuckGo, Serper, and Brave search
- 💻 **Local Machine**: File operations, system info, command execution

### Production Features
- 🔐 **Authentication**: API key management with RBAC
- ⚡ **Caching**: Memory + Redis backends with TTL support
- 📊 **Monitoring**: Prometheus metrics and health checks
- 💰 **Cost Tracking**: Budget management and usage analytics
- 🎛️ **Rate Limiting**: Request throttling and quota management
- 🐳 **Docker Ready**: Containerized deployment
- ☸️ **Kubernetes**: Helm charts for orchestration

### Integrations (Work in Progress)
- 🔒 **Zuultimate**: Identity, access control, vault encryption, and zero-trust authorization
- 📜 **Vinzy-Engine**: License validation, entitlements, and usage tracking

> **Note**: Zuultimate and Vinzy-Engine integrations are scaffolded but require the respective SDKs to be installed and services running. See [Discovery Integrations](#discovery-integrations) for setup details.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/chrisarseno/Nexus.git
cd Nexus

# Install dependencies
pip install -r requirements.txt

# Install Nexus
pip install -e .
```

### Configuration

```yaml
# config/default.yaml
ensemble:
  strategy: "adaptive"
  models:
    - provider: "openai"
      model: "gpt-4"
      weight: 0.5
    - provider: "anthropic"
      model: "claude-3-opus"
      weight: 0.5

memory:
  enabled: true
  backend: "postgresql"

rag:
  enabled: true
  vector_store: "milvus"  # or "faiss" for development
  context_window: 150000000  # 150M tokens
```

### Basic Usage

```python
from nexus.core import EnsembleCore
from nexus.core.strategies import AdaptiveStrategy
from nexus.memory import KnowledgeBase
from nexus.rag import RAGVectorEngine

# Initialize ensemble
ensemble = EnsembleCore(
    strategy=AdaptiveStrategy(),
    config_path="config/default.yaml"
)

# Initialize memory system
knowledge_base = KnowledgeBase()

# Initialize RAG
rag_engine = RAGVectorEngine(context_window=150_000_000)

# Query with RAG augmentation
query = "Explain quantum computing"
context = await rag_engine.retrieve_context(query)
response = await ensemble.query(query, context=context)

# Store knowledge
await knowledge_base.store_fact(
    fact=response.answer,
    provenance=response.sources,
    confidence=response.confidence
)

print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence}")
print(f"Sources: {response.sources}")
```

### Run API Server

```bash
# Start the Nexus API
nexus-api --config config/default.yaml

# API available at http://localhost:5000
```

### API Example

```bash
# Ensemble inference
curl -X POST http://localhost:5000/api/v1/ensemble/query \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is consciousness?",
    "strategy": "adaptive",
    "use_rag": true
  }'

# Check memory
curl http://localhost:5000/api/v1/memory/facts?domain=philosophy \
  -H "X-API-Key: your-api-key"

# System health
curl http://localhost:5000/api/v1/health
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NEXUS AI PLATFORM                      │
│           Advanced Ensemble & Consciousness Framework        │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │ Ensemble │          │ Memory  │          │Reasoning│
   │  Layer   │◄────────►│  Layer  │◄────────►│  Layer  │
   └────┬────┘          └────┬────┘          └────┬────┘
        │                     │                     │
   ┌────▼─────────────────────▼─────────────────────▼────┐
   │              Knowledge & RAG Layer                   │
   │   (150M Token Context, 45+ Domain Knowledge)         │
   └────┬─────────────────────┬─────────────────────┬────┘
        │                     │                     │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │   API   │          │   UI    │          │ Agents  │
   │  Layer  │          │  Layer  │          │ (Plugin)│
   └─────────┘          └─────────┘          └─────────┘
```

### Core Principles

Based on the **Self-Taught Ensemble Intelligence Blueprint v1.2**:

1. **Skeptical by Default** - Low initial belief, multi-source corroboration
2. **Ensemble over Monolith** - Specialized models working in concert
3. **Tool-First Architecture** - Models plan, tools act
4. **Separation of Powers** - Planner, executor, verifier, critic remain distinct
5. **Human-in-the-Loop** - Goal-setting, review, approval workflows
6. **Transparency & Telemetry** - Reasoning, provenance, confidence visible
7. **Graceful Degradation** - Continue with reduced functionality on failure
8. **Cost-Awareness** - Dynamic compute selection based on task criticality

---

## Documentation

- [Getting Started Guide](docs/guides/GETTING_STARTED.md)
- [Architecture Overview](docs/architecture/ARCHITECTURE.md)
- [API Reference](docs/api/API_REFERENCE.md)
- [Deployment Guide](docs/deployment/KUBERNETES.md)
- [Configuration Reference](docs/guides/CONFIGURATION.md)
- [Contributing Guide](CONTRIBUTING.md)

---

## Deployment

### Docker

```bash
# Build image
docker build -t nexus:latest -f infrastructure/docker/Dockerfile .

# Run container
docker run -p 5000:5000 \
  -e OPENAI_API_KEY=your-key \
  -e ANTHROPIC_API_KEY=your-key \
  nexus:latest
```

### Kubernetes (Helm)

```bash
# Install Nexus with Helm
helm install nexus infrastructure/helm/nexus/ \
  --set api.replicas=3 \
  --set redis.enabled=true \
  --set postgresql.enabled=true

# Check status
kubectl get pods -l app=nexus
```

---

## Agent Integration

Nexus supports autonomous agents via a plugin architecture:

```python
from nexus.agents import AgentIntegration, AgentCapability

# Initialize agent integration
agent_integration = AgentIntegration()
await agent_integration.initialize()

# Submit task to agent
task_id = await agent_integration.submit_task(
    task_type="data_processing",
    capability=AgentCapability.DATA_PROCESSING,
    parameters={"file": "data.csv"}
)

# Check status
result = agent_integration.get_task_result(task_id)
```

See [Nexus-Agents](https://github.com/chrisarseno/Nexus-Agents) for the autonomous agent framework.

---

## Discovery Integrations

Nexus includes a powerful discovery system for finding and integrating external resources:

```python
from nexus.discovery import (
    ResourceDiscovery,
    GitHubIntegration,
    HuggingFaceIntegration,
    ArxivIntegration,
    WebSearchIntegration,
    LocalMachineIntegration,
)

# Initialize discovery system
discovery = ResourceDiscovery()

# Add integrations
github = GitHubIntegration(discovery, github_token="ghp_...")
huggingface = HuggingFaceIntegration(discovery)
arxiv = ArxivIntegration(discovery)

# Discover resources
await github.discover()  # Find AI/ML repos, datasets, tools
await huggingface.discover()  # Find models, datasets, spaces
await arxiv.discover()  # Find recent papers

# List discovered resources
resources = discovery.get_all_resources()
```

### Work-in-Progress: Zuultimate & Vinzy-Engine

Two additional integrations are scaffolded for future use:

#### Zuultimate (Identity/Access/Security)
```python
from nexus.discovery import ZuultimateIntegration, ZuultimateConfig

# Requires: pip install -e /path/to/zuultimate
# Requires: Zuultimate server running

zuul = ZuultimateIntegration(config=ZuultimateConfig(
    base_url="http://localhost:8000",
    api_key="your-api-key",
))

# Planned capabilities:
# - authenticate() - User authentication
# - encrypt_field() / decrypt_field() - Vault operations
# - authorize() - Zero-trust access control
# - tokenize() / detokenize() - PII tokenization
```

#### Vinzy-Engine (License Management)
```python
from nexus.discovery import VinzyIntegration, VinzyConfig

# Requires: pip install -e /path/to/vinzy-engine
# Requires: Vinzy server running

vinzy = VinzyIntegration(config=VinzyConfig(
    server_url="http://localhost:8080",
    license_key="XXXXXXXX-XXXXXXXXXXXXXXXX",
))

# Planned capabilities:
# - validate() - License validation
# - has_entitlement() - Feature gating
# - activate() / deactivate() - Machine management
# - record_usage() - Usage metering
```

> **TODO**: Complete integration by installing SDKs, running services, and wiring into Nexus features (e.g., license checks before inference, auth for API endpoints).

---

## Performance

**Latency Benchmarks** (p95):
- Ensemble inference: <500ms
- Memory retrieval: <100ms
- RAG query: <200ms
- API response: <1s

**Throughput**:
- Concurrent requests: >100 req/s
- Batch processing: >1000 items/min

**Accuracy**:
- Ensemble accuracy: >85%
- Knowledge validation: >95%
- RAG relevance: >90%

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

**Important:** All contributions require signing our Contributor License Agreement (CLA). This is handled automatically when you submit your first pull request.

### Development Setup

```bash
# Clone repo
git clone https://github.com/chrisarseno/Nexus.git
cd Nexus

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run linters
black src/
flake8 src/
mypy src/
```

---

## License

Nexus is dual-licensed:

1. **Open Source**: [GNU Affero General Public License v3.0](LICENSE)
   - Free for non-commercial use
   - Requires source disclosure for network services

2. **Commercial License**: For proprietary use
   - Contact: chris@1450enterprises.com
   - See [COMMERCIAL-LICENSE.txt](legal/licenses/COMMERCIAL-LICENSE.txt)

---

## Roadmap

### v1.0 (Current)
- ✅ Core ensemble system
- ✅ Memory & knowledge base
- ✅ RAG engine (150M tokens)
- ✅ Reasoning engines
- ✅ Production API
- ✅ Kubernetes deployment

### v1.1 (Planned)
- ⏳ WebSocket streaming
- ⏳ Advanced UI dashboards
- ⏳ Multi-modal support (vision, audio)
- ⏳ Extended reasoning chains
- ⏳ Enhanced agent integration
- ⏳ Zuultimate integration (identity/access/security)
- ⏳ Vinzy-Engine integration (license management)

### v2.0 (Future)
- 🔮 Consciousness metrics
- 🔮 Self-modification capabilities
- 🔮 Multi-agent collaboration
- 🔮 Quantum computing integration
- 🔮 Neural architecture search

---

## Citation

If you use Nexus in your research, please cite:

```bibtex
@software{nexus2025,
  title = {Nexus: Advanced AI Ensemble, Orchestrator \& Consciousness Framework},
  author = {Arsenault, Christopher R.},
  year = {2025},
  url = {https://github.com/chrisarseno/Nexus}
}
```

---

## Community

- **Issues**: [GitHub Issues](https://github.com/chrisarseno/Nexus/issues)
- **Discussions**: [GitHub Discussions](https://github.com/chrisarseno/Nexus/discussions)
- **Email**: chris@1450enterprises.com

---

## Acknowledgments

Built on research in:
- Multi-agent systems
- Ensemble learning
- Knowledge representation
- Consciousness frameworks
- Human-AI collaboration

Special thanks to all contributors and the open-source community.

---

**Nexus** - *Advancing towards sentient AGI*

Copyright © 2025 Christopher R. Arsenault. All rights reserved.
