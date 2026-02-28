# Commercial Licensing — nexus

This project is dual-licensed:

- **AGPL-3.0** — Free for open-source use with copyleft obligations
- **Commercial License** — Proprietary use without AGPL requirements

## Tiers

| Feature | Community (Free) | Pro ($149/mo) | Enterprise ($499/mo) |
|---------|:---:|:---:|:---:|
| Base reasoning & memory | Yes | Yes | Yes |
| RAG pipeline | Yes | Yes | Yes |
| Multi-model ensemble | — | Yes | Yes |
| Advanced reasoning chains | — | Yes | Yes |
| Discovery & intelligence | — | — | Yes |
| Strategic analysis | — | — | Yes |
| Support SLA | Community | 48h email | 4h priority |

## How It Works

- **No license key** — All code runs (AGPL mode). Source is visible per AGPL obligations.
- **License key set** — Only entitled features are unlocked. Blocked features show a clear error with upgrade instructions.
- **Server unreachable** — Fail-closed for gated features.

## Getting a License

Visit **https://1450enterprises.com/pricing** or contact sales@1450enterprises.com.

```bash
export VINZY_LICENSE_KEY="your-key-here"
export VINZY_SERVER="https://api.1450enterprises.com"
```

## Feature Flags

Flags follow the convention `nxs.{module}.{capability}`:

| Flag | Tier | Description |
|------|------|-------------|
| `nxs.reasoning.advanced` | Pro | Advanced reasoning chains |
| `nxs.ensemble.multi_model` | Pro | Multi-model ensemble |
| `nxs.discovery.intelligence` | Enterprise | Discovery & intelligence |
| `nxs.strategic.analysis` | Enterprise | Strategic analysis |
