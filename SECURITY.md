# Security and data boundaries

SmartHealth Agent is a local, single-process engineering implementation using synthetic physiological inputs. It is not a clinically validated diagnostic system. The current bearer token represents one development identity; it is not a multi-tenant authentication service.

## Reporting a vulnerability

Use this repository's private vulnerability-reporting entry if the owner has enabled it. Otherwise, open an issue requesting a private contact channel without including exploit details, tokens, physiological data or other personal information. A response-time commitment has not been established.

Useful reports describe the affected code/version, expected and actual behavior, and a minimal reproduction using synthetic data. Do not test against anyone else's deployment or upload real health records as a reproduction.

## Operator responsibilities

- Keep API/provider credentials in local process configuration or a managed secret store. `.env.example` contains placeholders only; the application does not automatically load `.env`.
- Use trusted model bundles, harness configuration and knowledge assets. Hash checks detect unexpected changes; they do not make an untrusted joblib/pickle file safe to load.
- Bind development services to loopback. Add trusted identity, TLS and deployment isolation before supporting remote or mutually untrusted users.
- Treat summaries, vectors and audit identifiers as potentially sensitive even when raw CSV data is excluded.
- Respect separate retention and deletion boundaries for Memory, artifacts, traces and backups. Deleting Memory does not erase all other stores.
- Follow the single-worker/local-Qdrant constraints in the [deployment guide](docs/deployment.md).

Known quality limitations, including a semantic-review failure in the scripted evaluation, are documented in the [evaluation report](evaluation/evaluation_report.md). Engineering checks and role separation do not guarantee that all report claims are supported by evidence.
