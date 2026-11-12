# Retrieval-Augmented Generation Architectures Using AWS Bedrock and Databricks: A Practical Engineering Guide

## Runnable companions

- [Vector Search Demo](https://github.com/r-aaron-graham/vector-database-integration) ranks a synthetic vector fixture locally.
- [Adaptive RAG Efficiency Lab](https://github.com/r-aaron-graham/adaptive-rag-efficiency-lab) compares fixed and guarded lexical retrieval with exact citation labels.
- [Agent Trust Control Plane](https://github.com/r-aaron-graham/agent-trust-control-plane) demonstrates scoped retrieval, policy decisions, and human review.

The examples run locally and illustrate individual parts of the cloud architecture below.

## Abstract

Standalone large language models (LLMs) are powerful generative systems, but they remain unreliable for enterprise decision support because their outputs can contain hallucinations: confident statements not warranted by underlying evidence. Their internal (parametric) knowledge is also difficult to update, audit, or scope to proprietary data. Retrieval-Augmented Generation (RAG) addresses these limitations by coupling an LLM with an explicit retrieval system over curated knowledge sources, enabling responses to be grounded in retrieved passages with provenance and improved factuality.

This article presents a production-grade engineering architecture for RAG that combines AWS Bedrock for managed model inference, Databricks for large-scale data processing and governed storage, a vector store (Databricks Vector Search or FAISS), and a Python orchestration layer for retrieval, reranking, prompt construction, and output validation. I walk through a pragmatic reference design including diagrams and code, explain how to reduce hallucinations via grounding, filtering, reranking, and validation, and propose an evaluation and monitoring stack that measures both offline quality and live grounding rate, latency, token usage, cost, and drift.

---

## Introduction

Hallucination is not a corner case. It is a predictable failure mode that emerges from how LLMs are trained and optimized, especially when the model is prompted to respond to questions that exceed its verifiable knowledge or require fresh, proprietary, or context-specific facts. Benchmarks such as TruthfulQA were explicitly designed to elicit untruthful answers that "sound right," illustrating how models can imitate misconceptions in training data rather than reason from evidence. Even with continuing improvements, major model reports emphasize that hallucinations remain a key limitation, with factuality being evaluated explicitly in internal adversarial settings.

A central cause is the limitation of purely parametric knowledge: the model's "knowledge" is embedded in weights learned from historical corpora, making it difficult to (a) update quickly, (b) scope to a specific enterprise policy set, and (c) provide auditable provenance. Enterprises therefore need retrieval systems that can incorporate and constrain model outputs using authoritative sources such as policies, engineering docs, customer contracts, incident postmortems, and product manuals. They also need governance controls to ensure the right user sees the right content.

RAG was first formalized as a general-purpose architectural pattern combining a generator with a dense retriever over an external index, motivated precisely by these constraints: augment generation with non-parametric memory that can be refreshed and cited without retraining the core model. In practice, modern RAG has evolved from a single "retrieve-then-prepend" step into an end-to-end pipeline with ingestion, chunking, embedding, indexing, retrieval with filters, reranking, prompt construction, grounded generation, output validation, and continuous monitoring.

AWS Bedrock provides a managed inference layer over multiple model providers with security and operational controls, while Databricks provides a governed lakehouse environment for data ingestion, transformation, storage (Delta Lake), and managed vector search integrated with Unity Catalog. This combination supports the enterprise RAG goal: keep data governed, retrieval reproducible, and generation grounded.

---

## Background and Core Concepts

**Large language models.** LLMs are autoregressive models trained to predict the next token. They excel at fluent generation but are not inherently constrained to truth unless constrained by supervision, tools, or evidence checks. In operational deployments, a "chat" interface typically wraps the model with system instructions, tool calls, and retrieval context.

**Embeddings.** Embeddings map text (or multimodal inputs) into vectors such that semantic similarity corresponds to geometric proximity. These vectors enable fast similarity search over large corpora. In AWS Bedrock, Amazon Titan Text Embeddings models are a common managed option for generating embeddings, producing a vector that can be indexed for retrieval.

**Vector databases and ANN search.** Given millions to billions of vectors, exact nearest neighbor search is expensive. Real systems use approximate nearest neighbor (ANN) indexes such as IVF/PQ and HNSW to trade small recall loss for large performance gains. FAISS is a widely used similarity search library that implements high-performance ANN methods, including GPU-accelerated variants.

**Dense retrieval and semantic search.** Dense retrieval uses dual encoders to embed queries and passages into the same vector space and retrieve top-k nearest passages. Dense Passage Retrieval (DPR) demonstrated large gains over BM25 baselines for open-domain QA in top-k retrieval accuracy, illustrating why dense methods are foundational to modern RAG.

**Sparse retrieval (BM25) and hybrid search.** Sparse lexical models like BM25 remain valuable for exact-match features such as IDs, proper nouns, and citations, and serve as a robust baseline across heterogeneous domains. Modern hybrid search blends dense and sparse results. Databricks Vector Search, for example, uses BM25 for keyword relevance and combines it with vector similarity via Reciprocal Rank Fusion (RRF).

**Prompt grounding.** Grounding means constraining generation to retrieved evidence. In RAG, grounding is not just "adding context." It includes retrieval confidence thresholds, metadata filters, and output requirements such as citations and refusal when evidence is insufficient.

**Reranking.** Reranking improves precision by re-scoring a candidate set of retrieved chunks (top-k) with a stronger model such as a cross-encoder, LLM-based scorer, or vendor reranker, and selecting the best subset for the prompt. Databricks Vector Search explicitly supports a reranker step in queries and exposes debug timing for reranking latency.

### Conceptual RAG Pipeline

```
User Query
  -> Query Processing (rewrite, normalize, metadata intent)
  -> Embed Query
  -> Retrieve Top-k (vector / hybrid + filters)
  -> Rerank or compress contexts
  -> Prompt Construction (instructions + citations + schema)
  -> LLM Generation
  -> Output Validation (citations, safety, refusal checks)
  -> Answer + Attributions
  -> Observability (logs, metrics, traces)
```

This reflects the engineering reality that "retrieval" is only one stage. Reliability depends on pipeline-wide controls: filters, reranking, validation, and monitoring.

---

## End-to-End Architecture

This section describes a production-grade RAG architecture using (a) AWS Bedrock for LLM inference and embeddings, (b) Databricks for ingestion, transformation, governance, and vector search, and (c) a Python service for orchestration, prompt construction, reranking, and validation.

### Component Overview

**Object storage and raw corpora.** In AWS-centered deployments, raw documents typically land in Amazon S3 (PDFs, HTML, markdown, tickets, wiki exports), then are ingested into a lakehouse table for downstream processing. Databricks recommends incremental ingestion patterns such as Auto Loader for continuously arriving files.

**Databricks lakehouse storage (Delta Lake).** Delta Lake is an open-source table format that adds ACID transactions and scalable metadata handling via a transaction log over cloud object storage. It is commonly used to store normalized document chunks and their metadata, enabling reproducible indexing and auditable lineage.

**Embedding generation.** You can generate embeddings in two broad ways:
- Use Bedrock embedding models directly (e.g., Titan Text Embeddings) and write the resulting vectors into Delta tables.
- Let Databricks Vector Search compute embeddings as part of creating a Delta Sync Index, keeping indexes synced with the underlying Delta table as it updates.

**Vector index (Databricks Vector Search or FAISS).**
- Databricks Vector Search integrates with Unity Catalog governance and supports automatic syncing (Delta Sync Index) so you can treat the index as a managed serving artifact.
- FAISS is often used for local prototyping or on-prem deployments where a managed service is not desired.

**Runtime retrieval with filters, hybrid search, and reranking.** Databricks Vector Search supports approximate nearest neighbor search, hybrid search, metadata filters, and an explicit reranker with debug timing information.

**LLM inference (AWS Bedrock).** AWS Bedrock provides inference endpoints and APIs (InvokeModel / Converse) for foundation models, with optional guardrails and provisioning tiers for throughput guarantees.

**Python orchestration layer.** A typical orchestration layer (FastAPI, Flask, or a Databricks App) is responsible for query preprocessing, retrieval calls, reranking logic, prompt assembly, model invocation, output validation, and logging/tracing.

This design separates the data plane (indexing) from the serving plane (interactive queries), which is essential for scaling and for preventing ingestion spikes from degrading online latency.

---

## Data Pipeline and Implementation

### Ingestion and Normalization at Scale

Databricks Auto Loader provides an incremental ingestion mechanism (`cloudFiles`) for continuously arriving files in cloud storage. In production, documents are normalized into a canonical schema:

- `doc_id`: stable identifier (URI, repo path, internal ID)
- `source`: system of record (wiki, ticketing, handbook)
- `timestamp/version`: content time or change version
- `security_labels`: ACL tags or ABAC attributes
- `text`: extracted text
- `metadata`: structured fields (product, region, jurisdiction)

Unity Catalog provides central access control, auditing, and lineage across tables and downstream artifacts, which matters because a RAG system is only as trustworthy as its access boundaries.

### Chunking Strategies and Why Chunk Size Matters

Chunking determines the retrieval "unit." Overly small chunks risk losing context (definitions separated from constraints). Overly large chunks reduce retrieval precision and can inflate prompt tokens. A pragmatic heuristic is to chunk along semantic boundaries (headings, sections, paragraphs) and then enforce a size cap with limited overlap. Header-aware splitting for markdown is a common refinement.

**Python example (LangChain):**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)
```

AWS documentation for Titan embeddings explicitly recommends segmenting documents into logical segments such as paragraphs or sections for RAG use cases.

### Embedding Generation with AWS Bedrock

A minimal embedding call pattern:

```python
import json
import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "amazon.titan-embed-text-v2:0"

payload = {"inputText": "Customer refund policy for EU subscriptions."}
resp = client.invoke_model(
    modelId=model_id,
    contentType="application/json",
    accept="application/json",
    body=json.dumps(payload).encode("utf-8"),
)

body = json.loads(resp["body"].read())
embedding = body["embedding"]
```

Model IDs vary by region and provider and should be discovered programmatically via `ListFoundationModels`.

### Vector Indexing with Databricks Vector Search

Databricks Vector Search builds a vector index from a Delta table and supports automatic sync as the underlying table updates (Delta Sync Index), keeping the index aligned with the governed data source.

**Minimal query example:**
```python
from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()
index = client.get_index(index_name="catalog.schema.my_index")

results = index.similarity_search(
    query_text="How do we handle EU subscription refunds?",
    columns=["doc_id", "chunk_id", "text", "source", "updated_at"],
    num_results=8,
    query_type="hybrid",       # blend lexical + vector
    filters="jurisdiction = 'EU'",  # SQL-like filter
)
```

Databricks documents hybrid search as combining similarity search with keyword search (BM25) via Reciprocal Rank Fusion, balancing semantic matching with exact lexical constraints.

### Reranking in Retrieval

Reranking is justified when the correct chunk is "in top-k" but not in the top few items sent to the model.

```python
from databricks.vector_search.reranker import DatabricksReranker

results = index.similarity_search(
    query_text="How to create a Vector Search index",
    columns=["id", "text", "parent_doc_summary", "date"],
    num_results=10,
    query_type="hybrid",
    reranker=DatabricksReranker(columns_to_rerank=["text", "parent_doc_summary"]),
)
```

LangChain and LlamaIndex both include reranking integrations and retriever wrappers for framework-level abstractions.

### Prompt Construction for Grounded Answers

The prompt must make grounding enforceable, not aspirational. A robust template typically includes:

- **System instruction:** answer only from provided context; if insufficient, say so.
- **Citation requirement:** each claim must reference a chunk ID.
- **Output schema:** a structured answer (e.g., JSON with `answer`, `citations`, `confidence`, `followups`).
- **Safety policy alignment (optional):** guardrails or post-filters.

### Bedrock Model Integration for Generation

AWS Bedrock supports InvokeModel and the Converse interface, which works with all models that support messages, allowing code reuse across models.

```python
import json
import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

system = [{"text": "You are a policy assistant. Answer only using the provided CONTEXT. Cite chunk_ids."}]
messages = [
    {"role": "user", "content": [{"text": "QUESTION: ...\n\nCONTEXT:\n[chunk_12] ...\n[chunk_41] ..."}]}
]

resp = client.converse(
    modelId=model_id,
    system=system,
    messages=messages,
    inferenceConfig={"maxTokens": 800, "temperature": 0.2},
)

answer_text = resp["output"]["message"]["content"][0]["text"]
usage = resp.get("usage", {})
```

Bedrock supports Provisioned Throughput to increase and stabilize model invocation capacity. Streaming APIs (`InvokeModelWithResponseStream` / `ConverseStream`) reduce perceived latency by returning partial outputs.

---

## Evaluation and Monitoring

A RAG system is simultaneously an ML system, a search system, and a distributed system. The quality posture therefore requires both offline evaluation (before deployment) and online monitoring (after deployment).

### Quantitative Evaluation: What to Measure and Why

**Retrieval quality metrics:**
- Recall@k / Hit rate@k: does top-k contain a gold passage?
- MRR: does ranking put the best passage early?
- Null-retrieval rate: how often does retrieval yield no relevant context?

**Generation quality metrics:**
- Answer correctness (exact match / rubric score)
- Faithfulness/grounding: are claims supported by retrieved evidence?
- Citation precision/recall: do citations correspond to supporting passages?

**End-to-end operational metrics:**
- p50/p95 latency by stage (retrieve, rerank, generate)
- Token usage and cost per request
- Error/timeout rates, throttling, and model saturation behavior

### Example Results from the Literature

The Self-RAG paper provides a useful quantitative comparison across tasks, including citation precision/recall on ASQA and factuality (FactScore) on biography generation. A condensed view of selected results:

| System | PopQA Accuracy | PubHealth Accuracy | Bio FactScore | ASQA Citation Precision | ASQA Citation Recall |
|---|---|---|---|---|---|
| Llama2 7B (no retrieval) | 14.7 | 34.2 | 7.9 | -- | -- |
| Alpaca 13B (no retrieval) | 24.4 | 55.5 | 22.9 | -- | -- |
| Llama2 7B (with retrieval) | 38.2 | 30.0 | 15.2 | 2.9 | 4.0 |
| Llama2-FT 7B (with retrieval) | 48.7 | 64.3 | 31.0 | 5.0 | 7.5 |
| Self-RAG 13B | 55.8 | 74.5 | 31.7 | 70.3 | 71.3 |

Two operational lessons stand out: (1) retrieval alone does not guarantee citation accuracy, as naive baselines can have very low citation precision/recall; and (2) enforcing self-critique/selection mechanisms improves both task accuracy and attribution quality.

### Monitoring in Production: Metrics That Catch Failures Early

A monitoring stack should separate (a) system health, (b) retrieval health, and (c) grounding quality.

**System health metrics:**
- p50/p95 latency for API, retrieval, rerank, Bedrock inference
- Timeout rate, throttling rate, retries
- Queue depth (if async ingestion or batch updates)

**Cost and capacity metrics:**
- Per-request input tokens, output tokens, total tokens
- Cache read/write token metrics (if prompt caching enabled)
- Cost per request and cost per tenant

**Retrieval and grounding metrics:**
- Null-retrieval rate
- Retrieval score distribution drift (embedding drift proxy)
- Citation coverage (fraction of answers with citations)
- Citation precision audits (sampled human review or automated checks)

Reference-free RAG evaluation frameworks such as RAGAS are designed to score faithfulness and relevance without requiring ground-truth labels for every query, supporting faster iteration cycles.

### Practical Hallucination Reduction Controls (What Actually Works)

Hallucination mitigation is most effective when implemented as layered controls:

1. **Retrieval correctness controls:** metadata filters (jurisdiction, product, access level), hybrid search for exact term matching, and reranking when top-k contains relevant passages but ordering is wrong.
2. **Generation constraints:** low temperature, explicit refusal on insufficient evidence, and schema-constrained output where feasible.
3. **Post-generation validation:** citation presence and "claim-evidence alignment" checks (automated + sampling). Frameworks like SelfCheckGPT motivate consistency-based detection approaches when ground truth is absent.

---

## Security, Governance, and Production Deployment

Enterprise RAG must satisfy two simultaneous constraints: (a) protect data (PII, customer-sensitive content, regulated artifacts), and (b) keep the system operationally reliable under real traffic.

### Security and Data Governance

**AWS security controls.** Amazon Bedrock can be accessed through interface VPC endpoints via AWS PrivateLink, enabling private connectivity without traversing the public internet. This is commonly paired with customer-managed VPC deployment patterns on Databricks for tighter network control.

Bedrock's data protection documentation states that Bedrock does not store or log prompts and completions and does not use them to train AWS models, which is relevant for vendor risk assessments and compliance posture.

**Access control and auditing (Databricks).** Unity Catalog provides centralized access control, auditing, lineage, and governance across data and AI assets. Databricks audit logs provide fine-grained tracking of workspace and data events, which can be integrated into enterprise SIEM workflows.

**Prompt and safety governance (Bedrock Guardrails).** Amazon Bedrock Guardrails provide content filters, denied topics, sensitive information filters, and other safeguard policies that can be applied to model inference, agents, or knowledge base queries. In a RAG context, guardrails are a complement to grounding: they reduce unsafe or disallowed outputs even if retrieved content contains risky instructions or if the user attempts prompt injection.

### Production Deployment Strategies

You have multiple deployment topologies depending on where you want the application layer to live:

- **Databricks-native app:** build and deploy a Databricks App that queries Vector Search and calls Bedrock, keeping governance and user authentication integrated with the platform.
- **External Python service (Kubernetes/ECS):** run FastAPI as a stateless service, calling Vector Search via REST/SDK and Bedrock via boto3. This is common when you need custom auth, custom routing, or multi-region serving. The RAG mechanics remain unchanged; only deployment changes.
- **Databricks Model Serving gateway:** Databricks supports "external models" in Mosaic AI Model Serving, enabling centralized management of third-party model providers, with documented support for Amazon Bedrock as an external model provider.

### Scaling and What Breaks First

In RAG systems, the first bottleneck is usually one of:
- LLM inference saturation / throttling (Bedrock quotas and throughput constraints)
- Reranking latency (rerankers can dominate retrieval time)
- Vector store QPS or filter overfetch behavior (filters applied after overfetch can lead to empty returns despite matching items)

**Mitigations:**
- Use Provisioned Throughput for predictable Bedrock capacity in production.
- Use Databricks Vector Search high QPS configuration for sustained throughput on standard endpoints.
- Apply graceful degradation: skip reranking under load, shrink context for hot paths, prefer hybrid retrieval for exact-match queries, and enforce timeouts.
- Move ingestion/embedding to async pipelines via Lakeflow Jobs so that data-plane spikes do not impact serving-plane latency.
- Use caching where supported: Bedrock documents token cache metrics, indicating explicit support for prompt caching in the runtime monitoring model.

---

## Limitations and Future Directions

### Limitations of RAG

**Context window limits and prompt budget.** Even with retrieval, generation is constrained by the model's context window and by token cost. The engineering implication is that retrieval must be selective: you cannot "dump the whole handbook" into a prompt and expect consistent grounding.

**Retrieval failures.** RAG fails if it retrieves irrelevant or stale content. Dense retrieval improves semantic matching but is still sensitive to embedding drift, domain shift, and noisy chunking choices. Robust baselines (BM25/hybrid) and evaluation on heterogeneous benchmarks such as BEIR are standard mitigation strategies.

**Stale data and index lag.** If the index is not kept in sync with source-of-truth updates, the system can confidently produce obsolete guidance. Databricks Vector Search's Delta Sync Index design explicitly targets this by syncing an index as the Delta table updates, but the upstream ingestion pipeline still needs correctness guarantees.

**Overreliance on retrieval as a new attack surface.** Prompt injection and data poisoning can enter through retrieved content (e.g., malicious text in documents). Guardrails and post-validation help, but governance and document trust scoring are required for high-stakes use.

### Future Research Directions

**Agentic retrieval and selective grounding.** Research trends such as Self-RAG show that "retrieve every time" is not optimal. Models can learn to decide when retrieval is needed and to critique the supportiveness and relevance of evidence.

**Multimodal RAG.** As enterprises store diagrams, images, and scanned PDFs, multimodal embeddings and multimodal retrieval become critical. Bedrock includes Titan multimodal embeddings documentation, and Databricks has published platform guidance for multimodal pipelines.

**Knowledge graphs and structured retrieval.** Hybrid architectures that combine vector retrieval with entity-centric knowledge graphs can improve determinism, enable stronger provenance, and handle structured queries involving IDs and relationships.

**Self-refining pipelines.** Evaluation-driven retrieval refinement, including automatic hard-negative mining from production queries, embedding model refreshes, and continuous "grounding regressions" using reference-free metrics, are emerging as a practical MLOps discipline specific to RAG.

---

## Architecture Trade-offs at a Glance

| Architecture | Strengths | Weaknesses |
|---|---|---|
| Bedrock + Databricks | Enterprise governance (Unity Catalog), scalable pipelines (Spark/Delta), managed inference (Bedrock), managed vector search and reranking | Higher platform complexity and potentially higher cost; requires careful latency engineering with rerankers and token budgets |
| OpenAI + external vector DB | Often simplest to start; fewer moving parts if you already have a vector DB | Governance and networking depend on the vector DB + app stack; multi-tenant isolation and auditability can be harder without an integrated catalog |
| Self-hosted LLM + FAISS | Maximum control; can run in restricted environments | Infrastructure complexity (serving, scaling, security patches), performance tuning burden; requires strong internal ops maturity |

---

## Glossary

**ANN (Approximate Nearest Neighbor):** Fast similarity search that trades small recall loss for large speed gains.

**Chunk:** A retrieval unit (span of text) produced during ingestion; must be sized to preserve meaning while enabling precise retrieval.

**Dense retrieval:** Retrieval using vector similarity of embeddings (e.g., DPR-style dual encoders).

**Grounding rate:** Operational metric measuring whether answers are supported by retrieved context (often proxied by citation coverage and sampled audits).

**Hybrid retrieval:** Combined lexical + vector retrieval; may use fusion methods such as RRF.

**Reranker:** A stronger scoring model applied to top-k retrieved candidates to improve precision.

**Unity Catalog:** Databricks governance layer for access control, auditing, and lineage across data and AI assets.

---

*Aaron Graham | VP of Stack & Integration, CloudNexxus Group | Lead AI Engineer, AMIE K-12 AI Project*
