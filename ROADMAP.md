# BharatPolicy AI - Product Roadmap

This document outlines the vision for post-prototype development and enhancement.

## Prototype Scope (Current - Submission Ready)

**✓ Core Features Implemented:**
- RAG-based budget Q&A with multi-source citations
- Rule-based scheme recommendations with profile matching
- Personalized budget impact summaries
- Windows batch script runner setup
- Fallback mode (works offline, no LLM, no heavy dependencies)

**✓ Data & Infrastructure:**
- 235 indexed chunks from 2 budget PDFs
- 20 government schemes with full metadata
- Local hash embeddings (no model download)
- Optional FAISS acceleration

---

## Phase 1: Production Readiness (Month 1)

### Authentication & Multi-User Support
- [ ] User account system (email/password or SSO)
- [ ] JWT-based API authentication
- [ ] Per-user history and saved recommendations
- **Impact**: Enable real-world deployment, track usage

### Enhanced Language Support
- [ ] Full Hindi response generation (not prompt-only)
- [ ] Regional language templates (Marathi, Tamil, Gujarati)
- [ ] Script auto-detection and normalization
- **Impact**: Serve non-English speaking citizens

### Deployment Infrastructure
- [ ] Docker containerization
- [ ] Kubernetes deployment config
- [ ] AWS/GCP/Azure deployment guides
- [ ] CI/CD pipeline (GitHub Actions)
- **Impact**: Easy replicas, auto-scaling, monitoring

---

## Phase 2: Data & Quality (Month 2)

### Dynamic Index Management
- [ ] Admin UI for PDF upload/delete/versioning
- [ ] Automatic incremental indexing
- [ ] Index health checks and metrics
- **Impact**: Non-technical users can manage PDFs

### Improved Recommendations
- [ ] Feedback loop: "was this scheme helpful?"
- [ ] Collaborative filtering (similar users' choices)
- [ ] Contextual keyword boost (e.g., "urgent" → urgent schemes first)
- **Impact**: Better relevance over time

### Data Quality
- [ ] Fact-checking layer (verify claims against source)
- [ ] Source attribution with document links
- [ ] Citation confidence scoring
- **Impact**: Compliance with accuracy guarantees

---

## Phase 3: Analytics & Insights (Month 3)

### User Analytics
- [ ] Track popular questions and schemes
- [ ] Demographic insights (which profiles→ which schemes)
- [ ] Regional resource allocation insights
- **Impact**: Feedback to policy makers

### Metrics & Monitoring
- [ ] Latency dashboards per endpoint
- [ ] Error rate and fallback frequency tracking
- [ ] Model drift detection
- **Impact**: Proactive system health

### Feedback System
- [ ] Post-summary survey ("Did this help?")
- [ ] Scheme outcome tracking (did user apply?)
- [ ] Long-term follow-up (which scheme benefited you?)
- **Impact**: Measure real-world impact

---

## Phase 4: Advanced Features (Month 4+)

### Enterprise Features
- [ ] White-label deployment
- [ ] Custom scheme databases (state/org-specific)
- [ ] Webhook integrations (Slack, email, SMS)
- **Impact**: B2B partnerships

### AI Enhancements
- [ ] Fine-tuned local LLM (privacy + cost)
- [ ] Question clarification (follow-ups for ambiguous queries)
- [ ] Scheme pre-fetching (query intent detection)
- **Impact**: Faster, more accurate responses

### Mobile & Accessibility
- [ ] Mobile-responsive Streamlit → React frontend
- [ ] Voice input/output support
- [ ] Screen reader optimization
- [ ] Offline mode with local cache
- **Impact**: Broader audience reach

---

## Technical Debt & Quality

### Code Quality
- [ ] Unit test coverage → 80%+
- [ ] Integration test suite
- [ ] Linting & type checking (mypy, pylint)
- [ ] Performance benchmarks

### Documentation
- [ ] API OpenAPI/Swagger spec
- [ ] Architecture decision records (ADRs)
- [ ] Troubleshooting guides per module
- [ ] Video tutorials

### Security
- [ ] OWASP top 10 audit
- [ ] Penetration testing
- [ ] Data encryption (at rest, in transit)
- [ ] GDPR/privacy compliance

---

## Known Limitations (Proto → Post-Proto)

| Limitation | Phase | Mitigation |
|------------|-------|-----------|
| No user authentication | 1 | JWT + OAuth2 |
| Hindi is prompt-guided only | 2 | Full translation service |
| Single PDF index (static) | 2 | Dynamic admin UI |
| No feedback loop | 3 | Survey + tracking |
| Local embeddings only | 2 | Optional fine-tuned model |
| Windows batch scripts only | 1 | Docker + K8s |
| No mobile | 4 | React frontend |

---

## Success Metrics (Post-Prototype)

- **Adoption**: 50k+ active users in 6 months
- **Accuracy**: 90%+ scheme match relevance (user feedback)
- **Performance**: <500ms p99 latency for all endpoints
- **Reliability**: 99.9%+ uptime
- **Impact**: Track 10k+ scheme applications completed via platform

---

## Stakeholder Roadmap Alignment

**Government**: Policy feedback, citizen insights, compliance audit trail  
**Citizens**: Accessible Q&A, scheme discovery, application support  
**NGOs**: Data export, offline deployment, regional customization  
**Developers**: Open API, self-hosted option, plugin architecture

---

**Roadmap Review Cadence**: Monthly (adjust based on usage patterns)  
**Feature Request Channel**: GitHub Issues + User Feedback Form  
**Last Updated**: March 28, 2026
