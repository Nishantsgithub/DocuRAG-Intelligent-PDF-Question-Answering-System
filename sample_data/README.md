# Sample Data

Place PDF documents in this directory to test the DocuRAG pipeline.

**Disclaimer:** This project does not include any confidential, proprietary, or
client data. All documents used for development and testing are either:

- Publicly available academic papers (e.g., "Attention Is All You Need",
  open-access NLP papers)
- Synthetically generated documents created solely for testing purposes

No real user data, client information, or sensitive material is stored in this
repository.

## Suggested public PDFs for testing

| Document | Source |
|---|---|
| Attention Is All You Need | [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) |
| BERT: Pre-training of Deep Bidirectional Transformers | [arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) |
| Retrieval-Augmented Generation for NLP | [arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401) |

Download any of the above and drop the PDF here, then run:

```bash
# Via the API
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@sample_data/attention.pdf"

# Or via the Streamlit UI at http://localhost:8501
```
