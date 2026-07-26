# ZI 01 — Long-document table reconstruction

Reconstructs Table 1 for one explicitly named HF dataset at a time. The PDF
does not disclose its dataset revisions, splits, tokenizer, or optimization
settings; the launcher therefore records all supplied values and is a protocol
reconstruction, not an exact reproduction. Default: `SetFit/20_newsgroups`.

Example: `bash rebuttal/01_zi_long_documents/run.sh --dataset-id SetFit/20_newsgroups`.
