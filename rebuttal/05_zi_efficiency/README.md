# ZI 05 — Original efficiency grid

Runs the PDF's length grid `256,512,1024,2048,4096` with the existing isolated
FP32 attention benchmark. It intentionally retains our adapters and reports
parameter mismatches; it cannot reproduce unavailable GMM-RKS, Mamba, MetaLA,
or FlashAttention implementations from the PDF.
