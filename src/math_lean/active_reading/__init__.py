"""Active Reading prototype for ProofWiki.

Two-stage synthetic data generation following Lin et al. 2025
(arXiv:2508.09494): a model first proposes diverse study strategies
for a source document, then applies each strategy to generate a
self-training augmentation.
"""