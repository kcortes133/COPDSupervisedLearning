"""
Gene-set enrichment analysis stub.

This module is a placeholder for hooking the pipeline's transcriptomics
output into a gene-set enrichment analysis (GSEA) via `gseapy`. The
intended workflow is:

    1. Pull the top-ranked transcriptomic features from the trained
       voting classifier (`getTopOmics`).
    2. Map those feature indices back to gene IDs using the
       `Transcriptomics/2021-09-28 original/counts_raw.tsv` table that
       `main.py` already loads as `geneIDs`.
    3. Run `gseapy.prerank` (or `gseapy.enrichr`) against a curated
       pathway database (Hallmark, KEGG, GO).

Currently `getTopOmics` is unimplemented — the function returns nothing.
Left in the repo as a hook for future work.
"""

import gseapy


def getTopOmics(features, topFeatures):
    """Stub: return the top-N features for downstream GSEA.

    Not yet implemented. See module docstring for the intended workflow.

    Parameters
    ----------
    features : array-like
        Per-feature importance scores from the trained classifier.
    topFeatures : int
        Number of top features to return.
    """
    return
