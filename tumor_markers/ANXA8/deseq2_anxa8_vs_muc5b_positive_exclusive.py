#!/usr/bin/env python3
"""Paired pseudobulk DESeq2: ANXA8-only versus MUC5B-only PDAC tumor cells."""

import argparse
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


# All PDAC samples with tumor cells after the updated annotation.
# Excludes SP074/SP076/SP089 (zero tumor).
TARGET_SAMPLES = [
    "C3N-02295-14Fp1U1",
    "C3L-03395-13Fp1U1",
    "HT227P1-S1H1L1U2",
    "HT1046P1-S1H1Fp1U1",
    "HT910P1-S1H1Fp1U1",
    "HT700P1-S1H1Fp1U1",
    "HT535P1-S1H1Fp1U1",
    "HT943P1-S1H2Fp1U1",
    "HT873P1-S1H1Fp1U2",
    "C3L-03394-13Fp1U1",
    "HT608P1-S1H2Fp1U1",
    "C3N-02589-11Fp1U1",
    "HT124P1-S1H3A1Fp1U1",
    "SP041P1-A20-S1Fp1U3_2",
    "C3L-02116-11Fp1U1",
    "SP090P1-A3-S1Fp1U1",
]

DEFAULT_H5AD = (
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/PDAC/annotated.Yuwei.updated.h5ad"
)
DEFAULT_OUTPUT = "ANXA8_vs_MUC5B_DESeq2_positive_exclusive"
DEFAULT_RSCRIPT = (
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/conda/envs/scenv/bin/Rscript"
)

R_CODE = r"""
suppressPackageStartupMessages(library(DESeq2))

args <- commandArgs(trailingOnly = TRUE)
count_file <- args[[1]]
metadata_file <- args[[2]]
output_dir <- args[[3]]

counts <- read.delim(
  count_file, row.names = 1, check.names = FALSE, quote = ""
)
metadata <- read.delim(
  metadata_file, row.names = 1, check.names = FALSE, stringsAsFactors = FALSE
)

stopifnot(identical(colnames(counts), rownames(metadata)))
metadata$sample_id <- factor(metadata$sample_id)
metadata$group <- factor(
  metadata$group,
  levels = c("MUC5B_only", "ANXA8_only")
)

dds <- DESeqDataSetFromMatrix(
  countData = round(as.matrix(counts)),
  colData = metadata,
  design = ~ sample_id + group
)

# Remove genes with too little information before model fitting.
keep <- rowSums(counts(dds)) >= 10 & rowSums(counts(dds) > 0) >= 2
dds <- dds[keep, ]
if (nrow(dds) == 0) {
  stop("No genes passed the minimum count filter.")
}

dds <- DESeq(dds)
res <- results(
  dds,
  contrast = c("group", "ANXA8_only", "MUC5B_only"),
  alpha = 0.05
)
res_df <- as.data.frame(res)
res_df$gene <- rownames(res_df)
res_df <- res_df[
  order(is.na(res_df$padj), res_df$padj, res_df$pvalue),
  c("gene", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj")
]

write.csv(
  res_df,
  file.path(output_dir, "DESeq2_ANXA8_only_vs_MUC5B_only.csv"),
  row.names = FALSE,
  quote = FALSE
)

normalized <- as.data.frame(counts(dds, normalized = TRUE))
normalized$gene <- rownames(normalized)
normalized <- normalized[, c("gene", setdiff(colnames(normalized), "gene"))]
write.table(
  normalized,
  file.path(output_dir, "DESeq2_normalized_pseudobulk_counts.tsv"),
  sep = "\t", row.names = FALSE, quote = FALSE
)

saveRDS(dds, file.path(output_dir, "DESeq2_dds.rds"))

pdf(
  file.path(output_dir, "DESeq2_MA_plot.pdf"),
  width = 6, height = 5.5, useDingbats = FALSE
)
plotMA(res, alpha = 0.05, ylim = c(-5, 5))
dev.off()

volcano <- res_df[
  is.finite(res_df$log2FoldChange) &
    !res_df$gene %in% c("ANXA8", "MUC5B"),
]
volcano$minus_log10_padj <- -log10(pmax(volcano$padj, 1e-300))
volcano$status <- "Not significant"
volcano$status[
  !is.na(volcano$padj) &
    volcano$padj < 0.05 &
    volcano$log2FoldChange > 1
] <- "Higher in ANXA8_only"
volcano$status[
  !is.na(volcano$padj) &
    volcano$padj < 0.05 &
    volcano$log2FoldChange < -1
] <- "Higher in MUC5B_only"
n_up <- sum(volcano$status == "Higher in ANXA8_only")
n_down <- sum(volcano$status == "Higher in MUC5B_only")

pdf(
  file.path(output_dir, "DESeq2_volcano_plot.pdf"),
  width = 4.8, height = 5.5, useDingbats = FALSE
)
cols <- c(
  "Not significant" = "grey70",
  "Higher in ANXA8_only" = "#D62728",
  "Higher in MUC5B_only" = "#1F77B4"
)
plot(
  volcano$log2FoldChange,
  volcano$minus_log10_padj,
  pch = 16,
  cex = 0.55,
  col = unname(cols[volcano$status]),
  xlab = "log2 fold change (ANXA8_only / MUC5B_only)",
  ylab = "-log10 adjusted p-value",
  main = "Paired pseudobulk DESeq2"
)
abline(v = c(-1, 1), h = -log10(0.05), lty = 2, col = "grey40")
csf2 <- volcano[volcano$gene == "CSF2", ]
if (nrow(csf2) == 1) {
  points(
    csf2$log2FoldChange,
    csf2$minus_log10_padj,
    pch = 21,
    cex = 1.4,
    lwd = 1.2,
    col = "black",
    bg = "#FFD700"
  )
  text(
    csf2$log2FoldChange,
    csf2$minus_log10_padj,
    labels = "CSF2",
    pos = 4,
    offset = 0.5,
    font = 2
  )
}
legend(
  "topright",
  c(
    "Not significant",
    paste0("Up in ANXA8_only (n=", n_up, ")"),
    paste0("Down in ANXA8_only (n=", n_down, ")"),
    "CSF2"
  ),
  col = c(cols, "black"),
  pt.bg = c(NA, NA, NA, "#FFD700"),
  pch = c(16, 16, 16, 21),
  bty = "n",
  cex = 0.8
)
dev.off()

cat(
  "DESeq2 complete:",
  nrow(res_df), "tested genes;",
  sum(res_df$padj < 0.05, na.rm = TRUE), "genes with FDR < 0.05\n"
)
"""


def decode_strings(values):
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O"}:
        return np.array(
            [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in values
            ],
            dtype=object,
        )
    return values.astype(str)


def read_h5ad_vector(parent, key):
    node = parent[key]
    if isinstance(node, h5py.Dataset):
        return decode_strings(node[:])

    encoding = node.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")
    if encoding != "categorical":
        raise ValueError(f"Unsupported H5AD encoding for {key}: {encoding}")

    categories = decode_strings(node["categories"][:])
    codes = node["codes"][:]
    result = np.full(codes.shape, "", dtype=object)
    valid = codes >= 0
    result[valid] = categories[codes[valid]]
    return result


def aggregate_pseudobulk(h5ad_path, output_dir, chunk_size, target_samples):
    with h5py.File(h5ad_path, "r") as handle:
        sample_ids = read_h5ad_vector(handle["obs"], "dataset")
        cell_types = read_h5ad_vector(handle["obs"], "cell_type")
        genes = read_h5ad_vector(handle["var"], "_index")

        if len(np.unique(genes)) != len(genes):
            raise ValueError("Gene names in adata.var_names are not unique.")
        gene_lookup = {gene: index for index, gene in enumerate(genes)}
        missing = {"ANXA8", "MUC5B"} - set(gene_lookup)
        if missing:
            raise ValueError(f"Genes absent from H5AD: {sorted(missing)}")

        counts_group = handle["layers"]["counts"]
        if counts_group.attrs.get("encoding-type") not in {
            "csr_matrix",
            b"csr_matrix",
        }:
            raise ValueError("The counts layer must use CSR matrix encoding.")

        shape = tuple(int(value) for value in counts_group.attrs["shape"])
        if shape != (len(sample_ids), len(genes)):
            raise ValueError("Counts matrix dimensions do not match obs/var.")

        target_set = set(target_samples)
        is_target = np.array([sample in target_set for sample in sample_ids])
        is_tumor = cell_types.astype(str) == "Tumor"
        eligible = is_target & is_tumor

        accumulators = {
            (sample, group): np.zeros(len(genes), dtype=np.int64)
            for sample in target_samples
            for group in ("MUC5B_only", "ANXA8_only")
        }
        summary = {
            sample: {
                "n_tumor_cells": int(np.sum(eligible & (sample_ids == sample))),
                "n_ANXA8_positive": 0,
                "n_MUC5B_positive": 0,
                "n_both_positive_excluded": 0,
                "n_ANXA8_only": 0,
                "n_MUC5B_only": 0,
            }
            for sample in target_samples
        }

        indptr = counts_group["indptr"][:]
        data_source = counts_group["data"]
        index_source = counts_group["indices"]
        anxa8_index = gene_lookup["ANXA8"]
        muc5b_index = gene_lookup["MUC5B"]

        print(
            f"Found {eligible.sum():,} tumor cells in the "
            f"{len(target_samples)} target samples."
        )

        for start in range(0, shape[0], chunk_size):
            stop = min(start + chunk_size, shape[0])
            selected_local = np.flatnonzero(eligible[start:stop])
            if selected_local.size == 0:
                continue

            data_start = int(indptr[start])
            data_stop = int(indptr[stop])
            chunk = sparse.csr_matrix(
                (
                    data_source[data_start:data_stop],
                    index_source[data_start:data_stop],
                    indptr[start : stop + 1] - data_start,
                ),
                shape=(stop - start, shape[1]),
            )
            selected_counts = chunk[selected_local]
            selected_samples = sample_ids[start:stop][selected_local]
            anxa8_positive = (
                selected_counts[:, anxa8_index].toarray().ravel() > 0
            )
            muc5b_positive = (
                selected_counts[:, muc5b_index].toarray().ravel() > 0
            )
            anxa8_only = anxa8_positive & ~muc5b_positive
            muc5b_only = muc5b_positive & ~anxa8_positive
            both_positive = anxa8_positive & muc5b_positive

            for sample in target_samples:
                sample_mask = selected_samples == sample
                if not np.any(sample_mask):
                    continue
                stats = summary[sample]
                stats["n_ANXA8_positive"] += int(
                    np.sum(sample_mask & anxa8_positive)
                )
                stats["n_MUC5B_positive"] += int(
                    np.sum(sample_mask & muc5b_positive)
                )
                stats["n_both_positive_excluded"] += int(
                    np.sum(sample_mask & both_positive)
                )
                stats["n_ANXA8_only"] += int(
                    np.sum(sample_mask & anxa8_only)
                )
                stats["n_MUC5B_only"] += int(
                    np.sum(sample_mask & muc5b_only)
                )

                for group, group_mask in (
                    ("ANXA8_only", anxa8_only),
                    ("MUC5B_only", muc5b_only),
                ):
                    mask = sample_mask & group_mask
                    if np.any(mask):
                        accumulators[(sample, group)] += np.asarray(
                            selected_counts[mask].sum(axis=0)
                        ).ravel().astype(np.int64)

            print(f"Processed rows {start:,}-{stop:,}")

    summary_df = pd.DataFrame.from_dict(summary, orient="index")
    summary_df.index.name = "sample_id"
    summary_df.to_csv(output_dir / "cell_group_summary.tsv", sep="\t")

    paired_samples = summary_df.index[
        (summary_df["n_ANXA8_only"] > 0)
        & (summary_df["n_MUC5B_only"] > 0)
    ].tolist()
    if len(paired_samples) < 2:
        raise ValueError(
            "Fewer than two samples contain both exclusive-positive groups."
        )

    sample_names = []
    metadata_rows = []
    count_columns = []
    for sample in paired_samples:
        for group in ("MUC5B_only", "ANXA8_only"):
            pseudobulk_name = f"{sample}__{group}"
            sample_names.append(pseudobulk_name)
            count_columns.append(accumulators[(sample, group)])
            metadata_rows.append(
                {
                    "pseudobulk_id": pseudobulk_name,
                    "sample_id": sample,
                    "group": group,
                    "n_cells": summary_df.loc[sample, f"n_{group}"],
                }
            )

    count_df = pd.DataFrame(
        np.column_stack(count_columns), index=genes, columns=sample_names
    )
    count_df.index.name = "gene"
    count_df.to_csv(output_dir / "pseudobulk_raw_counts.tsv", sep="\t")

    metadata_df = pd.DataFrame(metadata_rows).set_index("pseudobulk_id")
    metadata_df.to_csv(output_dir / "pseudobulk_metadata.tsv", sep="\t")
    return paired_samples


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", default=DEFAULT_H5AD)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Number of CSR rows loaded per chunk.",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help="Override target sample IDs (default: 16 PDAC samples with tumor).",
    )
    parser.add_argument(
        "--skip-deseq2",
        action="store_true",
        help="Only create pseudobulk count and metadata files.",
    )
    parser.add_argument(
        "--rscript",
        default=DEFAULT_RSCRIPT,
        help="Path to Rscript with DESeq2 installed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_samples = list(args.samples) if args.samples else list(TARGET_SAMPLES)
    print(f"Using {len(target_samples)} samples: {', '.join(target_samples)}")

    paired_samples = aggregate_pseudobulk(
        Path(args.h5ad), output_dir, args.chunk_size, target_samples
    )
    print(f"Paired samples retained for DESeq2: {len(paired_samples)}")

    if args.skip_deseq2:
        return

    r_script = output_dir / "run_deseq2.R"
    r_script.write_text(R_CODE)
    command = [
        args.rscript,
        str(r_script),
        str(output_dir / "pseudobulk_raw_counts.tsv"),
        str(output_dir / "pseudobulk_metadata.tsv"),
        str(output_dir),
    ]
    print("Running paired DESeq2...")
    print(" ".join(command))
    subprocess.run(command, check=True)
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
