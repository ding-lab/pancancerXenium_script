# 2026/02/05
# edited by Winnie Pan 
# (Optimized for Memory & File Size)
import os
import numpy as np
import pandas as pd
import sys
import argparse
import random
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy.external as sce
import gc  # Added for garbage collection

def log(message):
    print(f"[LOG] {message}", flush=True)


def setup_scanpy_settings():
    sc.settings.verbosity = 3  # verbosity level
    sc.settings.set_figure_params(dpi=300, facecolor='white', figsize=(10, 8), frameon=False)


def normalize_anndata(adata, copy=False):
    if copy:
        adata = adata.copy()
    
    log("Storing raw counts in layers['counts']")
    adata.layers["counts"] = adata.X.copy()
    
    log("Normalizing total counts per cell")
    sc.pp.normalize_total(adata)
    
    log("Log transforming")
    sc.pp.log1p(adata)
    
    log("Storing log-normalized data in layers['lognorm']")
    adata.layers["lognorm"] = adata.X.copy()
    
    return adata


def run_harmony_integration(adata, batch_key='dataset', n_pcs=50, max_iter_harmony=10):
    """
    Run Harmony batch correction.
    """
    log(f"Running Harmony integration (batch_key='{batch_key}', n_pcs={n_pcs})")
    
    sce.pp.harmony_integrate(adata, key=batch_key, basis="X_pca", adjusted_basis="X_pca_harmony", max_iter_harmony=max_iter_harmony)
    
    log(f"Harmony integration complete. Stored in adata.obsm['X_pca_harmony']")
    
    return adata

def process_single_cancer_type(input_file, output_file, batch_key='dataset', n_pcs=50, n_neighbors=30,
                               leiden_resolutions=[0.2, 0.3, 0.5], seed=1234, use_harmony=True, max_iter_harmony=10):
    
    # Set seeds
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "50")

    log(f"Processing: {input_file}")

    # Load AnnData
    adata = sc.read_h5ad(input_file)
    log(f"Loaded AnnData: {adata.n_obs:,} cells × {adata.n_vars:,} genes (Xenium 5K Mode)")
    
    # sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    # n_cells_before = adata.n_obs
    # adata = adata[adata.obs['n_genes_by_counts'] > 20, :].copy() # 至少表达20个基因  loose cutoff for MM
    # adata = adata[adata.obs['total_counts'] > 50, :].copy()     # 至少有50个转录本
    # log(f"QC Filtering: Removed {n_cells_before - adata.n_obs} low-quality cells.")
    
    # 1. Normalization
    if "lognorm" not in adata.layers:
        log("Normalizing data...")
        adata = normalize_anndata(adata, copy=False)
    else:
        log("Using existing lognorm layer")
        adata.X = adata.layers["lognorm"].copy()

    # =========================================================
    # Step 2: HVG Selection (Targeting Top 3000 for 5K panel)
    # =========================================================
    # log(f"Finding highly variable genes...")
    # try:
        # sc.pp.highly_variable_genes(
            # adata, 
            # batch_key=batch_key, 
            # subset=False,
            # flavor='seurat'
        # )
    # except Exception as e:
        # log(f"Warning: Failed to use batch-aware HVG ({e}), falling back to standard HVG.")
        # sc.pp.highly_variable_genes(adata)

    # n_hvg = int(adata.var["highly_variable"].sum())
    # log(f"HVGs selected: {n_hvg}")

    # =========================================================
    # Step 3: Low-Memory Scale & PCA Strategy (The Fix)
    # =========================================================
    # Instead of scaling the whole adata (which causes OOM and huge files),
    # we extract HVGs to a temp object, scale that, run PCA, and copy result back.
    
    # log("Creating temporary HVG subset for PCA (saving memory)...")
    
    # 3.1 Extract HVGs to a new, smaller AnnData object
    # .copy() is crucial to detach it from the main object
    # adata_hvg = adata[:, adata.var['highly_variable']].copy()
    
    # 3.2 Scale ONLY the subset
    # We can safely use zero_center=True here because the matrix is small (3000 genes)
    # log("Scaling HVG subset (zero mean, unit variance)...")
    sc.pp.scale(adata)
    
    # 3.3 Run PCA on the subset
    log(f"Running PCA on HVG subset (n_pcs={n_pcs}, seed={seed})")
    sc.pp.pca(adata, n_comps=n_pcs, random_state=seed)
    
    # 3.4 Transfer PCA coordinates back to the main adata
    log("Transferring PCA results to main AnnData...")
    # adata.obsm['X_pca'] = adata.obsm['X_pca'].copy()
    
    # Optional: Transfer variance info if needed for elbow plots
    # adata.uns['pca'] = adata.uns['pca']
    
    # 3.5 Cleanup to free memory immediately
    # del adata_hvg
    gc.collect()
    
    log("PCA calculation complete. Main matrix remains sparse.")

    # =========================================================

    # Harmony integration
    if use_harmony:
        if batch_key not in adata.obs.columns:
            raise KeyError(f"Batch key '{batch_key}' not found.")
        
        log(f"Running Harmony (batch_key={batch_key})...")
        try:
            adata = run_harmony_integration(adata, batch_key=batch_key, n_pcs=n_pcs, max_iter_harmony=max_iter_harmony)
        except Exception as e:
            raise RuntimeError(f"Harmony failed: {e}")
    else:
        raise RuntimeError("Harmony disabled but required.")

    # Compute neighbors and UMAP
    if 'X_pca_harmony' not in adata.obsm:
        raise RuntimeError("X_pca_harmony missing.")

    log(f"Computing neighbors/UMAP on Harmony PCA...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, metric="cosine",random_state=seed, use_rep='X_pca_harmony', key_added="neighbors_harmony")
    sc.tl.umap(adata, random_state=seed, neighbors_key="neighbors_harmony")

    # Store UMAP
    adata.obsm['X_umap_harmony'] = adata.obsm['X_umap'].copy()
    
    # Leiden Clustering
    if "neighbors_harmony" in adata.uns:
        for r in leiden_resolutions:
            key = f"leiden_harmony_{r:g}"
            sc.tl.leiden(adata, resolution=r, key_added=key,  random_state=seed, neighbors_key="neighbors_harmony")
            log(f"→ {key}: {adata.obs[key].nunique()} clusters")

    # Save AnnData
    # Since we didn't touch adata.X (only X_pca), the file remains small and sparse!
    log(f"Saving AnnData: {output_file}")
    adata.write(output_file)
    
    # Plotting
    if 'X_umap_harmony' in adata.obsm:
        plot_file = os.path.splitext(output_file)[0] + "_umap_harmony.pdf"
        
        # Prepare color keys (batch + leiden clusters)
        color_keys = [batch_key] + [k for k in adata.obs.columns if k.startswith("leiden_harmony_")]
        
        # Simple plot logic
        sc.pl.umap(adata, color=color_keys, save=False, show=False, ncols=3, frameon=False)
        plt.savefig(plot_file, bbox_inches='tight', dpi=300)
        plt.close()
        log(f"Harmony UMAP plot saved to {plot_file}")

    return adata

def main():
    parser = argparse.ArgumentParser(description="Process merged Xenium AnnData files with normalization, PCA, Harmony, UMAP, and Leiden clustering.")
    parser.add_argument("--input_dir", required=True, help="Path to directory containing merged .h5ad files (one per cancer type)")
    parser.add_argument("--output_dir", required=True, help="Path to output directory for processed files")
    parser.add_argument("--batch_key", default="dataset", help="Column name in obs to use for batch correction (default: 'dataset')")
    parser.add_argument("--n_pcs", type=int, nargs="+", default=[50], help="Number of principal components (default: 50)")
    parser.add_argument("--n_neighbors", type=int, default=30, help="Number of neighbors for graph construction (default: 15)")
    parser.add_argument("--leiden_resolutions", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7],
                        help="Leiden clustering resolutions (default: 0.3 0.4 0.5 0.6 0.7)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility (default: 1234)")
    parser.add_argument("--no_harmony", action="store_true", help="Skip Harmony integration (use standard PCA)")
    parser.add_argument("--max_iter_harmony", type=int, default=10, help="Maximum iterations for Harmony (default: 10)")
    parser.add_argument("--pattern", default="*_merged.h5ad", help="File pattern to match input files (default: '*_merged.h5ad')")
    args = parser.parse_args()

    setup_scanpy_settings()
    os.makedirs(args.output_dir, exist_ok=True)

    input_dir = Path(args.input_dir)
    input_files = list(input_dir.glob(args.pattern))

    if not input_files:
        log(f"ERROR: No files found matching pattern '{args.pattern}' in {args.input_dir}")
        sys.exit(1)

    log(f"Found {len(input_files)} file(s) to process")
    log(f"Will process with n_pcs values: {args.n_pcs}")

    for input_file in sorted(input_files):
        for n_pcs in args.n_pcs:
            log(f"\n{'='*60}")
            log(f"Processing: {input_file.name} with n_pcs={n_pcs}")
            log(f"{'='*60}")

            if len(args.n_pcs) > 1:
                output_filename = input_file.stem + f"_processed_cosine_n30_npcs{n_pcs}.h5ad"
            else:
                output_filename = input_file.stem + "_processed_cosine.h5ad"
            output_file = os.path.join(args.output_dir, output_filename)

            try:
                process_single_cancer_type(
                    input_file=str(input_file),
                    output_file=output_file,
                    batch_key=args.batch_key,
                    n_pcs=n_pcs,
                    n_neighbors=args.n_neighbors,
                    leiden_resolutions=args.leiden_resolutions,
                    seed=args.seed,
                    use_harmony=not args.no_harmony,
                    max_iter_harmony=args.max_iter_harmony
                )
            except Exception as e:
                log(f"ERROR processing {input_file.name} with n_pcs={n_pcs}: {e}")
                import traceback
                traceback.print_exc()
                continue

    log("\n" + "="*60)
    log("All processing complete!")
    log("="*60)


if __name__ == "__main__":
    main()