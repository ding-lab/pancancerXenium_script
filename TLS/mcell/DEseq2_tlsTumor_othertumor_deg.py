# !/storage1/fs1/dinglab/Active/Projects/yuweiz/conda/envs/rapid_sc/bin/python3
import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial import KDTree
from scipy.sparse import issparse
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

print(f"{'='*60}")
print("TLS Boundary PyDESeq2 Analysis")
print(f"{'='*60}")

CANCER_TYPES = ['LUNG']

# Paths
DIR_TLS_SCORES = "/rdcw/fs1/dinglab/Active/Projects/pdc/projects/general/x1000/tls/tls_define/gen8_split/outputs/5k"
DIR_BOUNDARY = "/rdcw/fs1/dinglab/Active/Projects/pdc/projects/general/x1000/boundary/gen10/5k/outputs"
DIR_BOUNDARY_LUNG = "/storage1/fs1/dinglab/Active/Projects/pdc/projects/general/x1000/boundary/gen12_pdac-merged-lung-mcell2/outputs/LUNG/data"
DIR_H5AD = "/rdcw/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/01_cell_typing"
OUT_DIR = "/rdcw/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/01_cell_typing/01_analysis/03_TLS/tls_deg_output_100um_10"

# Threshold parameters
BOUNDARY_THRESHOLD_UM = 50.0   # Cell is on boundary if abs(dist_iface_um) <= 50
NEAR_TLS_THRESHOLD_UM = 100.0  # Cell is near TLS if Euclidean dist to nearest TLS centroid <= 200
MIN_CELLS_PER_GROUP = 10       # Require at least 10 cells in a group to form a pseudo-bulk profile

os.makedirs(OUT_DIR, exist_ok=True)

def get_raw_counts(adata_sub):
    """Safely extract raw counts and variable names from anndata object."""
    if 'counts' in adata_sub.layers:
        X = adata_sub.layers['counts']
        vars = adata_sub.var_names
    elif adata_sub.raw is not None:
        X = adata_sub.raw.X
        vars = adata_sub.raw.var_names
    else:
        X = adata_sub.X
        vars = adata_sub.var_names
        
    if issparse(X):
        return np.array(X.sum(axis=0)).flatten(), vars
    else:
        return np.array(X.sum(axis=0)), vars

for ct in CANCER_TYPES:
    print(f"\n[{ct}] Processing...")
    
    h5ad_path = os.path.join(DIR_H5AD, ct, "annotated.Yuwei.new.h5ad")
    if not os.path.exists(h5ad_path):
        print(f"  [SKIP] H5AD not found: {h5ad_path}")
        continue
        
    print(f"  -> Loading {h5ad_path}")
    # Backed mode reduces memory overhead for large spatial datasets
    adata = ad.read_h5ad(h5ad_path, backed='r')
    
    ct_tls_dir = os.path.join(DIR_TLS_SCORES, ct)
    if not os.path.exists(ct_tls_dir):
        print(f"  [SKIP] TLS directory not found: {ct_tls_dir}")
        continue
        
    samples = [d for d in os.listdir(ct_tls_dir) if os.path.isdir(os.path.join(ct_tls_dir, d))]
    print(f"  -> Found {len(samples)} samples to check.")
    
    pb_counts = []
    pb_meta = []
    var_names_global = None
    
    for sid in samples:
        score_path = os.path.join(DIR_TLS_SCORES, ct, sid, f"{sid}_scores.csv")
        bound_path = (
            os.path.join(DIR_BOUNDARY_LUNG, f"{sid}_boundary_distance.csv")
            if ct == "LUNG"
            else os.path.join(
                DIR_BOUNDARY, ct, sid, f"{sid}_boundary_distance.csv"
            )
        )
        
        if not (os.path.exists(score_path) and os.path.exists(bound_path)):
            continue
            
        sdf = pd.read_csv(score_path)
        bdf = pd.read_csv(bound_path)
        
        # 'short_id' in sdf matches 'cell_id' in bdf
        sdf['short_id'] = sdf['cell_id'].str.split('_').str[0]
        merged = sdf.merge(bdf[['cell_id', 'dist_iface_um']], left_on='short_id', right_on='cell_id', how='inner')
        
        # TLS cells
        tls_cells = merged[merged['region'] == 'TLS'].copy()
        if tls_cells.empty:
            continue
            
        # Get obs to find sample indices
        obs_df = adata.obs.copy()
        obs_df['short_id'] = obs_df.index.str.split('_').str[0]
        obs_df['sample_id'] = obs_df.index.str.split('_', n=1).str[1]
        
        s_idx_num = np.where(obs_df['sample_id'] == sid)[0]
        if len(s_idx_num) == 0:
            continue
            
        s_obs = obs_df.iloc[s_idx_num]
        s_spatial = adata.obsm['spatial'][s_idx_num]
        
        coord_df = pd.DataFrame(s_spatial, index=s_obs['short_id'], columns=['x', 'y'])
        
        # Join coordinates to TLS cells
        tls_cells = tls_cells.join(coord_df, on='short_id', how='inner')
        if tls_cells.empty:
            continue
            
        # TLS Centroids
        centroids = []
        if 'tls_region_number' in tls_cells.columns:
            for _, grp in tls_cells.groupby('tls_region_number'):
                centroids.append(grp[['x', 'y']].mean().values)
        else:
            centroids.append(tls_cells[['x', 'y']].mean().values)
            
        if not centroids:
            continue
            
        # Identify tumor cells
        tumor_merged = merged[merged['celltype'].isin(['Tumor', 'AT2'])].copy()
        tumor_merged = tumor_merged.join(coord_df, on='short_id', how='inner')
        if tumor_merged.empty:
            continue
            
        # Build KDTree and find distance to nearest TLS centroid
        kdtree = KDTree(centroids)
        coords = tumor_merged[['x', 'y']].values
        dists, _ = kdtree.query(coords)
        tumor_merged['dist_to_tls'] = dists
        
        # Group definitions
        is_boundary = tumor_merged['dist_iface_um'].abs() <= BOUNDARY_THRESHOLD_UM
        is_near_tls = tumor_merged['dist_to_tls'] <= NEAR_TLS_THRESHOLD_UM
        
        # 1. Boundary Tumor Cells Near TLS
        group_near = tumor_merged[is_boundary & is_near_tls]
        # 2. Other Tumor Cells (All other tumor cells on the boundary)
        group_other = tumor_merged[is_boundary & (~is_near_tls)]
        
        # Need enough cells to form pseudo-bulk
        if len(group_near) < MIN_CELLS_PER_GROUP or len(group_other) < MIN_CELLS_PER_GROUP:
            print(f"      [SKIP] {sid}: Low cells (Near={len(group_near)}, Other={len(group_other)})")
            continue
            
        print(f"      [ADD]  {sid}: Near={len(group_near)}, Other={len(group_other)}")
        
        # Match back to adata.obs_names
        near_idx_names = s_obs.index[s_obs['short_id'].isin(group_near['short_id'])]
        other_idx_names = s_obs.index[s_obs['short_id'].isin(group_other['short_id'])]
        
        # Extract subset to memory
        adata_near = adata[near_idx_names].to_memory()
        adata_other = adata[other_idx_names].to_memory()
        
        # Sum raw counts (Pseudo-bulk)
        sum_near, vnames = get_raw_counts(adata_near)
        sum_other, _ = get_raw_counts(adata_other)
        
        if var_names_global is None:
            var_names_global = vnames
            
        pb_counts.append(sum_near)
        pb_meta.append({'SampleID': sid, 'Condition': 'BoundaryNearTLS'})
        
        pb_counts.append(sum_other)
        pb_meta.append({'SampleID': sid, 'Condition': 'OtherTumor'})
        
    if not pb_meta:
        print(f"  [RESULT] No overlap found for {ct} (skipping DEG).")
        continue
        
    cond_counts = pd.DataFrame(pb_meta)['Condition'].value_counts()
    if len(cond_counts) < 2 or cond_counts.min() < 2:
        print(f"  [RESULT] Not enough samples for {ct} DESeq2 (need at least 2 per condition). Found: {cond_counts.to_dict()}")
        continue
        
    # Prepare PyDESeq2 Inputs
    counts_df = pd.DataFrame(pb_counts, columns=var_names_global)
    counts_df = counts_df.fillna(0).round().astype(int)
    
    meta_df = pd.DataFrame(pb_meta)
    meta_df.index = [f"{r.SampleID}_{r.Condition}" for _, r in meta_df.iterrows()]
    counts_df.index = meta_df.index
    
    # Filter lowly expressed genes across the pseudo-bulk samples
    counts_df = counts_df.loc[:, (counts_df.sum(axis=0) >= 10)]
    
    print(f"\n  -> Running PyDESeq2 for {ct} (Samples: {len(meta_df)//2}, Genes: {counts_df.shape[1]})...")
    try:
        dds = DeseqDataSet(
            counts=counts_df,
            metadata=meta_df,
            design_factors=["SampleID", "Condition"],
            refit_cooks=True,
            n_cpus=8
        )
        dds.deseq2()
        
        stat_res = DeseqStats(dds, contrast=("Condition", "BoundaryNearTLS", "OtherTumor"), n_cpus=8)
        stat_res.summary()
        
        res_df = stat_res.results_df
        if res_df is not None:
            res_df = res_df.dropna(subset=['pvalue']).sort_values('padj')
            
            # Add significance flags
            res_df['Significant'] = res_df['padj'] < 0.05
            
            out_file = os.path.join(OUT_DIR, f"{ct}_DEG_BoundaryNearTLS_vs_OtherTumor.csv")
            res_df.to_csv(out_file)
            sig_count = res_df['Significant'].sum()
            print(f"  -> Success! Saved to {out_file} ({sig_count} significant genes)")
        else:
            print(f"  [WARNING] DESeq2 produced no results dataframe for {ct}.")
            
    except Exception as e:
        print(f"  [ERROR] PyDESeq2 failed for {ct}: {e}")

print("\nFinished DEG Analysis Pipeline.")

