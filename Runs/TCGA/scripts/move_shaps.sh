#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_ROOT="${SRC_ROOT:-$RUN_DIR/output/multitask_nn}"
DST_ROOT="${DST_ROOT:-$RUN_DIR/output/lean}"

# Default is copy. Run with:
#   ./move_shaps.sh
# or move instead of copy:
#   ./move_shaps.sh move
ACTION="${1:-copy}"

if [[ "$ACTION" != "copy" && "$ACTION" != "move" ]]; then
  echo "ERROR: ACTION must be 'copy' or 'move'. Got: $ACTION" >&2
  exit 1
fi

pairs=(
  "ACC CTNNB1"
  "ACC MEN1"
  "ACC NF1"
  "ACC PRKAR1A"
  "BLCA FGFR3"
  "BLCA NFE2L2"
  "BLCA RB1"
  "BLCA TP53"
  "BRCA CDH1"
  "BRCA GATA3"
  "BRCA MAP3K1"
  "BRCA PIK3CA"
  "CESC ARID1A"
  "CESC GOLGB1"
  "CESC MUC17"
  "CESC PTEN"
  "COAD APC"
  "COAD BRAF"
  "COAD DCHS2"
  "COAD MUC5B"
  "COAD RNF43"
  "GBM ATRX"
  "GBM IDH1"
  "GBM RYR3"
  "HNSC CASP8"
  "HNSC HRAS"
  "HNSC NSD1"
  "LGG ATRX"
  "LGG CIC"
  "LGG IDH1"
  "LIHC CTNNB1"
  "LUAD EGFR"
  "LUAD KEAP1"
  "LUAD STK11"
  "LUSC NFE2L2"
  "MESO LATS2"
  "PAAD GNAS"
  "READ APC"
  "READ SOX9"
  "UVM EIF1AX"
  "UVM SF3B1"
)

# One shared destination folder for all cancers.
DST_SHAP_DIR="$DST_ROOT/shap_beeswarm"
mkdir -p "$DST_SHAP_DIR"

missing=0
handled=0

for pair in "${pairs[@]}"; do
  cancer="${pair%% *}"
  gene="${pair##* }"

  src_dir="$SRC_ROOT/$cancer/shap"
  src_file="$src_dir/beeswarm_${gene}.parquet"
  dst_file="$DST_SHAP_DIR/${cancer}_beeswarm_${gene}.parquet"

  if [[ ! -f "$src_file" ]]; then
    echo "MISSING: $src_file"
    missing=$((missing + 1))
    continue
  fi

  if [[ "$ACTION" == "move" ]]; then
    mv "$src_file" "$dst_file"
  else
    cp -n "$src_file" "$dst_file"
  fi

  # Shared app metadata useful for plotting/annotation.
  # Prefix with cancer to avoid overwriting files from other cancers.
  for shared in kfold_summary.csv shap_summary_feature_summary_matrix.csv; do
    if [[ -f "$src_dir/$shared" ]]; then
      cp -n "$src_dir/$shared" "$DST_SHAP_DIR/${cancer}_${shared}"
    fi
  done

  echo "OK: $cancer $gene -> $dst_file"
  handled=$((handled + 1))
done

echo "Done. Files handled: $handled"
echo "Missing files: $missing"
echo "Destination: $DST_SHAP_DIR/"
