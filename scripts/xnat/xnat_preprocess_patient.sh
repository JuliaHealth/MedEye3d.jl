#!/usr/bin/env bash
# ==============================================================================
# xnat_preprocess_patient.sh
#
# Master bash orchestrator to download raw scans from XNAT, rerun Skellytour,
# rerun the lymph node rules engine, build preprocessed HDF5, and upload to XNAT.
#
# Examples:
#   ./scripts/xnat/xnat_preprocess_patient.sh --project "FDM_DPI-2024-7-KRN_Lu177_PSMA_PIPELINE" --subject "pat6"
#   ./scripts/xnat/xnat_preprocess_patient.sh --project "FDM_DPI-2024-7-KRN_Prostata_bimodal_PIPELINE" --all
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PROJECT=""
SUBJECT=""
ALL_SUBJECTS=false
SERVER="${XNAT_SERVER:-}"
ALIAS="${XNAT_ALIAS:-}"
SECRET="${XNAT_SECRET:-}"
OVERWRITE=false
SKIP_SKELLY=false
KEEP_WORK_DIR=false
WORK_DIR_BASE="/tmp/medeye3d_xnat_preproc"

print_help() {
    cat << EOF
Usage: $(basename "$0") --project <PROJECT_ID> [--subject <SUBJECT_ID> | --all] [OPTIONS]

Required:
  --project <ID>         XNAT Project ID (e.g. FDM_DPI-2024-7-KRN_Lu177_PSMA_PIPELINE)
  --subject <ID>         Specific subject ID / label to process (e.g. pat6)
  --all                  Iterate and process ALL subjects in the project

Options:
  --server <URL>         XNAT Server URL (default: from cached credentials or XNAT_SERVER)
  --alias <TOKEN>        24h Alias Token (default: from cached credentials or XNAT_ALIAS)
  --secret <TOKEN>       24h Secret Token (default: from cached credentials or XNAT_SECRET)
  --work-dir <PATH>      Base directory for temporary data (default: /tmp/medeye3d_xnat_preproc)
  --overwrite            Force reprocessing and overwrite existing HDF5 on XNAT
  --skip-skellytour      Skip Skellytour bone segmentation rerun if already computed
  --keep-work-dir        Do not delete scratch DICOM files after uploading HDF5
  -h, --help             Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --subject)
            SUBJECT="$2"
            shift 2
            ;;
        --all)
            ALL_SUBJECTS=true
            shift
            ;;
        --server)
            SERVER="$2"
            shift 2
            ;;
        --alias)
            ALIAS="$2"
            shift 2
            ;;
        --secret)
            SECRET="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR_BASE="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        --skip-skellytour)
            SKIP_SKELLY=true
            shift
            ;;
        --keep-work-dir)
            KEEP_WORK_DIR=true
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

if [ -z "$PROJECT" ]; then
    echo "ERROR: --project <ID> is required."
    print_help
    exit 1
fi

if [ -z "$SUBJECT" ] && [ "$ALL_SUBJECTS" = false ]; then
    echo "ERROR: Either --subject <ID> or --all must be specified."
    print_help
    exit 1
fi

# Set environment credentials if provided via CLI
if [ -n "$SERVER" ]; then export XNAT_SERVER="$SERVER"; fi
if [ -n "$ALIAS" ]; then export XNAT_ALIAS="$ALIAS"; fi
if [ -n "$SECRET" ]; then export XNAT_SECRET="$SECRET"; fi

cd "$REPO_ROOT"

# Check authentication via Julia XnatAuth
echo "=============================================================================="
echo " XNAT PREPROCESSING & UPLOAD PIPELINE"
echo " Project: $PROJECT"
echo "=============================================================================="
echo "▶ Verifying XNAT credentials..."

AUTH_CHECK=$(julia --project=. -e '
using MedEye3d.XnatAuth
ok, creds, msg = XnatAuth.auto_verify_credentials()
if ok
    println("AUTH_OK:", creds.server_url)
else
    println("AUTH_FAIL:", msg)
end
')

if [[ "$AUTH_CHECK" == AUTH_FAIL* ]]; then
    echo "❌ Authentication failed: ${AUTH_CHECK#AUTH_FAIL:}"
    echo "Please configure credentials in %APPDATA%/MedEye3D/xnat_credentials.json or pass --alias and --secret."
    exit 1
fi

SERVER_URL="${AUTH_CHECK#AUTH_OK:}"
echo "✓ Authenticated with XNAT: $SERVER_URL"

# Function to process a single patient
process_patient() {
    local SUBJ="$1"
    local SUBJ_WORK_DIR="$WORK_DIR_BASE/$PROJECT/$SUBJ"
    local H5_FILE="$SUBJ_WORK_DIR/${SUBJ}_preprocessed.h5"

    echo "------------------------------------------------------------------------------"
    echo "Processing Subject: $SUBJ"
    echo "Workspace:          $SUBJ_WORK_DIR"
    echo "------------------------------------------------------------------------------"

    # Check status on XNAT if not overwrite
    if [ "$OVERWRITE" = false ]; then
        STATUS=$(julia --project=. -e '
using MedEye3d.XnatAuth, MedEye3d.XnatClient
ok, creds, _ = XnatAuth.auto_verify_credentials()
st = XnatClient.get_case_status(creds, ARGS[1], ARGS[2])
println(st.status)
' "$PROJECT" "$SUBJ")
        if [ "$STATUS" = "PROCESSED" ] || [ "$STATUS" = "PREPROCESSED" ]; then
            echo "ℹ Case $SUBJ is already $STATUS on XNAT. Skipping (use --overwrite to force)."
            return 0
        fi
    fi

    mkdir -p "$SUBJ_WORK_DIR"

    # 1. Download scans
    echo "Step 1/4: Downloading scans from XNAT..."
    julia --project=. "$SCRIPT_DIR/xnat_preproc_worker.jl" download \
        --project "$PROJECT" --subject "$SUBJ" --out-dir "$SUBJ_WORK_DIR"

    # 2. Process: Skellytour + Lymph Node Rules + Alignment + HDF5 packing
    echo "Step 2/4: Running full preprocessing (Skellytour, Lymph Node Rules, Registration)..."
    EXTRA_FLAGS=""
    if [ "$SKIP_SKELLY" = true ]; then
        EXTRA_FLAGS="--skip-skellytour"
    fi
    julia --project=. "$SCRIPT_DIR/xnat_preproc_worker.jl" process \
        --dir "$SUBJ_WORK_DIR" --out-h5 "${SUBJ}_preprocessed.h5" $EXTRA_FLAGS

    # 3. Upload to XNAT
    echo "Step 3/4: Uploading preprocessed HDF5 to XNAT..."
    julia --project=. "$SCRIPT_DIR/xnat_preproc_worker.jl" upload \
        --project "$PROJECT" --subject "$SUBJ" --h5 "$H5_FILE" --status "PREPROCESSED"

    # 4. Cleanup
    if [ "$KEEP_WORK_DIR" = false ]; then
        echo "Step 4/4: Cleaning up temporary scan files..."
        rm -rf "$SUBJ_WORK_DIR/exp_"* "$SUBJ_WORK_DIR/"*.zip
        echo "✓ Cleaned up scratch scan files."
    else
        echo "Step 4/4: Kept scratch files at $SUBJ_WORK_DIR."
    fi

    echo "✅ Successfully completed: $PROJECT / $SUBJ"
}

if [ "$ALL_SUBJECTS" = true ]; then
    echo "▶ Querying all subjects for project: $PROJECT..."
    SUBJECTS=$(julia --project=. -e '
using MedEye3d.XnatAuth, MedEye3d.XnatClient
ok, creds, _ = XnatAuth.auto_verify_credentials()
subjects = XnatClient.list_subjects(creds, ARGS[1])
for s in subjects
    lbl = get(s, "label", get(s, "ID", ""))
    !isempty(lbl) && println(lbl)
end
' "$PROJECT")

    for s in $SUBJECTS; do
        process_patient "$s"
    done
else
    process_patient "$SUBJECT"
fi

echo "=============================================================================="
echo " ALL REQUESTED CASES PROCESSED SUCCESSFULLY"
echo "=============================================================================="
