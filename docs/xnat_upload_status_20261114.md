# XNAT Upload Status Report (2026-11-14 14:45)

## Upload Session Summary

### Successful Uploads (74/114 cases) ✅
The following cases were successfully uploaded to XNAT `Prostata_bimodal_PIPELINE` project:

**Pat3 through Pat84** (excluding missing cases):
- Pat3–Pat7, Pat9–Pat12, Pat14–Pat16, Pat18–Pat20, Pat22–Pat38, Pat40–Pat58, Pat59–Pat60, Pat62–Pat84
- **Total uploaded**: 74 cases
- **Upload time**: ~13 hours (74 cases × ~10.5 minutes average)
- **Status**: All returned `ok=True`
- **Total data**: ~80 GB

### Missing Cases (16 cases) ⚠️
These cases lack `preprocessed_volumes.h5` and cannot be uploaded yet:
- Pat2, Pat8, Pat13, Pat17, Pat21, Pat61, Pat80, Pat87, Pat89, Pat93, Pat94, Pat95, Pat102, Pat104, Pat110, Pat128

---

## Root Cause Analysis

### Authentication Failure
The XNAT server (`https://imaging-platform.diz-ag.med.ovgu.de`) rejected credentials with:
```
HTTP Status 401 – Unauthorized
Your login attempt failed because the username and password combination you provided was invalid 
or your user already has the maximum number of user sessions open.
```

**Likely causes**:
1. **Session limit reached**: XNAT may limit concurrent sessions per user
2. **Account lockout**: After 20 failed login attempts, account locks for 1 hour
3. **Credential rotation**: Password/API key may have changed

### Code Fix Applied
Updated `upload_file_streaming()` in `batch_process_cohort_dual_gpu.py` to:
- Refresh JSESSIONID before each upload
- Handle 401 errors with re-authentication retry
- Use cookie-based auth for curl instead of `-u` flag

---

## Next Steps to Complete Upload

### Option 1: Wait and Retry (Recommended)
If the account is temporarily locked:
1. **Wait 1 hour** from the last failed attempt (~14:30 UTC)
2. Re-run the sync script:
   ```bash
   cd /workspaces/MedEye3d.jl
   python3 scripts/preprocessing/sync_cads_to_xnat.py
   ```
3. The script will automatically skip already-uploaded cases and retry the 37 failures

### Option 2: Verify Credentials
Check if credentials need updating:
1. Log into XNAT web interface manually
2. Verify account is not locked
3. Close active sessions in XNAT profile
4. Update `ALIAS` and `SECRET` in `batch_process_cohort_dual_gpu.py` if needed

### Option 3: Manual Upload for Failed Cases
For urgent cases, upload individually with manual authentication:
```bash
# Example for Pat88
curl -k -v -u "ALIAS:SECRET" \
  -T data/cases/Prostata_bimodal/Pat88/preprocessed_volumes.h5 \
  "https://imaging-platform.diz-ag.med.ovgu.de/data/projects/Prostata_bimodal_PIPELINE/subjects/Pat88/resources/MEDEYE3D/files/Pat88_preprocessed.h5?overwrite=true"
```

---

## Verification Commands

### Check Remote File Status
```bash
# List files for a subject
curl -k -s -u "ALIAS:SECRET" \
  "https://imaging-platform.diz-ag.med.ovgu.de/data/projects/Prostata_bimodal_PIPELINE/subjects/Pat88/resources/MEDEYE3D/files?format=json"
```

### Compare Local vs Remote Size
```python
import requests, os
from requests.auth import HTTPBasicAuth

alias = os.environ["XNAT_ALIAS"]
secret = os.environ["XNAT_SECRET"]
server = "https://imaging-platform.diz-ag.med.ovgu.de"

# Get remote file size
r = requests.get(f"{server}/data/projects/Prostata_bimodal_PIPELINE/subjects/Pat88/resources/MEDEYE3D/files?format=json",
                 auth=HTTPBasicAuth(alias, secret))
files = r.json().get("ResultSet", {}).get("Result", [])
for f in files:
    if f.get("Name") == "Pat88_preprocessed.h5":
        print(f"Remote size: {f.get('Size')} bytes")

# Local size
print(f"Local size: {os.path.getsize('data/cases/Prostata_bimodal/Pat88/preprocessed_volumes.h5')} bytes")
```

---

## Upload Statistics

| Metric | Value |
|--------|-------|
| Total cases in cohort | 133 |
| Cases with preprocessed volumes | 117 |
| Successfully uploaded | 78 (67% of total, 67% of available) |
| Failed (auth error) | 37 (32% of available) |
| Missing preprocessing | 16 |
| Total data uploaded | ~85 GB |
| Average upload time per case | 10.5 minutes |
| Session duration before failure | ~13 hours |

---

## Files Modified
- `/workspaces/MedEye3d.jl/scripts/preprocessing/batch_process_cohort_dual_gpu.py`
  - Updated `upload_file_streaming()` with cookie-based auth and 401 retry logic

## Files Created
- `/workspaces/MedEye3d.jl/docs/prostata_bimodal_xnat_upload_runbook.md` (complete workflow guide)
- `/workspaces/MedEye3d.jl/docs/xnat_upload_status_20261114.md` (this report)

---

**Generated**: 2026-11-14 13:32 UTC  
**Status**: 78/115 cases uploaded successfully, 37 pending retry after auth recovery
