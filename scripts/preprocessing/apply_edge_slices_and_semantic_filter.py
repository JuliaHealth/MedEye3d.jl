#!/usr/bin/env python3
"""
1. Unconditionally remove lesion voxels in the first 2 and last 2 Z slices
   (`z in {0, 1, Z-2, Z-1}`) across all Prostata_bimodal `preprocessed_volumes.h5`
   files (independent of semantic filter), updating CENTROIDS, organ_mapping,
   segment_names.json, matches.json, and scene_hierarchy.json.
2. Call `qwen3.8-27b` on the DIZ KI-Plattform API to extract all structures
   from `_meta_/max_anatomy_labels.json` that correspond to pathological /
   suspicious / lesion / metastatic findings in `_meta_/metadata.json` dictations,
   and persist `_meta_/semantic_filter.json`.

Usage:
  python apply_edge_slices_and_semantic_filter.py [--cases Pat1 Pat51] [--workers 6] [--sync-cache] [--force-llm]
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np
import requests

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CASES_DIR = os.path.join(REPO_ROOT, "data", "cases", "Prostata_bimodal")
CACHE_ROOT = os.path.expanduser("~/.medeye3d/xnat_cache/Prostata_bimodal_PIPELINE")
CENSUS_PATH = os.path.join(REPO_ROOT, "data", "xnat_bimodal_census.json")

sys.path.insert(0, SCRIPT_DIR)
from cads_anatomy_wiring import remove_edge_slice_lesions


def _load_dotenv(path: str, target: dict):
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in target:
                target[k] = v


def get_llm_config() -> tuple:
    cfg = {}
    for k in ("DIZ_API_KEY", "DIZ_API_BASE", "SEMANTIC_LLM_MODEL"):
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    _load_dotenv(os.path.join(REPO_ROOT, ".env"), cfg)
    _load_dotenv("/mnt/big/project_ssd/slicer_lesion_text_extension/data/.env", cfg)
    _load_dotenv("/mnt/big/project_ssd/gemini_manag_code/.env.docker", cfg)
    api_key = cfg.get("DIZ_API_KEY", "")
    api_base = cfg.get("DIZ_API_BASE", "https://ki-plattform.diz-ag.med.ovgu.de/api/").rstrip("/")
    model = cfg.get("SEMANTIC_LLM_MODEL", "qwen3.8-27b")
    return api_base, api_key, model


def _normalize_organ_tokens(name: str) -> set:
    """Normalize raw `max_anatomy` and formatted `organ_mapping` names to overlapping token sets."""
    s = (name or "").strip().lower()
    if not s or s in ("unknown", "background"):
        return set()
    for pfx in ("nvseg_", "skellytour_", "totalseg_", "abdominal_", "pelvic_", "thoracic_", "neck_"):
        s = s.replace(pfx, "")
    s = re.sub(r"lymph\s*node[s]?", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    tokens = {t for t in s.split() if t and t not in ("of", "the", "and", "in", "area", "station", "level")}
    return tokens


CANONICAL_LABELS_PATH = os.path.join(REPO_ROOT, "assets", "canonical_max_anatomy_labels.json")
_CANONICAL_LABELS_CACHE = None


def _get_canonical_labels() -> set:
    global _CANONICAL_LABELS_CACHE
    if _CANONICAL_LABELS_CACHE is None:
        if os.path.isfile(CANONICAL_LABELS_PATH):
            with open(CANONICAL_LABELS_PATH, "r", encoding="utf-8") as cf:
                _CANONICAL_LABELS_CACHE = {str(x).strip() for x in json.load(cf) if str(x).strip()}
        else:
            _CANONICAL_LABELS_CACHE = set()
    return _CANONICAL_LABELS_CACHE


_PARENCHYMAL_SINGLETONS = frozenset({
    "liver", "spleen", "pancreas", "prostate", "thyroid",
    "stomach", "esophagus", "heart", "lung", "brain", "sacrum",
})
_PELVIC_BONE_TOKENS = frozenset({"hip", "ilium", "ilii", "pubis", "pubic", "ischium", "acetabulum"})


def organ_matches_mentioned(organ_name: str, mentioned_list: list) -> bool:
    """Return True if an organ name matches any structure in `mentioned_list`."""
    onorm = (organ_name or "").strip().lower()
    if not onorm or onorm in ("unknown", "background"):
        return False
    m_low = {str(m).strip().lower() for m in mentioned_list if str(m).strip()}
    if onorm in m_low:
        return True
    # Normalize nvseg_ / formatting equivalence (e.g. 'prostate' <-> 'nvseg_prostate' <-> 'prostatic urethra')
    o_core = re.sub(r"^nvseg_", "", onorm).replace("_", " ").strip()
    for m in m_low:
        m_core = re.sub(r"^nvseg_", "", m).replace("_", " ").strip()
        if o_core == m_core:
            return True
        if o_core == "prostatic urethra" and "prostate" in m_core:
            return True
        if "prostate" in m_core and "prostate" in o_core:
            return True
        if "seminal vesicle" in o_core and "seminal vesicle" in m_core:
            return True
    o_toks = _normalize_organ_tokens(organ_name)
    if not o_toks:
        return False
    o_has_pelvis = bool(o_toks & _PELVIC_BONE_TOKENS) or ("pelvis" in o_toks and "bone" in o_toks)
    o_sides = o_toks & {"left", "right"}
    for m in m_low:
        m_toks = _normalize_organ_tokens(m)
        if not m_toks:
            continue
        if o_toks == m_toks or (len(m_toks) >= 2 and m_toks.issubset(o_toks)) or (len(o_toks) >= 2 and o_toks.issubset(m_toks)):
            return True
        # Match single-token parent organs against segmental sub-labels (e.g. liver <-> liver_segment_6), excluding cyst labels
        if "cyst" not in o_toks and "cyst" not in m_toks:
            if len(o_toks) == 1 and next(iter(o_toks)) in _PARENCHYMAL_SINGLETONS and o_toks.issubset(m_toks):
                return True
            if len(m_toks) == 1 and next(iter(m_toks)) in _PARENCHYMAL_SINGLETONS and m_toks.issubset(o_toks):
                return True
        # Pelvic os coxae aliases (hip_left <-> ilium/pubis/ischium/pelvis_bone left)
        m_has_pelvis = bool(m_toks & _PELVIC_BONE_TOKENS) or ("pelvis" in m_toks and "bone" in m_toks)
        if o_has_pelvis and m_has_pelvis:
            m_sides = m_toks & {"left", "right"}
            if not o_sides or not m_sides or o_sides == m_sides:
                return True
    return False


def strip_edge_slices_in_h5(h5_path: str, edge_slices: int = 2) -> dict:
    """Remove lesions in first/last `edge_slices` Z slices and keep HDF5 metadata consistent."""
    removed_ids = []
    trimmed = False
    with h5py.File(h5_path, "r+") as hf:
        if "BASELINE" not in hf or "PET_Lesions_0.nii.gz" not in hf["BASELINE"]:
            return {"trimmed": False, "removed_ids": []}
        b = hf["BASELINE"]
        les_ds = b["PET_Lesions_0.nii.gz"]
        attrs_0 = {a: les_ds.attrs[a] for a in ("spacing", "origin", "direction") if a in les_ds.attrs}
        les_arr = les_ds[:].astype(np.int16)

        z_dim = les_arr.shape[0]
        had_edge = z_dim > 2 * edge_slices and bool(np.any(les_arr[:edge_slices] > 0) or np.any(les_arr[-edge_slices:] > 0))
        if had_edge:
            removed_ids = remove_edge_slice_lesions(les_arr, edge_slices=edge_slices)
            del b["PET_Lesions_0.nii.gz"]
            ds_new = b.create_dataset("PET_Lesions_0.nii.gz", data=les_arr, compression="gzip", compression_opts=4, shuffle=True)
            for ak, av in attrs_0.items():
                ds_new.attrs[ak] = av
            trimmed = True

        # Also clean MRI TFM_* lesion volumes of edge-slice artifacts
        mr_gold_arr = np.zeros_like(les_arr)
        tfm_keys = sorted([k for k in hf.keys() if k.startswith("TFM_Transform_FollowUp_to_Baseline_")])
        for tk in tfm_keys:
            tp_num = tk.replace("TFM_Transform_FollowUp_to_Baseline_", "").replace(".tfm", "")
            lk = f"PET_Lesions_{tp_num}.nii.gz"
            if lk in hf[tk]:
                mds = hf[tk][lk]
                mattrs = {a: mds.attrs[a] for a in ("spacing", "origin", "direction") if a in mds.attrs}
                marr = mds[:].astype(np.int16)
                if marr.shape[0] > 2 * edge_slices and (np.any(marr[:edge_slices] > 0) or np.any(marr[-edge_slices:] > 0)):
                    remove_edge_slice_lesions(marr, edge_slices=edge_slices)
                    del hf[tk][lk]
                    new_mds = hf[tk].create_dataset(lk, data=marr, compression="gzip", compression_opts=1)
                    for ak, av in mattrs.items():
                        new_mds.attrs[ak] = av
                if tk.endswith("_1.tfm"):
                    mr_gold_arr = marr

        if not trimmed:
            return {"trimmed": False, "removed_ids": []}

        # Reconcile CENTROIDS, organ_mapping, segment_names.json, matches.json, scene_hierarchy.json
        surviving_ids = sorted([int(x) for x in np.unique(les_arr) if x > 0])
        surviving_set = {str(x) for x in surviving_ids}

        m_grp = hf["_meta_"]
        om = json.loads(m_grp["organ_mapping"][()]) if "organ_mapping" in m_grp else {}
        sn = json.loads(m_grp["segment_names.json"][()]) if "segment_names.json" in m_grp else {}
        om0 = {k: v for k, v in om.get("0", {}).items() if str(k) in surviving_set}
        sn0 = {k: v for k, v in sn.get("0", {}).items() if str(k) in surviving_set}
        for lid in surviving_ids:
            if str(lid) not in om0:
                om0[str(lid)] = "Unknown"
            if str(lid) not in sn0:
                sn0[str(lid)] = f"Lesion {lid}"
        om["0"] = om0
        sn["0"] = sn0

        mr_surviving_ids = sorted([int(x) for x in np.unique(mr_gold_arr) if x > 0])
        mr_surv_set = {str(x) for x in mr_surviving_ids}
        for tp_k in list(om.keys()):
            if tp_k != "0" and isinstance(om[tp_k], dict):
                om[tp_k] = {k: v for k, v in om[tp_k].items() if str(k) in mr_surv_set}
        for tp_k in list(sn.keys()):
            if tp_k != "0" and isinstance(sn[tp_k], dict):
                sn[tp_k] = {k: v for k, v in sn[tp_k].items() if str(k) in mr_surv_set}
        sn1 = sn.get("1", {})

        if "CENTROIDS" in hf:
            del hf["CENTROIDS"]
        c_grp = hf.create_group("CENTROIDS")
        for lid in surviving_ids:
            zc, yc, xc = np.mean(np.argwhere(les_arr == lid), axis=0)
            c_grp.create_dataset(f"tp0_lid{lid}", data=np.array([round(xc) + 1, round(yc) + 1, round(zc) + 1], dtype=np.int32))
        for mlid in mr_surviving_ids:
            zc, yc, xc = np.mean(np.argwhere(mr_gold_arr == mlid), axis=0)
            c_grp.create_dataset(f"tp1_lid{mlid}", data=np.array([round(xc) + 1, round(yc) + 1, round(zc) + 1], dtype=np.int32))

        mr_best_parent = {}
        if mr_surviving_ids and surviving_ids:
            for mlid in mr_surviving_ids:
                mr_m = (mr_gold_arr == mlid)
                best_lid, best_cnt = None, 0
                for lid in surviving_ids:
                    ov = int(np.sum(mr_m & (les_arr == lid)))
                    if ov > best_cnt:
                        best_cnt, best_lid = ov, lid
                if best_lid is not None:
                    mr_best_parent.setdefault(best_lid, []).append(mlid)
        matches = []
        for lid in surviving_ids:
            children = [
                {
                    "name": "PET_Lesions_1",
                    "raw_lesion": f"Segment_{mlid}",
                    "node": "child",
                    "group_id": int(lid),
                    "lesion": sn1.get(str(mlid), "Prostate Primary Lesion"),
                }
                for mlid in mr_best_parent.get(lid, [])
            ]
            matches.append({
                "name": "PET_Lesions_0",
                "raw_lesion": f"Segment_{lid}",
                "node": "parent",
                "group_id": int(lid),
                "lesion": sn0.get(str(lid), f"Lesion {lid}"),
                "children": children,
            })

        if "scene_hierarchy.json" in m_grp:
            sh = json.loads(m_grp["scene_hierarchy.json"][()])
            for node in sh:
                if node.get("name") == "PET_Lesions_0":
                    node["segments"] = [sn0.get(str(i), f"Lesion {i}") for i in surviving_ids]
            del m_grp["scene_hierarchy.json"]
            m_grp.create_dataset("scene_hierarchy.json", data=json.dumps(sh))

        for mk, mv in (("organ_mapping", json.dumps(om)), ("segment_names.json", json.dumps(sn)), ("matches.json", json.dumps(matches))):
            if mk in m_grp:
                del m_grp[mk]
            m_grp.create_dataset(mk, data=mv)

    return {"trimmed": True, "removed_ids": removed_ids}


def extract_semantic_organs_qwen(report_text: str, max_anatomy_list: list, api_base: str, api_key: str, model: str = "qwen3.8-27b") -> dict:
    """Query Qwen via DIZ KI-Plattform API for max_anatomy structures with pathological mentions."""
    prompt = f"""Given the radiological dictation(s) and the exact list of anatomical structure names from `max_anatomy_labels`, return ONLY valid JSON with "reasoning" FIRST and "mentioned_max_anatomy" SECOND:
{{"reasoning": "...", "mentioned_max_anatomy": [...]}}

RULES:
1. First analyze every section of the dictation step-by-step inside "reasoning". Then populate "mentioned_max_anatomy" ONLY with the final confirmed labels from your reasoning — NEVER include labels you decided to exclude in your reasoning, and strictly preserve left/right laterality.
2. Include ALL structures from `max_anatomy_labels` (including both base and `nvseg_` variants when present in the list) where the dictation describes a pathological, suspicious, equivocal, tumor, lesion, fracture, or metastatic finding (e.g. prostate tumor/PSMA uptake, metastatic lymph nodes, osseous metastases, suspicious visceral/thyroid/lung lesions).
3. Pelvic bone mapping: ALWAYS map pelvic bone lesions in the `ilium`, `iliac crest`, `iliac wing`, `os ilii`, `ischium`, `pubis`, `pubic ramus`, or `acetabulum` to `hip_left`/`nvseg_left_hip` (left) or `hip_right`/`nvseg_right_hip` (right), because `hip_*` denotes the entire pelvic hip bone (os coxae).
4. German spine abbreviations: `BWK` = thoracic vertebra (`vertebrae_t*` / `nvseg_vertebrae_t*`), `LWK` = lumbar vertebra (`vertebrae_l*` / `nvseg_vertebrae_l*`), `SWK` = sacral (`vertebrae_s1` / `nvseg_vertebrae_s1` / `sacrum`), `HWK` = cervical vertebra (`vertebrae_c*`).
5. Do NOT include structures mentioned ONLY as unremarkable/negative/normal (e.g. "no evidence of suspicious uptake...", "no lymph node or bone metastases", "unremarkable") or physiological background uptake (salivary/parotid/submandibular/sublingual glands, lacrimal glands, liver/spleen/kidneys/ureters/urinary bladder/bowel background).
6. Do NOT include clearly benign non-neoplastic incidental findings (simple hepatic/renal cysts, simple lithiasis, benign PSMA-negative nodules, explicitly reactive/inflammatory fatty-hilar lymph nodes), and do NOT include blood vessels (`*_artery*`, `*_vena*`, `ts_iliac_*`) or muscles (`iliacus_*`, `psoas_*`) when cited solely as anatomical landmarks next to a lymph node or bone lesion.

max_anatomy_labels ({len(max_anatomy_list)}):
{json.dumps(max_anatomy_list)}

Radiological Dictation(s):
{report_text}
"""
    url = f"{api_base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last_err = None
    for attempt in range(4):
        use_no_think = attempt < 3
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 3000 if use_no_think else 5500,
        }
        if use_no_think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=240)
            r.raise_for_status()
            msg = r.json().get("choices", [{}])[0].get("message", {})
            raw = (msg.get("content") or "").strip()
            if not raw:
                raise ValueError("Empty content returned by LLM")
            clean = raw.replace("```json", "").replace("```", "").strip()
            m_obj = re.search(r"\{.*\}", clean, re.DOTALL)
            if m_obj:
                clean = m_obj.group(0)
            parsed = json.loads(clean)
            mentioned = [str(x).strip() for x in parsed.get("mentioned_max_anatomy", []) if str(x).strip()]
            # Keep only exact matches present in max_anatomy_list (plus normalized matches in max_anatomy_list)
            valid_set = set(max_anatomy_list)
            low_map = {x.lower(): x for x in max_anatomy_list}
            canonical = []
            for m in mentioned:
                if m in valid_set:
                    canonical.append(m)
                elif m.lower() in low_map:
                    canonical.append(low_map[m.lower()])
            return {
                "version": "1.1",
                "model": model,
                "mentioned_max_anatomy": sorted(set(canonical)),
                "reasoning": str(parsed.get("reasoning", "")).strip(),
            }
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Qwen semantic filter failed after retries: {last_err}")


def compute_and_store_semantic_filter(h5_path: str, api_base: str, api_key: str, model: str = "qwen3.8-27b", force: bool = False) -> dict:
    """Populate `_meta_/semantic_filter.json` in `h5_path`."""
    with h5py.File(h5_path, "r") as hf:
        if not force and "_meta_/semantic_filter.json" in hf:
            existing = json.loads(hf["_meta_/semantic_filter.json"][()])
            if existing.get("mentioned_max_anatomy") and existing.get("version") == "1.1" and existing.get("model") == model:
                return existing
        md = json.loads(hf["_meta_/metadata.json"][()]) if "_meta_/metadata.json" in hf else {}
        lbls = json.loads(hf["_meta_/max_anatomy_labels.json"][()]) if "_meta_/max_anatomy_labels.json" in hf else {}
        om = json.loads(hf["_meta_/organ_mapping"][()]) if "_meta_/organ_mapping" in hf else {}

    om_labels = {
        str(org).strip()
        for tp_om in om.values()
        if isinstance(tp_om, dict)
        for org in tp_om.values()
        if str(org).strip() and str(org).strip().lower() not in ("unknown", "background")
    }
    max_anat_list = sorted(
        {str(v).strip() for v in lbls.values() if str(v).strip()}
        | _get_canonical_labels()
        | om_labels
    )
    texts = []
    for k in sorted(md.keys()):
        v = md[k]
        if isinstance(v, dict):
            t = (v.get("EnglishDescription") or v.get("Description") or "").strip()
            if t:
                texts.append(f"[{k}]\n{t}")
    report_text = "\n\n".join(texts)

    if not report_text:
        sf_doc = {
            "version": "1.0",
            "model": model,
            "mentioned_max_anatomy": [],
            "allowed_lesion_ids_by_tp": {tp: [] for tp in om.keys()},
            "reasoning": "No radiological dictation available.",
        }
    else:
        sf_doc = extract_semantic_organs_qwen(report_text, max_anat_list, api_base, api_key, model=model)
        mentioned = sf_doc["mentioned_max_anatomy"]
        allowed_by_tp = {}
        for tp_k, tp_om in om.items():
            allowed = [
                int(lid_s)
                for lid_s, org in tp_om.items()
                if organ_matches_mentioned(str(org), mentioned)
            ]
            allowed_by_tp[str(tp_k)] = sorted(allowed)
        sf_doc["allowed_lesion_ids_by_tp"] = allowed_by_tp

    with h5py.File(h5_path, "r+") as hf:
        mg = hf["_meta_"]
        if "semantic_filter.json" in mg:
            del mg["semantic_filter.json"]
        mg.create_dataset("semantic_filter.json", data=json.dumps(sf_doc, ensure_ascii=False))
    return sf_doc


def process_one(pat: str, api_base: str, api_key: str, model: str, args, census: dict) -> str:
    h5_p = os.path.join(CASES_DIR, pat, "preprocessed_volumes.h5")
    edge_info = {"trimmed": False, "removed_ids": []}
    if not args.llm_only:
        edge_info = strip_edge_slices_in_h5(h5_p, edge_slices=2)
    sf_info = {}
    if not args.edge_only:
        sf_info = compute_and_store_semantic_filter(h5_p, api_base, api_key, model=model, force=args.force_llm)
    if args.sync_cache and pat in census:
        slbl = census[pat].get("subj_label", "")
        if slbl:
            dst_dir = os.path.join(CACHE_ROOT, slbl)
            os.makedirs(dst_dir, exist_ok=True)
            dst_h5 = os.path.join(dst_dir, f"{slbl}_preprocessed.h5")
            # Do not clobber a file actively modified within the last 20s by an open GUI session
            if not (os.path.isfile(dst_h5) and (time.time() - os.path.getmtime(dst_h5)) < 20.0):
                shutil.copy2(h5_p, dst_h5)
    n_ment = len(sf_info.get("mentioned_max_anatomy", []))
    n_keep0 = len(sf_info.get("allowed_lesion_ids_by_tp", {}).get("0", []))
    return f"{pat}: trimmed={edge_info['trimmed']} rm_edge={edge_info['removed_ids']} qwen_organs={n_ment} keep_tp0={n_keep0}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--edge-only", action="store_true")
    ap.add_argument("--llm-only", action="store_true")
    ap.add_argument("--force-llm", action="store_true")
    ap.add_argument("--sync-cache", action="store_true")
    args = ap.parse_args()

    api_base, api_key, model = get_llm_config()
    if not args.edge_only and not api_key:
        raise RuntimeError("DIZ_API_KEY not found for semantic filter LLM calls.")

    all_cases = sorted(
        [
            os.path.basename(p)
            for p in glob.glob(os.path.join(CASES_DIR, "Pat*"))
            if re.fullmatch(r"Pat\d+", os.path.basename(p))
            and os.path.isfile(os.path.join(p, "preprocessed_volumes.h5"))
        ],
        key=lambda s: int(re.search(r"\d+", s).group()),
    )
    targets = [c for c in all_cases if c in set(args.cases)] if args.cases else all_cases

    census = {}
    if args.sync_cache and os.path.isfile(CENSUS_PATH):
        with open(CENSUS_PATH, "r") as cf:
            census = json.load(cf)

    print(f"Processing {len(targets)} cases (workers={args.workers}, model={model}, edge_only={args.edge_only}, llm_only={args.llm_only})...", flush=True)
    t0 = time.time()
    ok, err = 0, 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(process_one, pat, api_base, api_key, model, args, census): pat for pat in targets}
        for i, fut in enumerate(as_completed(futs), 1):
            pat = futs[fut]
            try:
                msg = fut.result()
                ok += 1
                print(f"[{i}/{len(targets)}] {msg}", flush=True)
            except Exception as e:
                err += 1
                print(f"[{i}/{len(targets)}] FAIL {pat}: {e}", flush=True)
    print(f"Finished in {time.time()-t0:.1f}s: OK={ok}, ERR={err}")
    sys.exit(0 if err == 0 else 1)


if __name__ == "__main__":
    main()
