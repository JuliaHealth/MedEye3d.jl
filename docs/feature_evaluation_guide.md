# Clinical Workflow & Scientific Reporting: Feature Evaluation Guide

This document details all the new features, UI controls, and architectural fixes implemented since the last Git commit (`origin/quad_extension`). It serves as a step-by-step checklist to help you manually evaluate, test, and visualize the new capabilities in the MedEye3d application.

---

## 1. Clinical Case Profile 
**Purpose:** Binds the current session/patient to a specific clinical state, which is crucial for downstream deterministic reporting (e.g., E-PSMA).

* **Where to find it:** In the **Clinical Information & Indication** section.
* **Control:** A dropdown labeled `Profile:`.
* **How to evaluate:** 
  1. Open the dropdown and select `POST_RLT` (or `BCR`, `INITIAL_STAGING`).
  2. **What to observe:** The dropdown immediately reflects your choice. This selection is saved globally for the patient. 
  3. **Verification:** Click the `E-PSMA` button to generate a report. Look at the dark-grey header bar at the very top of the report window. You will observe that it dynamically reads: `Profile: POST_RLT`.

## 2. Key Image Bookmarking & Screenshot
**Purpose:** Allows radiologists to flag critical or highly representative lesions so they are visually prioritized in exported reports, and simultaneously triggers a snapshot of the current view.

* **Where to find it:** At the very top of the **Lesion Metadata** section.
* **Control:** A button initially labeled `☆ Key Image`.
* **How to evaluate:**
  1. Navigate to a lesion.
  2. Click the `☆ Key Image` button.
  3. **What to observe (UI):** The button instantly turns yellow, and the icon changes to a solid star (`★ Key Image`). Additionally, check your terminal logs for a message confirming a printscreen was saved.
  4. **Verification:** Generate the E-PSMA report. Look at "Synoptic Table 2" (Section 5). The lesion you bookmarked will be highlighted with a gold `★` next to its anatomical location.

## 3. Scientific Workflow (Action Panel)
**Purpose:** Provides a rapid, one-click triage interface for reviewing AI-generated lesion candidates, updating their clinical state instantly.

* **Where to find it:** At the very top of the **Lesion Metadata** section (just above Lesion Navigation).
* **Controls:** 
  * Four action buttons: `Accept` (Green), `Reject` (Red), `Correct` (Yellow), and `Export` (Grey).
  * A `State:` dropdown (e.g., `UNREVIEWED`, `ACCEPTED`, `REJECTED`, `CORRECTED`, `UNCERTAIN`, `NEW`, `RESOLVED`).
* **How to evaluate:**
  1. Note the current value in the `State:` dropdown (default is usually `UNREVIEWED`).
  2. Click the **Accept** (or Reject/Correct) button.
  3. **What to observe:** The `State:` dropdown instantly snaps to `ACCEPTED`. 
  4. **Background observation:** Check your terminal output. You will see a log indicating that an autosave was triggered, safely persisting your review state to the database.
  5. **Exporting:** Click the `Export` button. Check your home directory (`~/research_export.csv`). Open the file to verify it dumped a clean CSV containing the Lesion ID, Display Name, Observation State, Anatomy, and SUV metrics for your entire session.

## 4. Sweep Mode (Unmatched AI Candidates)
**Purpose:** Automatically identifies lesions that appear in follow-up scans but lack a corresponding baseline (TP0) segmentation. These are flagged for manual review to catch false positives or track newly emergent metastases.

* **Where to find it:** The **Lesion Navigation** section.
* **Control:** The main lesion selection dropdown (between the `<< Prev` and `Next >>` buttons).
* **How to evaluate:**
  1. Click to expand the lesion dropdown list.
  2. Scroll towards the bottom of the list.
  3. **What to observe:** You will see entries explicitly tagged with `[SWEEP]`, e.g., `1001: [SWEEP] Unmatched Candidate (Grp 1)`. 
  4. **Verification:** Select one of these `[SWEEP]` candidates. The viewport will snap to the lesion. You can then use the Scientific Workflow buttons (`Reject` or `Accept`) to triage this unmatched candidate.

## 5. Deterministic E-PSMA Structured Reporting
**Purpose:** Fully automates the translation of your metadata inputs into a standardized, deterministic oncological report.

* **Where to find it:** Main viewport action bar -> `E-PSMA` button.
* **How to evaluate:**
  1. Ensure you have set the Profile, toggled a Key Image, and annotated a few lesions.
  2. Generate the report and scroll through it.
  3. **What to observe:**
     * **Header:** Accurately reflects the `Profile:` you set.
     * **Synoptic Tables:** Cleanly formats all lesions, highlighting the `★ Key Images` and automatically appending `[NEW]` tags to lesions that didn't exist in the baseline scan.
     * **Section 6 (Overall Conclusion):** The deterministic engine evaluates all rows and calculates the final staging (e.g., `miT0 miN0 miM1b`) and applies the RECIP criteria (e.g., `PARTIAL METABOLIC RESPONSE`).

---

## 6. Under-the-Hood Architectural Fixes

While invisible in the UI, two major structural fixes were implemented to guarantee data integrity and application stability:

### A. Non-Destructive Expert Masks
* **The Problem:** Previously, manually touching up a mask via the paint tool overwrote the original AI prediction in the HDF5 file.
* **The Fix:** Mask autosaves now write strictly to `*_expert` datasets (e.g., `mask_expert`).
* **How to verify:** 
  1. Use the paint brush to modify a lesion mask. 
  2. Wait for the autosave to trigger.
  3. **What to observe:** In the terminal, look for the log: `[AUTOSAVE-MASK] Saved mask for TP X to BASELINE/mask_name_expert`. The original AI prediction is completely untouched.

### B. Vulkan Command Buffer Stability
* **The Problem:** Spawning the secondary independent GLFW window for the E-PSMA report occasionally caused silent crashes or rendering pipeline hangs due to a namespace collision (`allocate_command_buffers`) in Vulkan.jl.
* **The Fix:** Hard-qualified `Vulkan.allocate_command_buffers` inside `VulkanContext.jl` and `VulkanStaging.jl`.
* **How to verify:** You can now rapidly open, close, and refresh the E-PSMA report window without any fear of the OpenGL/Vulkan interop crashing the render loop.
