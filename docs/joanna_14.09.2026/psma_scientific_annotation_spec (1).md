---
title: "PSMA PET/CT Scientific Annotation Mode"
subtitle: "Software Requirements and UX Workflow Specification"
author: "Technical handoff specification"
date: "2026-09-13"
---

# 1. Document purpose

This document specifies the **Scientific Annotation Mode** of a PSMA PET/CT lesion annotation platform. It is intended as an implementation handoff for a professional software developer who will modify the existing application and may use an LLM coding assistant (for example Gemini) to refactor or rewrite parts of the code.

The specification consolidates the current prototype, the three-monitor radiology workstation configuration, the proposed lesion-by-lesion longitudinal workflow, keyboard and mouse interactions, data-state transitions, segmentation editing, promptable segmentation, registration quality control, autosave/versioning, and testable acceptance criteria.

The document deliberately focuses on the **scientific lesion annotation workflow**. The separate **clinical reporting workflow** (history/dictation, report generation, miTNM, E-PSMA, PSMA-RADS, clinical recommendations, Word export) is not part of the present implementation scope, although the architecture should allow both modes to share the same image viewer, lesion database, segmentation engine, registration data, and quantitative measurement services.

## 1.1 Primary objective

Create an expert annotation workflow in which a radiologist/nuclear medicine physician can efficiently:

1. review pre-segmented PSMA-positive lesions,
2. accept, reject, or correct each lesion,
3. add missed lesions using promptable segmentation,
4. follow the same lesion across up to approximately 10-15 registered examinations,
5. document disappearance/resolution,
6. handle unreliable registration without corrupting the global transform,
7. preserve the original AI result and all expert edits for research,
8. create an auditable, versioned lesion-level longitudinal dataset.

## 1.2 Primary unit of work

The primary unit is **not the examination and not the report**. The primary unit is the **lesion track**.

A lesion track has a stable identity across time:

`LesionTrack L007 -> TP0 -> TP1 -> TP2 -> ... -> TP14`

Each timepoint contains a lesion observation that can be present, corrected, resolved, uncertain, not evaluable, newly detected, or manually matched.

# 2. Product principles

The Scientific Annotation Mode should follow these principles.

## 2.1 Images on diagnostic monitors; controls on support monitor

The two high-resolution diagnostic portrait monitors are scarce diagnostic display space and should show almost exclusively image content. Buttons, long tables, metadata entry, model selection, and workflow status should primarily live on the smaller non-diagnostic support monitor.

## 2.2 One decision should normally advance the workflow

The most frequent pathway should be:

`inspect -> A (Accept) -> autosave -> load next timepoint of same lesion`

There should be no additional Save, Confirm, Next, Load, or Center click in the normal case.

## 2.3 Preserve spatial context

When the user changes lesion or timepoint, the viewer should automatically:

- center on the lesion or registered target location,
- synchronize axial/coronal/sagittal coordinates,
- preserve a consistent physical field of view where practical,
- preserve relevant PET/CT display presets,
- load the corresponding registered comparison on the second diagnostic monitor.

## 2.4 AI result must remain immutable

The original AI segmentation is research data and must never be overwritten. Expert corrections are stored as new mask versions.

## 2.5 Longitudinal review should be lesion-centric

After an anchor examination is cleaned, the preferred review order is:

`one lesion -> all available timepoints -> next lesion`

This reduces cognitive load compared with completing all lesions independently at every examination.

## 2.6 New lesions require a second sweep

A purely baseline-derived lesion list could miss lesions that appear later. Therefore, after track-wise review, the software must run an **unmatched-candidate sweep** for each follow-up timepoint.

## 2.7 Registration failure must be explicit

Poor registration must never silently alter lesion identity or create an apparent biological change. Registration quality, local offsets, and manual matches are separate auditable data fields.

# 3. Physical workstation and monitor responsibilities

The workstation has:

- **M1:** high-resolution diagnostic portrait monitor - current lesion/current timepoint.
- **M2:** high-resolution diagnostic portrait monitor - longitudinal comparison and registration context.
- **M3:** smaller non-diagnostic monitor, positioned to the left - workflow control, lesion queue, annotation actions, segmentation tools, metadata/status.

## 3.1 M1 - Diagnostic Monitor 1: Current Lesion / Current Timepoint

### Purpose

Answer the question: **What is this lesion now, and is its segmentation correct?**

### Default layout

- Large axial PET/CT fusion image occupying approximately the upper 55-60%.
- Coronal image lower left.
- Sagittal image lower right.
- Optional MIP or PET-only view can temporarily replace one secondary panel.
- Minimal text overlays only: lesion ID, timepoint, SUVmax, volume, slice index, HU/SUV under cursor, scale bar.

### Content that should not be permanently shown on M1

- Lesion list.
- Large Accept/Reject buttons.
- Long metadata forms.
- Longitudinal tables.
- Report text.
- RadLex lists.
- Clinical history.
- Large plots.

### Automatic behavior on lesion/timepoint change

1. Center current lesion or predicted registered target location.
2. Synchronize all orthogonal planes.
3. Show final/working mask contour.
4. Use a consistent lesion-centered FOV.
5. Prefetch adjacent slices and next timepoint where possible.

## 3.2 M2 - Diagnostic Monitor 2: Longitudinal Comparison / Registration

### Purpose

Answer the question: **Is this the same lesion across time, and what changed?**

### Default layout: Compare Mode

- Left: current timepoint.
- Right: reference timepoint, normally previous timepoint.
- Both views synchronized to the same registered anatomical location.
- Same orientation, zoom/FOV, and preferably the same PET display scale.
- Mask can be independently toggled.

### Reference options

The reference selector should support at least:

- Previous timepoint (default).
- Baseline/anchor timepoint.
- User-selected timepoint.

### Secondary area

A lesion-centered filmstrip may show TP0-TP14. Each thumbnail should show the same lesion-centered crop, not merely a whole-body thumbnail. Clicking a thumbnail changes the reference or current timepoint depending on mode.

### Temporary modes

1. **Compare** - default side-by-side.
2. **Flicker** - rapid current/reference alternation.
3. **Overlay** - registered overlay with opacity control.
4. **Registration Review** - landmarks, contours, local offset/manual matching tools.
5. **Whole-body context** - optional MIP/coronal context, not the default.

### Information intentionally hidden by default

Longitudinal SUV/volume trend plots should not dominate the initial segmentation decision because they can bias the reviewer. Metrics may be opened on demand after the image-based decision.

## 3.3 M3 - Support Monitor: Workflow Console

### Purpose

Answer the question: **What action should the user take, what is the current status, and what remains to be reviewed?**

### Persistent header

Show:

- pseudonymized patient/case ID,
- current lesion ID and label,
- current timepoint/total timepoints,
- current review state,
- registration QC state,
- autosave state,
- progress summary.

Example:

`Patient 014 | L07 Left ilium | TP03/12 | Unreviewed | Registration: Good | 42% complete`

### Primary action buttons

The following must remain immediately accessible:

- Accept.
- Reject.
- Edit/Correct.
- New Lesion.
- Prompt Segment.
- Uncertain.
- Resolved/Not Visible.
- Registration QC / Flag Registration.

### Lesion queue

Show a compact list with:

- lesion ID,
- anatomical label,
- lesion type,
- optional SUVmax,
- current observation status,
- track completion status.

Filters:

- All.
- Unreviewed.
- Accepted.
- Corrected.
- Rejected.
- Uncertain.
- New.
- Resolved.

### Segmentation tools

Include:

- View.
- Paint.
- Erase.
- Brush size.
- mask opacity.
- AI/prompt model selector (three available models).
- prompt mode: point, box, scribble if supported.
- Apply/Cancel while editing.

### Compact lesion information

Only the information necessary for annotation should be visible by default:

- anatomy,
- lesion type,
- volume,
- equivalent/longest diameter,
- SUVmax,
- SUVmean,
- optional SUVpeak,
- model confidence if available,
- CT correlate flag if required for research.

Long clinical reporting fields stay hidden in Scientific Mode.

# 4. Scientific workflow architecture

The recommended end-to-end workflow is a hybrid **Anchor -> Track -> Sweep -> QC** strategy.

## 4.1 Phase A - Preprocessing / case initialization

Before interactive review, the application should have access to:

1. all examinations/timepoints for the case,
2. DICOM-derived PET/CT geometry and quantitative metadata,
3. pre-segmentation output for all available timepoints,
4. anatomical segmentation if available,
5. inter-timepoint registration transforms,
6. optionally candidate matching suggestions across timepoints.

Where possible, preprocessing should be performed before the user opens the case so that annotation does not wait for avoidable computation.

## 4.2 Phase B - Anchor examination cleanup

The anchor examination is normally baseline/TP0 unless the protocol specifies another timepoint.

The user reviews every candidate and assigns one of:

- Accepted.
- Corrected.
- Rejected.
- Uncertain.
- New manually/prompt-created lesion.

At completion, each true lesion receives a stable LesionTrack ID.

## 4.3 Phase C - Lesion-track longitudinal validation

For each stable lesion track:

1. load lesion at TP0,
2. step through TP1, TP2, ... TPn,
3. compare current with previous/baseline on M2,
4. accept/correct/match/resolve/flag uncertainty,
5. autosave,
6. automatically move to next timepoint of the same lesion.

Only after all timepoints for that lesion are reviewed should the workflow move to the next lesion track.

## 4.4 Phase D - Unmatched/new-lesion sweep

For each non-anchor timepoint, show AI candidates that are not linked to an existing lesion track.

Each unmatched candidate is classified as:

- new true lesion -> create new LesionTrack,
- existing lesion missed by automatic matching -> link to existing LesionTrack,
- false positive -> reject,
- uncertain -> flag for review.

This prevents late-appearing lesions from being missed.

## 4.5 Phase E - Completion QC and lock

Before marking a case complete, the system checks:

- no unreviewed candidate remains,
- no unresolved uncertain item remains unless explicitly accepted as uncertain,
- each lesion track has a defined state for all required timepoints,
- all registration problems have a QC outcome,
- all expert masks are saved/versioned,
- all new lesions are linked into a track,
- all rejected candidates retain their original AI mask and reason if collected.

The user can then lock the case/annotation version for analysis while retaining the ability to create a later revision.

# 5. Interaction model and shortcuts

The final implementation should use one consistent shortcut map. Shortcuts must never perform destructive operations without an immediately reversible audit event.

## 5.1 Existing DICOM viewer mouse interactions to preserve

| Interaction | Function |
|---|---|
| Mouse wheel | Next/previous slice in active viewport |
| Ctrl + mouse wheel | Zoom |
| Shift + mouse wheel | Zoom (preserve current prototype behavior unless intentionally reassigned after user testing) |
| Right click | Localize the clicked anatomical point in the other orthogonal views |
| Right-button drag | Pan/move image |
| Double left click | Recenter/set the orthogonal cross-sections from the clicked plane/position; the exact existing behavior should be preserved and documented in code |

Note: A future usability test may reassign `Shift + wheel` to PET windowing, but the current implementation should not silently change an established interaction.

## 5.2 Core lesion decision shortcuts

| Shortcut | Action | Auto-advance |
|---|---|---|
| A | Accept current lesion observation/mask | Yes |
| R | Reject AI candidate as false positive | Yes |
| E | Enter Edit/Correct mode | No |
| N | Create a new lesion record | No |
| P | Enter prompt segmentation mode using default prompt model | No |
| U | Mark uncertain / needs review | Configurable; default Yes to next observation |
| X | Mark lesion Resolved / Not Visible at current TP | Yes |
| Shift + R | Flag registration as Questionable and enter Registration Review Mode | No |
| Enter | Apply current edit/prompt result/confirmation | Depends on context; after successful finalization, Yes |
| Esc | Cancel current transient tool/mode without committing | No |

## 5.3 Navigation shortcuts

Avoid overloading the same key for both timepoint and brush control.

Recommended final mapping:

| Shortcut | Action |
|---|---|
| Up / Down | Previous / next lesion in queue |
| Ctrl + Up / Ctrl + Down | Previous / next unreviewed lesion |
| Alt + Left / Alt + Right | Previous / next timepoint of current lesion |
| Home | First timepoint of current lesion |
| End | Last timepoint of current lesion |
| B | Use baseline as comparison reference on M2 |
| V | Use previous timepoint as comparison reference on M2 |
| C | Center current lesion in all M1 views |
| F | Toggle flicker mode on M2 |
| O | Toggle overlay mode on M2 |
| Q (hold) | Temporarily hide current lesion mask while held |

## 5.4 Editing shortcuts

While Edit mode is active:

| Interaction | Function |
|---|---|
| Left drag | Paint/add to mask |
| Right drag | Erase from mask; if this conflicts with pan, Edit mode must explicitly remap and visibly indicate it |
| [ | Decrease brush size |
| ] | Increase brush size |
| Q (hold) | Temporarily hide mask |
| Enter | Apply correction |
| Esc | Cancel edit and return to prior mask version |

Because right-drag is used for pan in View mode and erase in Edit mode, the active tool/mode must be visually unambiguous.

## 5.5 Prompt segmentation interactions

Recommended interaction:

| Interaction | Function |
|---|---|
| P | Activate default prompt segmentation model |
| Left click | Positive point |
| Ctrl + left click | Negative point |
| Shift + drag | Bounding box prompt |
| Enter | Run/apply segmentation when prompts are ready |
| Shift + P | Open model chooser for the three available prompt models |
| Esc | Cancel prompt mode |

A default prompt model should be configurable so the user does not select a model for every lesion.

# 6. Lesion and observation states

Separate the concepts of **candidate decision**, **longitudinal presence**, **registration quality**, and **match method**. Do not encode all meanings into one status string.

## 6.1 Candidate decision state

Recommended enum:

- `UNREVIEWED`
- `ACCEPTED`
- `CORRECTED`
- `REJECTED`
- `UNCERTAIN`
- `NEW_EXPERT_CREATED`

## 6.2 Longitudinal presence state

Recommended enum:

- `PRESENT`
- `RESOLVED_NOT_VISIBLE`
- `NOT_EVALUABLE`
- `UNKNOWN`

## 6.3 Match method

Recommended enum:

- `ANCHOR`
- `AUTO_MATCH`
- `MANUAL_MATCH`
- `NEW_AT_TIMEPOINT`
- `RECONNECTED_TRACK`
- `UNMATCHED`

## 6.4 Registration QC state

Recommended enum:

- `GOOD`
- `QUESTIONABLE`
- `LOCAL_ADJUSTED`
- `POOR_MANUAL_MATCH`
- `FAILED_NOT_EVALUABLE`

## 6.5 Segmentation origin

Recommended enum:

- `AI_PRESEGMENTATION`
- `PROMPT_MODEL_1`
- `PROMPT_MODEL_2`
- `PROMPT_MODEL_3`
- `MANUAL`
- `EXPERT_CORRECTION_OF_AI`
- `EXPERT_CORRECTION_OF_PROMPT`

# 7. Scenario workflows

The following workflows should be implemented as explicit UI/state-machine paths.

# 7.1 Scenario 1 - Normal lesion: correct pre-segmentation, lesion present

## Entry condition

- Lesion track exists.
- Current timepoint observation is unreviewed.
- AI mask is available.
- Registration to reference is sufficiently plausible.

## Step-by-step UI behavior

1. **Automatic load**
   - M1 centers the lesion in axial/coronal/sagittal views.
   - M2 shows current TP versus previous TP, synchronized to the lesion.
   - M3 highlights the current lesion and shows `UNREVIEWED`.

2. **Visual check on M1**
   - Reviewer assesses whether the mask corresponds to the lesion.
   - Optional `Q` temporarily hides mask.

3. **Longitudinal check on M2**
   - Reviewer confirms that anatomy and lesion identity are plausible across current/reference TP.

4. **Decision**
   - User presses `A` or clicks Accept on M3.

5. **Data update**
   - candidate decision -> `ACCEPTED`.
   - presence -> `PRESENT`.
   - accepted mask points to immutable AI mask or a copied finalized expert version according to storage design.
   - audit event written.

6. **Automatic next step**
   - autosave.
   - load next timepoint of the same lesion.
   - M1 updates to next TP.
   - M2 updates to next TP versus previous TP.
   - M3 returns observation state to `UNREVIEWED` for the new TP.

## Required click count in ideal case

One decision keystroke per observation: `A`.

# 7.2 Scenario 2 - Incorrect segmentation: lesion detected but mask is wrong

## Entry condition

- Lesion is real and correctly matched.
- AI segmentation boundary is inaccurate.

## Step-by-step UI behavior

1. M1/M2 load as in Scenario 1.
2. User presses `E` or clicks Edit/Correct on M3.
3. M3 switches to editing controls: Paint, Erase, Brush Size, Apply, Cancel.
4. M1 enters a clearly indicated Edit Mode.
5. User edits mask in M1:
   - left drag add,
   - right drag erase,
   - wheel changes slice,
   - `[`/`]` changes brush size,
   - `Q` hides mask temporarily.
6. Orthogonal views update the working mask in real time.
7. User reviews final contour in all planes and optionally against M2.
8. User presses `Enter` or clicks Apply.
9. System:
   - preserves original AI mask,
   - creates a new expert mask version,
   - sets decision -> `CORRECTED`,
   - presence -> `PRESENT`,
   - records correction source and edit metadata,
   - recomputes quantitative lesion metrics,
   - writes audit event,
   - autosaves,
   - advances to next timepoint of same lesion.

## Derived research metrics

Where feasible, automatically calculate:

- Dice/IoU between AI and expert mask,
- absolute and relative volume change,
- surface/centroid displacement,
- optional automatic correction category (`none`, `minor`, `major`) based on a configurable research rule.

Do not require the physician to manually enter these metrics.

# 7.3 Scenario 3 - Missing lesion: true lesion not segmented by AI

## Entry condition

- Reviewer sees a true lesion without an AI candidate/mask.

## Step-by-step UI behavior

1. Reviewer identifies suspicious uptake on M1.
2. M2 may be used to assess whether it was already visible at prior timepoints.
3. User presses `N` / clicks New Lesion.
4. System creates a provisional lesion object and activates lesion creation mode.
5. User presses `P` or selects Prompt Segment.
6. Default prompt model is activated; M3 shows point/box/scribble controls.
7. User supplies prompt on M1:
   - positive point,
   - optional negative point(s),
   - or bounding box.
8. User presses `Enter`; model returns a provisional mask.
9. New mask appears on M1 in all orthogonal views.
10. Reviewer:
    - presses `A` if correct, or
    - `E` to correct and then `Enter`.
11. System assigns stable new LesionTrack ID if this is truly a new track.
12. Match method is set to `NEW_AT_TIMEPOINT` unless retrospectively linked to an existing track.
13. System stores:
    - creation timepoint,
    - segmentation origin/model/version,
    - expert mask,
    - anatomy/lesion type if available,
    - audit history.
14. System optionally prompts or automatically searches prior TPs for a possible retrospective match.
15. Autosave and continue.

## Important behavior

The new lesion must become available for longitudinal review across all subsequent timepoints. It must not remain a one-off mask tied only to the current study.

# 7.4 Scenario 4 - Resolved/disappeared lesion

## Entry condition

- Lesion track was present at prior TP(s).
- At current registered location, no convincing residual lesion is visible.

## Step-by-step UI behavior

1. M1 loads current TP centered at the predicted registered location.
2. M2 shows current versus previous TP.
3. Reviewer confirms that prior lesion is visible on reference but not convincingly visible on current TP.
4. User presses `X` / clicks Resolved / Not Visible.
5. System sets:
   - presence -> `RESOLVED_NOT_VISIBLE`,
   - observation reviewed -> true.
6. The LesionTrack itself is **not deleted**.
7. No new mask is required at the current TP unless a protocol explicitly requires a residual empty/zero representation.
8. Audit event is stored.
9. Autosave.
10. System loads next TP of the same lesion.
11. If lesion becomes visible again at a later TP, the reviewer can reconnect it to the same track rather than creating a new ID.

## Required behavior on recurrence/reappearance

The UI must support `RECONNECTED_TRACK` or manual match so that a later visible lesion can be linked to the prior lesion identity.

# 7.5 Scenario 5 - Poor/unreliable registration

## Entry condition

- Current and reference anatomy do not align sufficiently for safe lesion matching.

## Step-by-step UI behavior

1. M1 shows current lesion/current expected region.
2. M2 side-by-side view reveals anatomical offset or mismatch.
3. User presses `Shift + R` / clicks Registration QC.
4. M3 sets registration state -> `QUESTIONABLE`.
5. M2 enters Registration Review Mode.
6. Registration Review Mode provides:
   - synchronized crosshair,
   - contour overlay,
   - landmark inspection,
   - local translation/offset tool,
   - optional local rotation if technically supported,
   - manual lesion match action,
   - Not Evaluable action.
7. Reviewer checks one or more stable anatomical landmarks.
8. One of three outcomes is chosen:

### Outcome A - local adjustment resolves mismatch

- Apply a **local lesion-level offset** for comparison.
- Do not overwrite the global registration transform.
- registration QC -> `LOCAL_ADJUSTED`.
- store local transform/offset separately.
- continue normal lesion decision.

### Outcome B - global/local registration remains poor but lesion identity is manually clear

- Reviewer manually selects/matches the corresponding lesion on the reference TP.
- match method -> `MANUAL_MATCH`.
- registration QC -> `POOR_MANUAL_MATCH`.
- store manual match event and coordinates.
- continue normal lesion decision.

### Outcome C - no reliable match possible

- Reviewer selects `Not Evaluable` / `U` according to UI context.
- presence or matchability -> `NOT_EVALUABLE`.
- registration QC -> `FAILED_NOT_EVALUABLE`.
- do not infer biological progression/regression from this comparison.
- save and continue.

# 7.6 Scenario 6 - False-positive AI candidate

## Entry condition

- AI produced a candidate/mask but reviewer determines it is not a true target lesion.

## Step-by-step UI behavior

1. Candidate is loaded on M1 and comparison is available on M2.
2. Reviewer presses `R` / clicks Reject.
3. candidate decision -> `REJECTED`.
4. AI mask is preserved in original form.
5. Optional one-click rejection reason may appear:
   - physiological uptake,
   - urinary activity,
   - benign structure,
   - artefact,
   - duplicate candidate,
   - other.
6. Rejection reason should be optional unless required by a study protocol.
7. Audit event is written.
8. Autosave and advance.

# 8. State-machine requirements

A simplified UI state machine is recommended.

## 8.1 Primary states

- `CASE_LOADING`
- `LESION_REVIEW`
- `EDIT_MASK`
- `NEW_LESION`
- `PROMPT_SEGMENTATION`
- `REGISTRATION_REVIEW`
- `MANUAL_MATCH`
- `SAVING`
- `CASE_QC`
- `CASE_COMPLETE`

## 8.2 Core transitions

`CASE_LOADING -> LESION_REVIEW`

From `LESION_REVIEW`:

- `A -> SAVING -> LESION_REVIEW(next observation)`
- `R -> SAVING -> LESION_REVIEW(next observation)`
- `E -> EDIT_MASK`
- `N -> NEW_LESION`
- `P -> PROMPT_SEGMENTATION`
- `X -> SAVING -> LESION_REVIEW(next observation)`
- `U -> SAVING or flagged queue according to configuration`
- `Shift+R -> REGISTRATION_REVIEW`

From `EDIT_MASK`:

- `Enter -> SAVING -> LESION_REVIEW(next observation)`
- `Esc -> LESION_REVIEW(current observation)`

From `PROMPT_SEGMENTATION`:

- `Enter -> provisional mask -> LESION_REVIEW or EDIT_MASK`
- `Esc -> NEW_LESION or LESION_REVIEW`

From `REGISTRATION_REVIEW`:

- local adjustment accepted -> `LESION_REVIEW`
- manual match -> `MANUAL_MATCH -> LESION_REVIEW`
- not evaluable -> `SAVING -> LESION_REVIEW(next observation)`
- cancel -> `LESION_REVIEW(current observation)`

## 8.3 Guard conditions

Examples:

- `Accept` requires a current candidate or an accepted expert-created mask.
- `Corrected` requires a stored expert mask version.
- `Resolved` is valid only for an existing LesionTrack, not a new unmatched candidate.
- `Manual Match` requires both current and target observation/coordinates.
- case completion requires no unresolved mandatory states.

# 9. Proposed persistent data model

The exact implementation language/database can vary. The following logical entities should exist.

## 9.1 Case / PatientRecord

Suggested fields:

- `case_id` (pseudonymized internal ID)
- study/project ID
- optional cohort/site ID
- number of timepoints
- annotation status
- current locked annotation version

Avoid storing directly identifying patient information in the research annotation layer unless explicitly required by the existing architecture and governance.

## 9.2 ExaminationTimepoint

Suggested fields:

- `timepoint_id`
- `case_id`
- ordinal index (`TP0`, `TP1`, ...)
- acquisition date/time
- PET series ID
- CT series ID
- tracer
- injected activity if available
- uptake time if available
- voxel geometry / image metadata reference
- registration transform(s) to anchor and/or adjacent TPs
- preprocessing status

## 9.3 LesionTrack

Suggested fields:

- `lesion_track_id`
- `case_id`
- stable human-readable label
- lesion class (prostate, lymph node, bone, organ, other)
- canonical anatomy label
- anchor timepoint
- creation method
- active/resolved history derived from observations
- optional ontology identifiers

## 9.4 LesionObservation

One record per lesion track per timepoint where required.

Suggested fields:

- `observation_id`
- `lesion_track_id`
- `timepoint_id`
- candidate decision state
- presence state
- match method
- registration QC state
- active/final mask version ID
- current centroid in image/physical coordinates
- current anatomy label
- SUVmax/SUVmean/SUVpeak
- volume
- longest/equivalent diameter
- confidence/uncertainty if used
- reviewer comment
- reviewed flag
- reviewed timestamp

## 9.5 SegmentationVersion

Suggested fields:

- `segmentation_version_id`
- `observation_id`
- parent version ID
- source/origin enum
- model name
- model version/hash
- prompt data if applicable
- immutable mask file/object reference
- creation timestamp
- creator (AI/user)
- final/expert-approved flag
- derived metrics

### Rule

Never update mask bytes in place. Create a new SegmentationVersion and repoint `active/final mask` to the selected version.

## 9.6 RegistrationRecord

Suggested fields:

- source timepoint
- target timepoint
- global transform reference
- algorithm/version
- QC state
- optional lesion-level local transform/offset
- local adjustment creator
- manual landmark coordinates if recorded
- timestamp

Global transform and local lesion-level adjustment must remain distinguishable.

## 9.7 AuditEvent

Suggested fields:

- `event_id`
- case ID
- lesion track ID if relevant
- observation ID if relevant
- user/reviewer ID
- timestamp
- event type
- previous state
- new state
- segmentation version IDs before/after
- tool/model used
- optional free-text reason

Examples:

- lesion accepted,
- lesion rejected,
- edit started,
- mask correction applied,
- prompt segmentation run,
- new lesion created,
- registration flagged,
- local offset applied,
- manual match confirmed,
- resolved marked,
- undo/redo.

# 10. Autosave and recovery

## 10.1 Autosave rule

Every committed lesion decision should autosave immediately.

Committed actions include:

- Accept.
- Reject.
- Apply correction.
- Finalize new lesion.
- Resolved.
- Uncertain.
- registration QC outcome.
- manual lesion match.

## 10.2 UI behavior

M3 should show non-blocking status:

- `Saving...`
- `Saved` with timestamp or checkmark.
- `Save failed - retrying`.
- `Offline/local save` if applicable.

Do not interrupt the user with modal success dialogs.

## 10.3 Failure recovery

- Maintain a local pending-event queue if backend write temporarily fails.
- Do not auto-advance irreversibly until the action is safely persisted locally or remotely according to the deployment architecture.
- On restart, restore the last patient/lesion/timepoint and pending edits where technically feasible.

# 11. Longitudinal matching behavior

## 11.1 Automatic matching

Automatic candidate matching may use registration, centroid distance, anatomy, lesion masks, or model output. However, automatic matching is a proposal, not ground truth.

The reviewer must be able to override it.

## 11.2 Same lesion versus new lesion

The UI must make the distinction explicit:

- **Link to existing track**.
- **Create new track**.

A new segmentation at TP5 should not automatically imply a new LesionTrack if the lesion existed earlier.

## 11.3 Split and merge cases

Even if not in the first MVP, the data model should not prevent:

- one prior lesion splitting into multiple current lesions,
- multiple prior lesions merging into one current region.

Possible future relation table:

`LesionTrackRelation(parent_track_id, child_track_id, relation_type, timepoint_id)`

Do not encode split/merge by silently reusing IDs.

# 12. Quantitative measurements

After a final mask is accepted/corrected, calculate and persist the agreed quantitative features.

Minimum recommended:

- volume in cc/mL,
- equivalent or longest diameter,
- SUVmax,
- SUVmean.

Optional if already technically available and validated:

- SUVpeak,
- lesion centroid,
- lesion-to-reference ratios,
- radiomics features,
- surface area.

Measurements must be recomputed after expert mask correction. The software should record the mask version from which each measurement set was derived.

# 13. Bias-aware presentation

For scientific annotation, avoid showing downstream interpretation that could bias basic lesion/mask ground-truth creation unless the study protocol explicitly requires it.

Therefore, during primary mask review, keep the following hidden or secondary:

- response category,
- longitudinal SUV trend plot,
- final staging,
- clinical treatment response labels,
- generated report text.

The reviewer may open them later for dedicated tasks.

# 14. UI feedback and visual language

A consistent status color/icon scheme should be used, but status must never be communicated by color alone.

Recommended examples:

- Accepted: check icon + text.
- Corrected: edit/check icon + text.
- Rejected: X icon + text.
- Unreviewed: open circle + text.
- Uncertain: question mark + text.
- Resolved: hollow/strike icon + text.
- Registration questionable: warning icon + text.

## 14.1 Mode indication

M1/M2 should clearly display active mode:

- VIEW MODE
- EDIT MODE
- PROMPT MODE
- REGISTRATION REVIEW

This is essential because mouse behavior can change by mode.

# 15. Performance and usability targets

These are UX targets, not hard promises until benchmarked on the target hardware.

- UI response to button/shortcut: perceived immediate, ideally <100 ms.
- Switching to prefetched next lesion/timepoint: ideally <1 s.
- Adjacent timepoints and next lesion should be prefetched in the background.
- No modal dialog for routine decisions.
- Normal accepted observation should require one decision keystroke after visual review.
- Image viewport should retain diagnostic resolution and not be reduced by unnecessary UI chrome.
- Application should remain usable with 10-15 timepoints and dozens of lesion tracks.

# 16. Configuration and user preferences

Persist per-user or per-project settings where appropriate:

- default prompt model,
- mask opacity,
- mask/surface/anatomy visibility,
- preferred PET scale/window,
- preferred CT preset,
- current/reference comparison default (previous vs baseline),
- automatic advance on Accept/Reject/Resolved,
- default lesion-centered FOV,
- shortcut customization if later supported.

# 17. Scientific mode versus clinical mode

The codebase should separate **shared domain/services** from **mode-specific UI**.

## Shared components

- DICOM/PET/CT viewer.
- image loading/cache.
- image registration data.
- lesion/track database.
- segmentation services.
- prompt segmentation services.
- mask editing.
- quantitative measurements.
- anatomy masks.
- audit/versioning.

## Scientific mode UI

- lesion queue,
- accept/reject/correct/new/resolved,
- longitudinal tracking,
- registration QC,
- dataset completion QC.

## Clinical mode UI - deferred

Will later include:

- patient history/dictation,
- indication,
- PSA/Gleason/TNM/prior therapies,
- E-PSMA,
- PSMA-RADS,
- miTNM,
- structured report,
- LLM-assisted narrative report,
- bilingual report,
- export.

Do not couple Scientific Mode to clinical report generation logic.

# 18. Implementation priorities

## Priority 1 - Core review state machine

Implement first:

1. stable LesionTrack IDs,
2. observation states,
3. Accept/Reject/Edit/New/Uncertain/Resolved actions,
4. keyboard shortcuts,
5. autosave and audit events,
6. automatic next observation/timepoint.

## Priority 2 - Three-monitor workspace

1. M1 current lesion image-only layout,
2. M2 current/reference longitudinal compare,
3. M3 workflow console,
4. synchronized lesion/timepoint selection across windows.

## Priority 3 - Mask correction/versioning

1. immutable AI mask,
2. expert mask versions,
3. Paint/Erase/Undo/Redo,
4. metric recomputation,
5. correction metrics.

## Priority 4 - Prompt segmentation

1. default prompt model,
2. three-model selector,
3. point/negative-point/box prompts,
4. generated mask review and correction,
5. new LesionTrack creation.

## Priority 5 - Longitudinal tracking

1. auto-match proposals,
2. track-wise TP navigation,
3. unmatched candidate sweep,
4. reconnect resolved/reappearing lesion,
5. manual match.

## Priority 6 - Registration QC

1. questionable flag,
2. Registration Review Mode,
3. landmark/crosshair comparison,
4. local offset without modifying global transform,
5. manual match,
6. not-evaluable status.

## Priority 7 - Completion QC and research export

1. unresolved-item checks,
2. lock/version annotation set,
3. export structured data and mask references,
4. provenance/model-version metadata.

# 19. Testable acceptance criteria

The developer should consider these as minimum functional acceptance tests.

## 19.1 Normal lesion

Given a presegmented lesion at TP3, when the user presses `A`:

- status becomes Accepted/Present,
- audit event is created,
- data is autosaved,
- next TP of same lesion loads automatically,
- M1 and M2 update synchronously.

## 19.2 Corrected lesion

Given an AI mask, when the user enters Edit mode, modifies mask, and presses Enter:

- original mask remains unchanged,
- new expert mask version exists,
- metrics are recomputed,
- status is Corrected/Present,
- next TP loads.

## 19.3 Missing lesion

When user creates a lesion using prompt segmentation:

- new lesion receives a stable track ID,
- prompt/model provenance is stored,
- final mask is versioned,
- lesion appears in lesion queue and longitudinal workflow.

## 19.4 Resolved lesion

When user presses `X` for an established lesion:

- lesion track remains intact,
- current observation becomes Resolved/Not Visible,
- next TP loads,
- later reappearance can be linked to same track.

## 19.5 Poor registration

When user flags registration:

- M2 enters Registration Review Mode,
- global transform cannot be accidentally overwritten by a lesion-level local offset,
- local adjustment/manual match/not-evaluable result is persisted with provenance.

## 19.6 False-positive candidate

When user presses `R`:

- AI mask remains stored,
- candidate decision becomes Rejected,
- optional rejection reason can be stored,
- workflow advances.

## 19.7 Crash/restart

After a committed decision and unexpected restart:

- committed state is recovered,
- no expert mask is lost,
- current case can resume at next unresolved observation.

# 20. Suggested software module boundaries

Exact names are illustrative.

- `ViewerController`
  - M1/M2 viewport synchronization.
  - coordinate transforms.
  - lesion centering.
  - display presets.

- `AnnotationWorkflowController`
  - state machine.
  - next observation logic.
  - action guards.
  - queue filters.

- `LesionTrackService`
  - create/link/reconnect tracks.
  - longitudinal presence states.
  - manual matching.

- `SegmentationService`
  - load AI masks.
  - prompt model invocation.
  - mask version creation.
  - editing operations.

- `MeasurementService`
  - volume/SUV/diameter recomputation.
  - mask-version-specific metrics.

- `RegistrationService`
  - global transforms.
  - local lesion-level offsets.
  - registration QC.

- `PersistenceService`
  - transactional save.
  - autosave queue.
  - recovery.

- `AuditService`
  - immutable event log.

- `PrefetchCache`
  - next TP/lesion image and mask preloading.

- `ScientificModeView`
  - M3 workflow UI.

- `ClinicalModeView`
  - deferred; must not be required by scientific workflow.

# 21. Event/API design suggestion

For maintainability, UI buttons and keyboard shortcuts should dispatch the same semantic commands rather than duplicate logic.

Examples:

- `AcceptObservation(observation_id)`
- `RejectCandidate(observation_id, reason=None)`
- `StartEdit(observation_id)`
- `ApplyMaskCorrection(observation_id, mask_version)`
- `CreateNewLesion(case_id, timepoint_id)`
- `RunPromptSegmentation(model_id, prompts)`
- `MarkResolved(observation_id)`
- `MarkUncertain(observation_id)`
- `FlagRegistration(timepoint_pair, lesion_track_id)`
- `ApplyLocalRegistrationOffset(...)`
- `ConfirmManualMatch(...)`
- `MarkNotEvaluable(...)`
- `AdvanceToNextObservation()`

This allows keyboard, mouse, automated testing, and future UI redesigns to reuse the same application logic.

# 22. Concurrency and transactional behavior

When saving a committed observation, update related data atomically where practical:

1. new segmentation version if any,
2. observation state,
3. quantitative metrics,
4. audit event,
5. current workflow pointer/progress.

If one operation fails, the UI should not falsely report success.

# 23. Undo/redo policy

Undo/redo should be available for mask editing and ideally for the immediately preceding annotation action.

Recommended policy:

- mask edits: standard undo/redo stack within current Edit session,
- after Apply: create a new version rather than deleting prior version,
- changing Accepted to Rejected later should create a new audit event, not rewrite history,
- no audit record should be physically deleted during routine use.

# 24. Data export for research

The final export should be machine-readable and lesion/timepoint oriented.

Minimum logical export columns/fields:

- case ID,
- timepoint ID/date,
- lesion track ID,
- lesion class/anatomy,
- candidate decision,
- presence state,
- match method,
- registration QC,
- segmentation version ID,
- segmentation origin/model/version,
- volume,
- diameter,
- SUVmax,
- SUVmean,
- reviewer ID,
- review timestamp,
- optional rejection reason,
- optional comment.

Masks should be exported/referenced in a format preserving physical geometry and traceability to source images.

# 25. AI-assisted code refactoring (Gemini or another coding model)

An LLM may be used to accelerate refactoring, but it should not be asked to rewrite the entire application blindly in one pass.

## 25.1 Safe workflow

1. Put current working code under version control and create a dedicated branch.
2. Add tests for existing viewer interactions before refactoring.
3. Refactor one module boundary at a time.
4. Require the model to preserve specified interactions and data semantics.
5. Compile/run tests after each change.
6. Review diffs manually.
7. Never allow the model to silently change DICOM geometry, SUV calculations, image orientation, mask coordinate systems, or registration transforms.
8. Keep AI-generated code changes reviewable and reversible.

## 25.2 Information to provide to the coding model

For each requested change, provide:

- current relevant files/functions,
- desired state transition,
- exact UI event/shortcut,
- data fields changed,
- invariants that must remain unchanged,
- acceptance test,
- prohibited side effects.

## 25.3 Reusable refactoring prompt template

> You are modifying an existing PSMA PET/CT scientific annotation application. Implement only the requested feature while preserving existing DICOM geometry, SUV quantification, image orientation, registration transforms, and current mouse controls. The application uses a lesion-track-centric workflow with immutable AI masks and versioned expert masks. UI buttons and keyboard shortcuts must dispatch the same semantic application command. Do not redesign unrelated modules. First summarize the affected modules and state transitions, then produce the minimal code changes, tests, and migration notes. The feature is complete only when the supplied acceptance criteria pass.

Then append one specific feature, for example:

> Feature: pressing A in LESION_REVIEW must atomically mark the current observation ACCEPTED + PRESENT, create an audit event, autosave, and advance to the next timepoint of the same lesion. M1 and M2 must update synchronously. The original AI mask must remain immutable.

# 26. Developer handoff checklist

Before coding begins, confirm:

- [ ] Current framework/language and GUI toolkit.
- [ ] How M1/M2/M3 are implemented (separate windows, screen assignment, persistence of window positions).
- [ ] Current image coordinate convention and physical coordinate transforms.
- [ ] Existing DICOM/PET SUV handling.
- [ ] Existing segmentation file format and geometry.
- [ ] Existing registration representation.
- [ ] Current three prompt segmentation models and their interfaces.
- [ ] Current database/storage format.
- [ ] Whether annotations are single-user or multi-user.
- [ ] Whether offline/local autosave is required.
- [ ] Required export format for research.
- [ ] Existing shortcut handlers and conflicts.
- [ ] Undo/redo support in current code.
- [ ] Expected maximum number of timepoints/lesions per case.
- [ ] GPU/model execution location and expected latency.

# 27. Recommended implementation order for the current prototype

A low-risk sequence is:

1. **Do not redesign the image renderer first.** Preserve the working viewer and mouse controls.
2. Introduce explicit domain objects/enums for LesionTrack, LesionObservation, SegmentationVersion, RegistrationQC, and AuditEvent.
3. Centralize actions into an AnnotationWorkflowController/state machine.
4. Add autosave/versioning around the current segmentation functions.
5. Split Scientific Mode into the three monitor windows with synchronized selection state.
6. Implement Accept/Reject/Correct/Resolved and auto-advance.
7. Implement lesion-centric timepoint navigation.
8. Integrate promptable new-lesion creation.
9. Implement unmatched-candidate sweep.
10. Add Registration Review Mode and lesion-level local offset/manual matching.
11. Add completion QC and research export.
12. Only after this workflow is stable, build the separate Clinical Mode on the shared backend.

# 28. Final workflow summary

The intended normal scientific workflow is:

`Open case`

-> `Anchor TP cleanup`

-> `Stable lesion tracks created`

-> `Select L001`

-> `Review TP0 -> TP1 -> TP2 -> ... TPn`

-> `Accept / Correct / Resolve / Uncertain / Registration QC as required`

-> `Autosave after every committed action`

-> `Select next lesion track`

-> `After all tracks: unmatched-candidate sweep for each follow-up TP`

-> `Create/link/reject late candidates`

-> `Completion QC`

-> `Lock/version expert annotation dataset`

The core user experience should remain simple even though the data model is rich:

**M1: What is it now?**

**M2: Is it the same lesion and how did it change?**

**M3: What do I tell the system?**

