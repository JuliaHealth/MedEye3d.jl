---
title: "PSMA PET/CT Clinical Reporting Mode"
subtitle: "Software Requirements, UX Workflow and Semi-Structured Reporting Specification"
author: "Developer handoff specification"
date: "2026-09-13"
version: "1.0"
---

# 1. Document purpose

This document specifies the **Clinical Reporting Mode** of the existing PSMA PET/CT workstation. It is intended as an implementation handoff for a professional software developer who may use **Gemini Pro or another coding assistant** to refactor the current codebase.

The goal is not to build a second application. Scientific Annotation Mode and Clinical Reporting Mode should share the same core services: DICOM/PET/CT loading, multimodal visualization, segmentation, promptable segmentation, anatomical localization, lesion objects, quantitative measurements, longitudinal registration, lesion matching, reference-tissue measurements, and export infrastructure. The clinical mode is a different **workflow layer** optimized for routine reporting rather than exhaustive ground-truth annotation.

The target environment is a radiology/nuclear-medicine workstation with:

- **M1:** high-resolution diagnostic portrait monitor for the current examination.
- **M2:** high-resolution diagnostic portrait monitor for prior/longitudinal/whole-body context.
- **M3:** standard landscape support monitor for clinical context, workflow control, staging/response, dictation, report composition, and key-image management.

The final product should support the typical PSMA PET/CT scenarios:

1. Initial staging.
2. Biochemical recurrence/restaging.
3. Interim/response staging.
4. Pre-radioligand-therapy baseline.
5. Post-radioligand-therapy follow-up.
6. General restaging/follow-up.

Formal RECIST 1.1 support should be available only when the CT acquisition is suitable and the physician explicitly activates it. PROMISE/miTNM and PSMA-specific response logic should be rule-based and physician-confirmed. LLMs are used for language generation and structured extraction, **not as the authority for quantitative measurements or staging rules**.

## 1.1 Primary objective

Enable a radiologist/nuclear medicine physician to move efficiently from clinical question to signed report:

`CASE SETUP -> READ -> ASSESS -> DICTATE -> REPORT -> VALIDATE -> SIGN`

The software should perform routine mechanical work automatically:

- load and synchronize relevant current/prior imaging,
- surface findings that actually require physician attention,
- reuse structured data from segmentation and longitudinal tracking,
- calculate quantitative changes,
- suggest guideline-based staging/response classifications,
- preserve and compare prior report content,
- convert physician dictation plus structured findings into a concise semi-structured report,
- identify conflicts before sign-off,
- attach physician-selected key images automatically.

## 1.2 Clinical unit of work

The primary clinical unit is the **case**, but lesion objects remain the evidence layer underneath the case. The physician should not be forced to process every lesion individually in disseminated disease.

The interface therefore distinguishes:

- **Complete machine lesion set:** all detected/segmented lesions retained for quantification and research.
- **Clinically relevant findings:** lesions or lesion groups that change staging, response, management, diagnostic certainty, or the report.

This distinction is fundamental for efficiency.

# 2. Product principles

## 2.1 Diagnostic monitors show images; support monitor manages workflow

M1 and M2 are diagnostic displays. Permanent large tables, long forms, report prose, model-selection controls, and workflow buttons should not consume diagnostic image space. M3 may be information-dense because it is not used for primary diagnostic pixel evaluation.

## 2.2 The physician interprets; the software structures

The physician remains responsible for medical interpretation. The software should:

- measure,
- compare,
- calculate,
- organize,
- retrieve prior information,
- expose rule-based derived results,
- generate language only after the underlying facts are known.

## 2.3 Context-sensitive guided workflow

The interface should not expose every possible field at all times. Case type drives the active workflow and visible questions. For example:

- Initial staging emphasizes primary tumor, nodal and metastatic distribution, and miTNM.
- BCR emphasizes recurrence location, PSA context, prior therapy, and disease compartments.
- Post-RLT emphasizes current-vs-baseline/current-vs-previous comparison, new lesions, disease burden, and PSMA response.

## 2.4 Semi-structured report, not a visible database dump

The final report must remain readable as normal clinical prose. Structured lesion data remain underneath and generate the text, staging, response, tables, and images.

## 2.5 Previous report is context, never current truth

A previous report may be reused as a template, but old statements must not silently persist. The software should compare previous statements with current structured findings and explicitly flag contradictions.

## 2.6 Deterministic medical facts, generative language

The following must be deterministic or physician-entered and must never be invented by an LLM:

- SUV measurements,
- lesion dimensions and volumes,
- TMTV and lesion counts,
- anatomical laterality,
- timepoint identity,
- lesion-track identity,
- current/prior numerical deltas,
- reference tissue measurements,
- PROMISE/miTNM rule execution,
- RECIP rule execution,
- RECIST calculations,
- tracer/activity/uptake-time metadata,
- examination dates,
- segmentation geometry.

The LLM may:

- structure dictated history,
- map dictation to existing lesion/finding objects,
- rewrite findings into professional language,
- summarize longitudinal changes,
- draft the impression,
- identify language-level inconsistencies for physician review.

## 2.7 Explainability for derived classifications

Every derived staging/response result should have a visible **Why?** action. Example:

`miM1b -> because validated PSMA-positive osseous lesion(s) are present.`

`RECIP candidate PR -> TMTV decreased by X% and no new PSMA-positive lesions were confirmed.`

The physician must be able to verify the evidence without reverse-engineering the rule engine.

# 3. Workstation architecture

## 3.1 Monitor geometry

The workstation contains two portrait-oriented high-resolution diagnostic monitors and one landscape support monitor. The UI must be designed natively for this geometry rather than scaling a laptop layout.

| Monitor | Orientation | Primary purpose |
| --- | --- | --- |
| M1 | Portrait diagnostic | Current PSMA PET/CT and focused lesion assessment |
| M2 | Portrait diagnostic | Prior/current comparison, whole-body context, response context |
| M3 | Landscape standard | Clinical workflow, relevant findings, dictation, staging/response, report, key images |

## 3.2 Cross-monitor coordination

The three windows should behave as one workstation session. Selecting a finding on M3 must immediately update M1 and M2. Changing the current slice/point on M1 should optionally synchronize M2. The session state must include:

- current patient/case,
- current examination,
- selected prior/reference examination,
- selected finding/lesion,
- selected viewer mode,
- current workflow phase,
- staging/response state,
- report draft version,
- key-image selection.

Cross-monitor communication should be event-driven rather than implemented as duplicated business logic.

# 4. Monitor 1 - Current Examination

## 4.1 Purpose

M1 answers: **What is present in the current examination?**

It is the primary diagnostic image workspace.

## 4.2 Default portrait layout

Recommended default:

- Upper 55-60%: large axial PET/CT fusion.
- Lower left: coronal fusion.
- Lower right: sagittal fusion.
- Bottom narrow strip: series/display presets.

The dominant image may be changed by double-clicking another plane.

## 4.3 Required modes

At minimum:

- Fusion.
- PET only.
- CT soft tissue.
- CT bone.
- MIP.
- Optional subtraction/difference view when technically valid.

## 4.4 Image interaction

Reuse the Scientific Mode interaction model wherever possible:

- Mouse wheel: next/previous slice.
- Ctrl + wheel: zoom.
- Shift + wheel: PET upper-window/SUV scale or configured secondary window function.
- Right click: synchronize that anatomical point across views.
- Right drag: pan.
- Double left click: make the clicked plane dominant / recenter orthogonal views.
- Hold Q: temporarily hide lesion mask/contour.
- C: center selected finding.
- F1-F4: configured display presets.

## 4.5 Overlays

Keep overlays minimal:

- current examination date,
- tracer,
- plane and slice index,
- HU/SUV under cursor,
- PET scale,
- scale bar,
- selected finding ID and brief label,
- segmentation contour when enabled.

Do not display long clinical text, report prose, staging tables, or lesion queues on M1.

# 5. Monitor 2 - Context / Prior / Longitudinal Comparison

## 5.1 Purpose

M2 answers: **How does the current finding relate to prior imaging and to whole-body disease distribution?**

The content is dynamic by case type.

## 5.2 Mode A - Prior comparison

Default when a relevant prior examination exists.

Portrait layout:

- Upper area: current vs prior synchronized lesion-centered views.
- Middle area: orthogonal current/prior comparison or whole-body context.
- Lower area: current/prior MIP or a compact timepoint selector.

Features:

- same orientation,
- same physical FOV where possible,
- synchronized crosshair,
- synchronized PET scale option,
- prior/baseline reference selector,
- side-by-side,
- overlay,
- flicker.

## 5.3 Mode B - Initial staging without prior

When no useful prior exists, M2 should not show empty comparison panes. It becomes a whole-body staging monitor:

- large PET MIP,
- whole-body coronal fusion,
- optional skeleton/anatomy view,
- disease-compartment overview.

## 5.4 Mode C - Post-RLT / response

Default reference should be configurable:

- **Current vs pre-RLT baseline** as the primary therapy-response comparison.
- One action toggles to **Current vs previous cycle/examination**.

This distinction is mandatory because clinical questions differ: total response from baseline versus short-interval change from the previous cycle.

## 5.5 Mode D - Focused lesion comparison

Selecting a clinically important finding on M3 should create a focused current/prior comparison on M2, centered on the same lesion track.

## 5.6 Registration failure

If registration is questionable, M2 exposes the registration-review tools already defined for Scientific Mode. Clinical reporting must never present a numerical or visual comparison as reliable without recording registration quality.

# 6. Monitor 3 - Clinical Workflow Console

## 6.1 Primary concept

M3 should have three major user-facing phases rather than a permanently expanded multi-tab form:

1. **READ** - clinical context, prior report, relevant findings.
2. **ASSESS** - staging/response and quantitative synthesis.
3. **REPORT** - semi-structured report, validation, key images, sign-off.

Internal subpanels may still correspond to History, Prior, Findings, Staging/Response, and Report, but the UI should feel like a guided clinical workflow rather than form-filling software.

## 6.2 Persistent header

Always show:

- patient identifier appropriate to the deployment,
- current examination date,
- tracer,
- selected case type/workflow profile,
- prior/reference examination,
- PSA when available and relevant,
- workflow completion indicators.

Example:

`P014 | 68Ga-PSMA-11 | 13.09.2026 | BCR | Prior 07.04.2026 | PSA 3.8`

# 7. Case-type selection and workflow profiles

## 7.1 Workflow selector

At case start, determine or ask for a profile:

- Initial staging.
- Biochemical recurrence/restaging.
- Interim/response staging.
- Pre-RLT baseline.
- Post-RLT follow-up.
- General restaging/follow-up.

The profile can be preselected from order/indication text but should be physician-confirmable.

## 7.2 Why case profiles matter

A single static dashboard would overload the user. Profiles control:

- which clinical fields are requested,
- what M2 displays,
- which lesion/finding priorities are used,
- which rule engines are active,
- which report sections are generated,
- what quantitative summary is shown.

# 8. READ phase - History and indication

## 8.1 Dictation-first history capture

Primary interaction should be a large **Dictate History** control. The physician may dictate natural language such as:

`Prostate carcinoma after radical prostatectomy in 2023, Gleason 4+4, ISUP 4, initial pT3a pN0, biochemical recurrence with PSA 3.8 ng/mL and PSADT about six months, no current ADT.`

The speech/LLM pipeline should produce both:

1. verbatim/raw dictation text, and
2. structured candidate fields.

## 8.2 Structured history fields

At minimum support:

- indication/case type,
- PSA,
- PSA kinetics/PSADT when available,
- Gleason score,
- ISUP grade,
- initial TNM,
- current/previous therapies,
- prostatectomy,
- radiotherapy,
- ADT,
- AR pathway inhibitor,
- chemotherapy,
- prior RLT,
- relevant dates,
- free-text notes.

The physician should confirm only ambiguous or high-impact extracted fields.

## 8.3 Provenance

Each structured field should store provenance:

- imported from RIS/order,
- imported from prior report,
- extracted from dictation,
- manually entered,
- edited by physician.

# 9. READ phase - Prior report handling

## 9.1 Prior report panel

If a prior report exists, display it as context with three explicit actions:

- **Use as context**.
- **Compare with current findings**.
- **Use as report template**.

Avoid a primary action called simply "Copy report" because it encourages unsafe carryover.

## 9.2 Prior-to-current statement mapping

Where feasible, parse the prior report into structured statements or map its statements to known prior lesion tracks/compartments.

For each prior statement classify:

- still valid/unchanged,
- modified,
- resolved/no longer valid,
- contradicted by current findings,
- uncertain/unmapped.

## 9.3 Report template behavior

When **Use as report template** is selected:

- preserve section structure,
- update objective values from current data,
- remove statements contradicted by current validated findings,
- add new findings,
- preserve unchanged text only after validation,
- visually mark changes in the draft until physician approval.

# 10. READ phase - Findings review

## 10.1 Relevant Findings Queue

Clinical Mode should not force exhaustive lesion-by-lesion confirmation. M3 should prioritize a queue of findings that deserve physician attention.

Priority examples:

1. new lesion,
2. staging-changing lesion,
3. response-determining lesion,
4. suspected false positive,
5. large interval change,
6. low-confidence segmentation,
7. questionable anatomy/laterality,
8. lesion selected as RECIST target,
9. lesion suggested as a key image,
10. registration-dependent finding.

Button/shortcut: **Next Relevant Finding**.

## 10.2 Grouping

Default list should be grouped by disease compartment:

- prostate/prostate bed,
- pelvic nodes,
- extrapelvic nodes,
- bone,
- visceral/other.

In disseminated disease, the interface should show group counts and representative lesions rather than 80 equally prominent rows.

## 10.3 Lesion actions

Reuse the same core controls as Scientific Mode to reduce training burden:

- A = Accept.
- R = Reject.
- E = Edit/Correct.
- N = New lesion.
- P = Prompt segmentation.
- C = Center lesion.

Clinical Mode may treat high-confidence, noncritical machine lesions as accepted for quantification without requiring explicit individual confirmation, depending on deployment policy. However, any lesion used to change stage/response or included as a key finding must be physician-validated.

# 11. Finding dictation

## 11.1 Context-linked dictation

When a finding is selected, pressing **D** starts finding dictation linked to that object.

Example physician dictation:

`Left iliac bone metastasis clearly regressive compared with the prior study, still definitely PSMA positive.`

The application already knows:

- lesion ID,
- location,
- current SUVmax,
- previous SUVmax,
- current/prior volume,
- current/prior size,
- response state.

The report generator may therefore produce:

`The known PSMA-positive metastasis in the left ilium shows clear metabolic and volumetric regression (SUVmax 12.1 -> 8.7; segmented volume -28%).`

The physician supplies interpretation; the software supplies validated objective numbers.

## 11.2 Dictation provenance

Store:

- raw audio reference if permitted by deployment,
- raw transcription,
- linked finding IDs,
- generated/rewritten sentence,
- final physician-edited sentence.

# 12. ASSESS phase - Initial staging

## 12.1 Primary output

PROMISE/miTNM should be prominent.

The rule engine derives a candidate classification from validated findings and anatomy. Example:

- miT category,
- miN category,
- miM category and subcategory.

## 12.2 Physician confirmation

The candidate stage is not silently finalized. Provide:

- candidate classification,
- evidence list,
- Why? explanation,
- Confirm/Override controls,
- override reason field when applicable.

## 12.3 Display priorities

Initial staging view emphasizes:

- primary/prostate bed,
- local extension,
- pelvic nodes,
- extrapelvic nodes,
- bone metastases,
- visceral metastases,
- PSMA expression/reference tissues.

Do not make RECIST a prominent default in initial staging.

# 13. ASSESS phase - Biochemical recurrence

## 13.1 Primary question

`Where is recurrent disease?`

Provide a compartment summary:

| Compartment | Status |
| --- | --- |
| Prostate bed/local | Positive / Negative / Indeterminate |
| Pelvic nodes | Positive / Negative / Indeterminate |
| Extrapelvic nodes | Positive / Negative / Indeterminate |
| Bone | Positive / Negative / Indeterminate |
| Visceral | Positive / Negative / Indeterminate |

## 13.2 Context fields

Surface when available:

- current PSA,
- PSA kinetics/PSADT,
- prostatectomy/radiotherapy history,
- current systemic therapy,
- prior PSMA PET/CT date.

## 13.3 Output

Produce:

- recurrence distribution,
- candidate miTNM/PROMISE classification,
- concise quantitative disease summary,
- optional comparison with prior examination.

# 14. ASSESS phase - Pre-RLT baseline

## 14.1 Primary question

`What is the baseline PSMA-positive disease burden and distribution before radioligand therapy?`

## 14.2 Core quantitative summary

Display:

- TMTV / total PSMA-positive tumor volume,
- lesion count where meaningful,
- dominant/representative lesions,
- distribution by compartment,
- blood pool SUVmean,
- liver SUVmean,
- parotid SUVmean,
- uptake heterogeneity indicators if implemented,
- bulky disease flags,
- osseous/visceral involvement.

## 14.3 Baseline designation

One physician-confirmed action should mark the study as **RLT baseline**. Future post-RLT examinations should automatically offer this examination as the baseline reference on M2.

# 15. ASSESS phase - Interim/Post-RLT response

## 15.1 Comparison references

Support two references:

- Baseline/pre-RLT.
- Previous examination/cycle.

The UI must make it unambiguous which reference is active.

## 15.2 Response dashboard

Compact example:

| Parameter | Baseline | Previous | Current | Delta baseline |
| --- | ---: | ---: | ---: | ---: |
| TMTV | 128 cc | 74 cc | 51 cc | -60% |
| PSMA-positive lesions | 34 | 27 | 21 | -13 |
| Representative SUVmax | 31 | 18 | 14 | -55% |
| New lesions | - | 0 | 0 | - |

Values are generated from structured data, not LLM interpretation.

## 15.3 RECIP

Where the clinical context and implementation criteria are applicable, calculate a **candidate RECIP classification** using deterministic code. Show:

- candidate class,
- inputs used,
- new-lesion status,
- tumor-volume change,
- Why? explanation,
- physician Confirm/Override.

Do not infer RECIP solely from prose.

# 16. ASSESS phase - RECIST 1.1

## 16.1 Gating question

Before RECIST can be activated:

`Is the CT acquisition suitable for formal RECIST assessment?`

Options:

- Yes.
- No.
- Not assessed.

If No, the report can explicitly state that formal RECIST was not applied because the CT protocol was unsuitable.

## 16.2 Target lesions

When activated, allow physician selection of target lesions from validated findings. Store target/non-target state separately from PSMA lesion identity.

## 16.3 Deterministic calculations

The system calculates:

- sum of target-lesion diameters,
- baseline/nadir/current comparisons,
- percentage change,
- candidate RECIST category.

The physician confirms the final category.

# 17. REPORT phase - Semi-structured report architecture

## 17.1 Design goal

The report should be readable within seconds by urologists, oncologists, radiation oncologists, and nuclear medicine/radiology colleagues while preserving machine-readable structured data underneath.

## 17.2 Recommended final section order

1. **Indication / Clinical Information**.
2. **Comparison Examination**.
3. **Technique**.
4. **Reference Activity**.
5. **Findings** organized by anatomy/disease compartment.
6. **Longitudinal Comparison / Response** when applicable.
7. **Impression**.
8. **Synoptic Summary**.
9. **Key Findings / Key Images**.

## 17.3 Indication / Clinical information

Short prose generated from validated structured clinical context. Avoid dumping all database fields.

Example:

`Biochemical recurrence after radical prostatectomy for prostate carcinoma (initial pT3a pN0, Gleason 4+4/ISUP 4). Current PSA 3.8 ng/mL, PSADT approximately 6 months. No current ADT.`

## 17.4 Comparison examination

Example:

`Comparison: PSMA PET/CT dated 07.04.2026.`

If there are multiple relevant studies, the report may identify both baseline and immediately prior examinations.

## 17.5 Technique

Prefer automatic population from DICOM/worklist data. Fields may include:

- tracer,
- injected activity,
- uptake time,
- acquisition range,
- PET reconstruction details when desired,
- CT type/protocol,
- IV/oral contrast,
- diuretic administration.

Missing values must be marked as not available rather than invented.

## 17.6 Reference activity

Optional compact table or sentence containing validated reference measurements:

- mediastinal blood pool SUVmean,
- liver SUVmean,
- parotid SUVmean.

The report generator may use these values for standardized PSMA expression categories.

# 18. REPORT phase - Findings prose

## 18.1 Anatomical organization

Use consistent sections:

- Prostate / prostate bed.
- Regional pelvic lymph nodes.
- Extrapelvic lymph nodes.
- Skeleton.
- Visceral / other findings.

## 18.2 Oligometastatic versus polymetastatic presentation

For a small number of lesions, individual lesions may be described.

For disseminated disease, summarize at compartment level and list only representative, dominant, response-determining, or management-relevant lesions. The report should not become a 60-row lesion inventory.

## 18.3 Negative statements

Negative statements should be generated only when the relevant compartment has been adequately reviewed and the structured evidence supports the statement.

# 19. REPORT phase - Longitudinal comparison

## 19.1 Dedicated comparison section

Avoid scattering `compared with prior` through every sentence. When meaningful, include a concise dedicated section such as:

`Compared with 07.04.2026, there is clear metabolic and morphologic regression of the previously PSMA-positive pelvic lymph-node metastases. The left iliac bone metastasis shows a decrease in SUVmax from 12.1 to 8.7 and a 28% decrease in segmented volume. No new PSMA-positive lesions.`

## 19.2 Quantitative response block

For RLT/response cases, show a compact table rather than embedding every value in prose.

# 20. REPORT phase - Impression

## 20.1 Design rule

The Impression is the most important section for the referrer. Keep it short: normally 3-5 numbered statements.

Example BCR:

1. PSMA-positive local recurrence in the prostate bed.
2. Two PSMA-positive left external-iliac lymph-node metastases.
3. Solitary PSMA-positive osseous metastasis in the left ilium.
4. Molecular staging: candidate PROMISE/miTNM result, physician-confirmed.

Example post-RLT:

1. Marked reduction of PSMA-positive tumor burden compared with pre-therapy baseline.
2. No newly developed PSMA-positive lesions.
3. Physician-confirmed PSMA response classification.
4. Residual nodal and osseous PSMA-positive disease remains.

# 21. REPORT phase - Synoptic summary

Provide a compact structured block after the narrative impression. Example fields:

| Field | Value |
| --- | --- |
| Workflow | Biochemical recurrence |
| PROMISE/miTNM | physician-confirmed value |
| Positive compartments | prostate bed, pelvic nodes, bone |
| TMTV | 4.3 cc |
| Highest PSMA expression | Score 2 |
| New lesions | none |
| Response | not applicable / confirmed category |
| Reader confidence | high |

The visible summary should remain concise. The full structured data are stored in the database/export layer.

# 22. Key Images subsystem

## 22.1 Rationale

Some referrers benefit substantially from selected screenshots of the most important findings. Key-image creation must be integrated into normal reporting and require almost no extra work.

## 22.2 Marking a key finding

Each clinically relevant finding has a **star / Key Image** toggle. The physician may mark it from M3 or a viewer shortcut.

The software may suggest key findings, but the physician confirms them.

Suggested priority logic:

1. finding that changes M0 to M1 or otherwise changes stage,
2. newly developed lesion,
3. response-determining lesion,
4. primary/local recurrence,
5. clinically important equivocal finding,
6. representative lesion in disseminated disease.

Default maximum suggested key findings: approximately 3-5, configurable.

## 22.3 Automatic image capture - single examination

For each selected key finding automatically generate two standardized panels:

- axial PET/CT fusion,
- coronal PET/CT fusion,

or substitute sagittal when configured/selected as more informative.

Capture state should include:

- lesion-centered coordinate,
- same lesion contour visibility setting,
- physical FOV,
- PET scale/window,
- CT window,
- slice reference,
- optional arrow/marker,
- short caption.

## 22.4 Automatic image capture - longitudinal case

For a comparison finding, generate a four-panel composite:

- prior axial,
- current axial,
- prior coronal,
- current coronal.

Use the same registered center, physical FOV, and PET scale where appropriate. Clearly label dates and Current/Prior.

## 22.5 Captions

Captions are generated from structured data and physician interpretation, for example:

`Fig. 2. PSMA-positive metastasis in the left ilium, current SUVmax 8.7, with clear regression compared with the prior examination.`

The physician may edit the caption.

## 22.6 Data provenance for screenshots

Store source references for every image:

- study/series/SOP identifiers,
- slice or reconstructed plane definition,
- viewer coordinate/FOV,
- PET and CT display settings,
- segmentation/annotation version,
- current/prior timepoint IDs.

The screenshot is communication output; it must remain traceable to source imaging.

## 22.7 Export

At minimum support embedding in the final Word/PDF report. Optional future integration may support PACS-compatible key objects/secondary-capture workflows depending on institutional infrastructure.

# 23. Report Composer UI

## 23.1 Entering REPORT phase

The report should not occupy significant M3 space during active image reading. Enter the full Report Composer only after READ/ASSESS is substantially complete.

## 23.2 Layout

Recommended landscape support-monitor layout:

- Left column: report section navigator and validation status.
- Center: editable report draft.
- Right column: structured summary, conflicts, staging/response confirmation, key-image thumbnails.
- Bottom toolbar: Dictate, Regenerate selected section, Compare prior, Validate, Export/Sign.

## 23.3 Regeneration granularity

Do not regenerate the entire report for every edit. Support section-level regeneration:

- History.
- Findings subsection.
- Comparison.
- Impression.

Physician manual edits should not be silently overwritten.

# 24. Conflict checker

## 24.1 Mandatory pre-sign validation

Before sign-off, compare the report draft against structured data and rule-engine outputs.

Flag at least:

- report says no bone metastasis while validated bone lesion exists,
- miM1b lesion exists but report states M0,
- new lesion exists while report says no new lesions,
- left/right inconsistency,
- current numerical value differs from database value,
- resolved lesion copied from prior report as still present,
- response category inconsistent with deterministic response inputs,
- RECIST target measurement missing,
- baseline/previous dates inconsistent,
- tracer mismatch,
- unreviewed critical finding remains.

## 24.2 Severity levels

Use:

- **Blocking** - cannot sign until resolved.
- **Warning** - physician may sign after acknowledgement.
- **Informational** - no action required.

# 25. State model and workflow transitions

## 25.1 High-level states

`CASE_SETUP -> READ -> ASSESS -> REPORT_DRAFT -> VALIDATION -> SIGNED`

Optional transitions allow returning to earlier phases without data loss.

## 25.2 Case setup

Actions:

- load current examination,
- detect available priors,
- select/confirm workflow profile,
- load imported clinical data,
- initialize segmentation/lesion/registration services.

Exit condition: images ready and case profile known.

## 25.3 READ

Actions:

- history dictation/confirmation,
- prior report review,
- current imaging review,
- relevant-finding queue,
- selective segmentation correction,
- new lesion creation,
- finding dictation,
- key-image marking.

Exit condition: all critical findings reviewed or explicitly deferred.

## 25.4 ASSESS

Actions depend on profile:

- PROMISE/miTNM,
- BCR compartment summary,
- pre-RLT baseline quantitative summary,
- RECIP candidate,
- optional RECIST.

Exit condition: required derived classifications confirmed/overridden.

## 25.5 REPORT_DRAFT

Generate semi-structured report from structured data, prior report context, and physician dictation.

## 25.6 VALIDATION

Run conflict checker, verify missing fields, verify key images, physician edits.

## 25.7 SIGNED

Lock signed report version and preserve provenance. Later amendments create a new version rather than silently overwriting the signed report.

# 26. Data model

## 26.1 Core entities

Suggested entities:

- Patient/Case.
- Examination/Timepoint.
- LesionTrack.
- LesionObservation.
- SegmentationVersion.
- MeasurementSet.
- AnatomicalClassification.
- ClinicalContext.
- PriorReport.
- FindingStatement.
- StagingAssessment.
- ResponseAssessment.
- DictationRecord.
- ReportDraft/ReportVersion.
- KeyImage.
- ValidationIssue.
- AuditEvent.

## 26.2 LesionTrack

Minimum fields:

- track_id,
- patient_id,
- canonical lesion type,
- canonical anatomy/laterality,
- created_from timepoint,
- active/inactive state,
- linked observations.

## 26.3 LesionObservation

Minimum fields:

- observation_id,
- track_id,
- timepoint_id,
- current mask version,
- AI mask reference,
- physician validation state,
- size/volume,
- SUVmax/SUVmean/SUVpeak if available,
- CT correlate,
- PSMA expression category,
- confidence,
- clinical relevance flags,
- staging contribution,
- response contribution,
- key-image flag,
- registration quality.

## 26.4 ClinicalContext

Store each field with value + source + confirmation status.

## 26.5 ReportVersion

Minimum fields:

- report_version_id,
- case_id,
- status draft/final/signed/amended,
- generated_at,
- generator/model version,
- source structured-data snapshot/hash,
- physician edits,
- final text,
- linked key images,
- validation results,
- signed_at/user.

# 27. Event-driven architecture

Recommended UI events:

- `CASE_LOADED`
- `WORKFLOW_PROFILE_CHANGED`
- `FINDING_SELECTED`
- `LESION_VALIDATED`
- `LESION_CORRECTED`
- `NEW_LESION_CREATED`
- `REFERENCE_EXAM_CHANGED`
- `DICTATION_STARTED`
- `DICTATION_PARSED`
- `STAGING_RECALCULATED`
- `RESPONSE_RECALCULATED`
- `KEY_IMAGE_TOGGLED`
- `REPORT_SECTION_GENERATED`
- `VALIDATION_RUN`
- `REPORT_SIGNED`

Viewer windows should subscribe to events rather than directly depending on report UI code.

# 28. Keyboard and mouse shortcuts

Maintain consistency with Scientific Mode.

| Shortcut | Clinical action |
| --- | --- |
| A | Accept current finding/segmentation when applicable |
| R | Reject false-positive candidate |
| E | Edit/correct segmentation |
| N | New lesion |
| P | Prompt segmentation |
| C | Center current finding |
| D | Start/stop finding dictation |
| Space | Next relevant finding |
| [ / ] | Previous/next examination or configured reference navigation |
| B | Compare with baseline |
| V | Compare with previous examination |
| F | Flicker comparison |
| O | Overlay comparison |
| Q hold | Temporarily hide mask |
| Ctrl+Enter | Generate/update report draft or selected section, depending on context |

Final sign-off should require an explicit protected action, not a single accidental keystroke.

# 29. Deterministic services versus LLM services

## 29.1 Deterministic services

Implement as normal tested code:

- DICOM metadata extraction,
- SUV computation/display values,
- lesion measurements,
- TMTV,
- lesion counts,
- longitudinal deltas,
- registration transforms,
- lesion matching identifiers,
- PROMISE/miTNM rules,
- RECIP rules,
- RECIST calculations,
- key-image viewer state capture,
- conflict checks based on structured facts.

## 29.2 LLM services

Permitted tasks:

- parse history dictation into candidate structured fields,
- parse prior report into candidate statements,
- map dictated finding language to selected finding IDs,
- rewrite validated facts into professional German/English report prose,
- summarize changes,
- create concise impression wording,
- propose captions,
- identify possible prose conflicts for deterministic validation.

## 29.3 LLM guardrails

Every generation call should receive a **fact package** rather than raw access to the whole application state. The prompt must explicitly instruct the model:

- use only supplied facts,
- never invent missing values,
- preserve laterality and dates exactly,
- do not calculate medical classifications unless the deterministic result is supplied,
- mark unknown facts as unknown/omit them,
- output structured JSON when the next pipeline step is machine parsing.

# 30. Suggested Gemini Pro integration pattern

## 30.1 General rule

Use Gemini Pro as a coding/refactoring assistant and as a bounded text-generation service. Do not ask it to perform a blind monolithic rewrite of the complete application.

## 30.2 Recommended coding workflow

1. Freeze current behavior with tests/screenshots where possible.
2. Identify one module to refactor.
3. Provide Gemini the current module, interfaces, this specification, and tests.
4. Ask for a change plan before code.
5. Require compatibility constraints.
6. Apply changes in small commits.
7. Run automated tests and manual imaging QA.
8. Review geometry/SUV/registration behavior explicitly.
9. Proceed to the next module only after acceptance criteria pass.

## 30.3 Suggested prompt for code refactoring

```text
You are refactoring an existing PSMA PET/CT workstation.
Treat the attached Clinical Reporting Mode specification as the product contract.

Task:
[INSERT ONE CONCRETE TASK]

Before writing code:
1. summarize the current behavior from the supplied source files,
2. identify the smallest set of files/modules that must change,
3. list risks and backward-compatibility constraints,
4. propose an implementation plan,
5. identify tests that should be added or updated.

Hard constraints:
- do not change DICOM geometry/orientation semantics,
- do not change SUV calculations,
- do not change image spacing/origin/direction behavior,
- do not change mask coordinate systems,
- do not overwrite original AI masks,
- do not change registration transforms unless explicitly requested,
- keep Scientific Annotation Mode functional,
- keep business logic separate from monitor-specific UI code,
- all staging/response calculations must remain deterministic,
- LLM output must never become the source of quantitative truth.

After I approve the plan, generate the patch in small, reviewable units.
For each changed unit explain exactly which requirement(s) it implements.
```

## 30.4 Suggested prompt for report-language generation

```text
Generate professional PSMA PET/CT report prose from the supplied structured facts.
Use only the facts provided in the JSON payload.
Do not invent, infer, or recalculate missing clinical facts.
Do not alter dates, laterality, lesion counts, SUV values, sizes, volumes, staging, or response classifications.
If a value is missing, omit it unless the template explicitly requires an 'unknown/not available' statement.
Return only the requested report sections.
Preserve physician-confirmed interpretation exactly in meaning.
```

# 31. Performance and usability targets

Clinical interaction should feel immediate.

Recommended targets where technically feasible:

- finding selection -> synchronized M1/M2 update without perceptible blocking,
- next relevant finding -> prefetch next images/segmentations,
- report section regeneration -> asynchronous and non-blocking,
- autosave after physician action,
- no unnecessary modal confirmation dialogs,
- no waiting for LLM calls before the physician can continue reading images.

LLM/network failure must not block basic diagnostic viewing, segmentation correction, staging calculation, or manual reporting.

# 32. Autosave, auditability, and versioning

Autosave after clinically meaningful actions:

- history confirmation,
- segmentation correction,
- finding validation/rejection,
- new lesion creation,
- staging/response confirmation,
- dictation processing,
- report section edit,
- key-image selection.

Never overwrite:

- original AI segmentation,
- prior signed report,
- previous report draft versions needed for audit,
- raw deterministic measurement records.

Maintain an audit trail with user, timestamp, action, old/new state, and relevant object IDs.

# 33. Safety and failure behavior

## 33.1 Missing prior examination

Fall back to M2 whole-body staging mode. Do not show empty prior panes.

## 33.2 Missing prior report

Hide prior-report template actions rather than presenting disabled clutter.

## 33.3 LLM unavailable

Allow:

- manual history entry,
- manual dictation transcript entry,
- deterministic structured report template,
- staging/response calculation,
- export.

## 33.4 Registration poor

Flag comparison values/visual overlays as potentially unreliable. Do not silently claim longitudinal lesion identity solely from a poor transform.

## 33.5 Incomplete critical review

Signing should be blocked or warned according to configured severity if a staging-changing/new/uncertain finding remains unreviewed.

# 34. Acceptance criteria by workflow

## 34.1 Initial staging

Given validated primary/nodal/metastatic findings, the system:

- displays current imaging on M1,
- uses M2 whole-body context when no prior exists,
- provides candidate miTNM with evidence,
- requires physician confirmation/override,
- generates anatomically organized report text,
- produces concise impression and synoptic summary.

## 34.2 BCR

Given current PSA context and prior therapy, the system:

- supports history dictation,
- displays current/prior comparison when available,
- summarizes recurrence compartments,
- highlights new/staging-changing findings,
- generates BCR-appropriate impression,
- does not force RLT/RECIST fields.

## 34.3 Pre-RLT

The system:

- shows TMTV and disease distribution,
- displays reference tissues,
- supports baseline designation,
- stores the examination as selectable future baseline,
- generates baseline theranostic report language.

## 34.4 Post-RLT

The system:

- defaults M2 to current vs baseline,
- can toggle current vs previous,
- calculates deterministic longitudinal metrics,
- identifies new lesions,
- produces candidate PSMA-response classification where applicable,
- requires physician confirmation,
- creates a concise longitudinal report.

## 34.5 RECIST

Formal RECIST tools remain inactive until CT suitability is confirmed. When active, target-lesion selection and numerical calculations are deterministic and auditable.

# 35. Acceptance criteria for key images

1. Physician can mark/unmark a finding as a key image with one action.
2. Single-study key finding automatically produces at least axial + coronal standardized snapshots.
3. Longitudinal key finding can automatically produce prior/current two-plane comparison.
4. Captured images preserve documented viewer parameters and source references.
5. Key-image captions use validated facts.
6. Images appear in report preview in physician-defined order.
7. Physician can remove/reorder/edit caption without changing lesion data.
8. Report generation does not require manual screenshot capture.

# 36. Acceptance criteria for prior-report-aware generation

1. Prior report can be loaded independently from current report draft.
2. `Use as template` never marks prior statements as current without comparison.
3. Resolved prior findings can be removed or rewritten as resolved.
4. Current new findings are inserted.
5. Numerical values in generated prose match current structured database values.
6. Carryover conflicts are surfaced by the conflict checker.
7. Physician manual edits remain visible and are not silently overwritten by regeneration.

# 37. Minimum report QA checks

Before sign-off verify:

- correct patient/case,
- correct examination date and tracer,
- correct reference examination,
- all critical findings reviewed,
- laterality consistent,
- report numbers match structured values,
- staging matches confirmed deterministic result,
- response matches confirmed deterministic result,
- prior resolved findings not accidentally retained,
- key images correspond to reported findings,
- no placeholder text remains.

# 38. Recommended implementation order

## Phase 1 - Shared workstation shell

- robust three-window layout,
- session/event bus,
- M1 current viewer,
- M2 context viewer,
- M3 READ/ASSESS/REPORT shell.

## Phase 2 - Clinical case profiles

- workflow selector,
- context-sensitive panels,
- relevant-finding queue,
- dynamic M2 layout rules.

## Phase 3 - Prior report and dictation

- prior report ingestion,
- history dictation extraction,
- finding dictation linkage,
- provenance.

## Phase 4 - Deterministic assessment

- PROMISE/miTNM engine integration,
- pre/post-RLT metrics,
- RECIP integration where defined,
- optional gated RECIST.

## Phase 5 - Report composer

- semi-structured template,
- section generation,
- prior-aware diff/update,
- physician editing,
- synoptic summary.

## Phase 6 - Key images

- star selection,
- deterministic snapshot capture,
- longitudinal composites,
- captions,
- Word/PDF integration.

## Phase 7 - Validation and hardening

- conflict checker,
- sign-off workflow,
- audit/versioning,
- offline/failure behavior,
- performance optimization,
- end-to-end clinical usability testing.

# 39. Non-goals for first implementation

Unless already present, do not delay the core clinical workflow for:

- decorative 3D rendering,
- complex longitudinal graphs always visible during reading,
- fully autonomous report signing,
- autonomous treatment recommendation,
- LLM-derived staging without deterministic validation,
- exhaustive manual lesion confirmation in every disseminated case,
- automatic export into every possible institutional system before core workflow validation.

# 40. Final intended user experience

A typical BCR case should feel like:

`open case -> confirm BCR profile -> dictate/confirm history -> review current/prior images -> handle only relevant findings -> dictate interpretation as needed -> confirm miTNM -> generate report -> validate conflicts -> review 3-5 key images -> sign`

A typical post-RLT case should feel like:

`open case -> baseline automatically identified -> M1 current, M2 current vs baseline -> review new/changed/uncertain findings -> toggle previous comparison when needed -> confirm response metrics/classification -> dictate overall interpretation -> generate/update report -> validate -> sign`

The core product philosophy is:

**The physician reads images and makes medical decisions. The software measures, remembers, compares, calculates, structures, and documents. The LLM formulates language from physician-validated facts.**

