# Product Requirements Document: TactileGeo

**AI-Powered Visual-to-Tactile Geometry Conversion Tool**

Version 1.1 · Prepared as a build spec for an AI coding agent (Codex)
Target: Cloud-hosted web application for inclusive STEM classrooms

---

## 0. How to Use This Document (Note to Coding Agent)

This PRD is written so it can be handed directly to a coding agent to scaffold and build the product. It is organized so each numbered section maps to a buildable unit of work. Build in the order of Section 12 (Implementation Roadmap). Where a decision is ambiguous, the agent should default to the choice stated here rather than asking — this doc has already made the calls. Every UI screen, API contract, data shape, and accessibility rule needed to build without further clarification is included below.

---

## 1. Problem Statement

Students with visual impairments in inclusive STEM classrooms struggle with geometry because textbook diagrams and worksheets are inherently visual. Manual tactile-graphic production (embossing, swell-paper tracing, hand-tracing raised images) is slow, expensive, and requires specialist staff — making it unsuited for on-demand, day-to-day classroom use. A teacher with a worksheet in front of them right now has no fast way to give a blind or low-vision student the same diagram their sighted classmates have.

## 2. Proposed Solution

An **offline-first, AI-driven web application** that automatically converts a photographed or scanned visual geometry diagram into a simplified, standardized, tactile-ready SVG image:

- Strips visual noise (shadows, gridlines, scan artifacts)
- Optimizes line geometry for touch, not sight
- Converts text labels into Braille
- Lets the teacher preview and adjust before export
- Outputs an embosser/swell-paper-ready SVG

The system is a **cloud-hosted web application**: teachers access it from any browser with an internet connection, images are uploaded to a hosted backend for processing, and no local install or on-prem server is required. Uploaded worksheets and generated previews are treated as transient, session-scoped data (see Section 10.1) even though the infrastructure itself is cloud-based — they are not retained as part of a persistent user account/history in v1.

## 3. Target Users & Personas

| Persona | Role | Needs |
|---|---|---|
| **Teacher (primary user)** | Classroom teacher of visually impaired (TVI) or inclusive-classroom generalist | Fast upload → preview → export flow; large touch targets; minimal training required; works without internet |
| **Student (indirect beneficiary)** | Blind / low-vision student | Never directly uses the app; receives the physical tactile output |
| **IT/School admin (secondary)** | Sets up the local server | Simple install, no cloud account, no recurring cost |

The teacher is the only user of the *software*. Design every screen for a sighted adult under time pressure in a classroom, not for the end student.

## 4. Scope

### 4.1 In Scope (v1)
- Cloud-hosted web app: FastAPI backend (containerized, deployed to a cloud host) + React frontend (static hosting/CDN), accessed over the public internet via HTTPS
- Image upload (.png, .jpg, .jpeg) of a geometry worksheet/diagram
- CV pipeline: adaptive thresholding → morphological filtering → vectorization (Hough + contours) → optional U-Net refinement for hand-drawn shapes
- OCR (EasyOCR or Tesseract; may run server-side in the cloud backend, or call a cloud OCR API if it improves accuracy — see Section 6.3)
- Braille conversion via **Liblouis** (Unified English Braille, Grade 1/2, with correct capital and number indicators)
- Live tactile preview with teacher-adjustable sliders (edge sensitivity, stroke weight, label placement)
- SVG export, sized and styled for embosser/swell-paper output
- BANA/ICEB-aligned tactile graphics rules (stroke width, textures instead of color, spatial kerning/collision avoidance)
- Full keyboard and screen-reader accessible teacher-facing UI (see Section 8)
- Transport security: all traffic over HTTPS/TLS; uploaded images encrypted in transit and at rest for the (short) duration they exist server-side

### 4.2 Out of Scope (v1)
- Persistent user accounts, login, or saved history across devices/sessions (see Section 10.1 — data is session-scoped even though hosting is cloud-based)
- Multi-tenant school/district admin console
- Mobile native apps (a responsive web UI is sufficient)
- Direct embosser driver integration (v1 exports a file the teacher sends to their existing embosser software/printer)
- Non-English Braille codes (Bharati Braille, Grade 2 contractions for other languages) — flagged as a v2 candidate, architecture should not preclude it (Liblouis supports this later)
- Real-time collaborative editing between multiple teachers
- Offline/no-connectivity mode (v1 requires an active internet connection; see Section 13 for the connectivity trade-off this introduces)

## 5. System Architecture

**Pattern:** Cloud-hosted client-server, accessed over the public internet.

```
┌─────────────────────────────┐  HTTPS   ┌──────────────────────────────┐        ┌──────────────────┐
│  Frontend (Browser)          │◄────────►│  Backend (Cloud host)         │◄──────►│  Cloud DB / Store  │
│  React + Vite (static host/  │          │  Fg (OpenCV)│  astAPI (Python), containerized│       │  Postgres (metadata)│
│  CDN)                         │          │  - Image preprocessin      │  Object storage    │
│  - Upload UI                 │          │  - Vectorization (cv2/Hough)   │        │  (images/SVGs,      │
│  - Live tactile preview      │          │  - U-Net refinement (PyTorch)  │        │   short TTL)         │
│  - Adjustment sliders        │          │  - OCR (EasyOCR/Tesseract, or  │        └──────────────────┘
│  - Export button              │          │    cloud OCR API)              │
└─────────────────────────────┘          │  - Braille (Liblouis)          │
                                          │  - SVG compiler + rules engine │
                                          └──────────────────────────────┘
```

- Frontend is deployed as a static build (e.g. Vercel/Netlify/S3+CDN); backend is deployed as a containerized FastAPI service (e.g. on a managed container platform) with autoscaling for classroom-hours traffic spikes.
- **Cloud DB / storage** (e.g. managed Postgres for session metadata, object storage such as S3-compatible storage for the uploaded image and generated SVGs) replaces the local-filesystem approach from the original offline design — but see Section 10.1: this is used as *transient* storage with an automatic expiry/deletion policy, not as permanent per-teacher history, per the scope decision in Section 4.2.
- All traffic is HTTPS-only; no plaintext HTTP endpoint should exist even for local dev beyond `localhost`.
- Standard cloud web-app hardening applies: rate limiting on upload/process endpoints, file-type/size validation, and CORS locked to the known frontend origin(s).

## 6. Processing Pipeline (Detailed)

### 6.1 Image Pre-processing & Denoising (OpenCV)
- **Adaptive Thresholding:** Localized Gaussian-window thresholding (`cv2.adaptiveThreshold`, `ADAPTIVE_THRESH_GAUSSIAN_C`) to keep lines intact under uneven classroom lighting/shadows from phone-camera scans.
- **Morphological Filtering:** Opening + erosion with a 3×3 structuring element to strip gridlines, creases, text-box borders, and scan artifacts, without destroying thin diagram lines.

### 6.2 Structural Vectorization & Feature Extraction
- **Primitive Detection:** Hough Line Transform for straight segments; `cv2.findContours` for closed polygon/circle boundaries.
- **Deep Learning Refinement:** A lightweight U-Net model runs concurrently on figures where edge detection is ambiguous (hand-drawn or complex shapes), producing a validated pixel mask of "real" geometry vs. artistic noise. Treat this as a pluggable module — ship v1 with a small pretrained/fine-tuned model, and design the interface so it can be swapped/retrained later without touching the rest of the pipeline.
### 6.2A AI Diagram Understanding Layer

Before vectorization, the system should analyze the uploaded image semantically.

The AI layer should:

- Identify whether the image contains a geometry diagram, graph, chart,
  scientific illustration, flow diagram, or unsupported content.
- Detect important geometric primitives such as points, lines, circles,
  arcs, polygons, axes, arrows and boundaries.
- Identify spatial relationships between detected elements.
- Identify labels and associate them with the corresponding geometric
  elements.
- Distinguish meaningful diagram geometry from decorative/background content.
- Assign confidence scores to detected elements.
- Flag ambiguous elements for teacher review rather than silently guessing.

The semantic representation should be converted into an intermediate
structured representation before generating the tactile SVG.
### 6.3 OCR + Braille Synthesis
- **Text Extraction:** EasyOCR or Tesseract running server-side by default; since the system is cloud-hosted, a managed cloud OCR API (e.g. Google Vision, AWS Textract, Azure OCR) may be swapped in later for higher accuracy — keep the OCR call behind a single internal interface/adapter so the provider can change without touching the rest of the pipeline.
- **Spatial Remapping:** Map each detected text bounding box to its nearest vertex/edge coordinate.
- **Braille Translation:** Use **Liblouis** (Python bindings) for Unified English Braille — do not hand-roll a dictionary substitution, since Liblouis correctly handles:
  - Capital indicator (Dot 6, ⠠) before capitalized letters
  - Number indicator (Dots 3-4-5-6, ⠼) before numeric sequences
  - Grade 2 contractions where applicable
- **Collision Logic:** Offset each Braille marker a minimum safe distance from any line or another marker so raised dots don't touch or blend with raised lines (see 6.4 spatial kerning).

### 6.4 Tactile Output Rules (Output Layer)
- **Format:** SVG (scalable, embosser/swell-paper compatible, no quality loss when resized).
- **Deterministic stroke width:** All lines normalized to ~1.5–2pt, matched to fingertip / Merkel-cell tactile sensitivity thresholds. This is a hard rule, not a suggestion — no line in the final export may fall outside this range unless the teacher explicitly overrides via the stroke-weight slider.
- **Texture instead of color:** Any fill/color/gradient in the source image must be replaced by an SVG pattern (cross-hatch, dashed, dotted) from a small fixed library of tactile-safe patterns — never rendered as a flat color fill in the export.
- **Spatial kerning check:** Before finalizing export, the compiler scans for parallel lines/contours closer than the minimum tactile clearance and either nudges them apart or flags them to the teacher as "may be hard to distinguish by touch," with a one-click "auto-fix spacing" action.
- **Compliance:** Rules should be traceable to **BANA** (Braille Authority of North America) and **ICEB** tactile graphics guidelines — keep a `TACTILE_RULES` config object in code with named constants (not magic numbers) so each rule can be cited/audited.
### 6.4A Tactile Simplification Engine

The system must not perform a pixel-for-pixel conversion of the source image.

Instead, it should generate a tactile representation optimized for
spatial understanding through touch.

The simplification engine should:

1. Remove decorative visual information.
2. Remove unnecessary background elements.
3. Remove shadows, gradients and photographic details.
4. Simplify complex curves and boundaries.
5. Consolidate visually equivalent elements.
6. Separate elements that would become tactilely ambiguous.
7. Preserve topologically important relationships.
8. Preserve labels and their association with diagram elements.
9. Convert visual distinctions such as color into tactile patterns.
10. Maintain sufficient spacing between independent tactile features.

The objective is semantic preservation rather than visual similarity.

### 6.5 Tactile Quality Assurance Engine

Every generated diagram must pass an automated tactile-readiness
validation stage before export.

The validator should evaluate:

- Stroke width
- Minimum feature spacing
- Braille-to-line clearance
- Braille-to-Braille spacing
- Label placement
- Element separation
- Closed/open geometry integrity
- Page boundary clearance
- Excessive geometric complexity
- Unsupported visual elements
- Potentially ambiguous tactile regions

The system should generate:

- Overall tactile readiness score
- Individual validation results
- Warnings
- Critical errors
- Recommended corrections

A diagram containing critical tactile violations must not be silently
exported as "print ready".

## 7. User Workflow (4 Steps)

1. **Upload** — Teacher opens the local URL, drags/uploads a photo of a worksheet.
2. **Review & Adjust** — Live tactile preview renders side-by-side with the original. Sliders let the teacher adjust edge sensitivity, stroke weight, and label placement in real time.
3. **Export** — One click ("Export for Embosser") generates a print-ready SVG and a local download link.
4. **Print** — Teacher sends the file to a tactile embosser or prints it on swell paper via a thermal fuser (external to this app).

## 8. Accessibility Requirements (Software Rules)

Accessibility applies on two levels: (A) the physical **tactile output**, already covered in Section 6.4, and (B) the **software itself**, which must be usable by a diverse range of teachers, including teachers who are themselves low-vision or motor-impaired. Both are mandatory, not optional polish.

### 8.1 Standard to Target
- **WCAG 2.2, Level AA** compliance for the entire teacher-facing web app, as a hard requirement.
- Semantic HTML first; ARIA only to fill genuine gaps, never as a substitute for correct native elements (e.g., use `<button>`, not `<div role="button">`, unless building a genuinely custom widget).

### 8.2 Concrete Rules for the Coding Agent to Implement
- **Color contrast:** All text vs. background must meet at least 4.5:1 contrast ratio (3:1 for large text ≥18px bold/24px regular). In the black/white Notion-style theme (Section 9), this is naturally satisfiable — verify every gray-on-white/gray-on-black pairing against this ratio and reject any that fail.
- **Never color alone:** Every state conveyed by color (error, success, "selected," slider value) must also be conveyed by an icon, label, or pattern — because a low-vision teacher may not perceive the color at all.
- **Full keyboard operability:** Every action reachable by mouse (upload, slider adjustment, export, undo) must be reachable via keyboard alone, in a logical tab order, with visible focus rings (never `outline: none` without a replacement focus style).
- **Sliders:** Implement as native `<input type="range">` or ARIA `slider` role with `aria-valuenow/min/max/text`, and make sure arrow keys work for fine adjustment. Provide a numeric text-input alternative next to every slider so precise values can be typed instead of dragged.
- **Screen reader support:** All images (including the live preview canvas) need meaningful `alt`/`aria-label` text. The live preview updates should be announced via an `aria-live="polite"` region (e.g., "Preview updated: edge sensitivity 65%") rather than silently repainting a canvas.
- **Text resizing:** Layout must not break when the browser text size is increased to 200%.
- **Motion:** Respect `prefers-reduced-motion` — any preview transition/animation must have a reduced/no-motion fallback.
- **Error messaging:** Upload/processing errors must be specific, in plain language, and programmatically associated with the relevant control (`aria-describedby`), not just a floating toast.
- **Target size:** Interactive elements (buttons, slider handles, upload zone) at minimum 44×44px touch/click target, per WCAG 2.2 SC 2.5.8.
- **No time limits:** No auto-dismissing critical messages; a processing job never silently times out without a clear, actionable message.
- **Testing requirement:** Ship with automated accessibility testing (e.g., `axe-core` / `jest-axe` in CI, or Playwright + axe) covering every screen, and treat any AA violation as a build-blocking bug, not a "nice to fix later."

### 8.3 Note on the Difference Between the Two Accessibility Layers
Do not conflate these two: the *software's* accessibility (Section 8) ensures a teacher of any ability can operate the tool; the *output's* accessibility (Section 6.4, BANA/ICEB tactile rules) ensures the printed diagram is actually readable by touch. Both must be treated as first-class, testable requirements — not just "make it accessible" as a vague goal.

## 9. UI / UX Specification — Notion-Style Black & White

### 9.1 Design Philosophy
Minimal, calm, high-contrast, content-first — modeled on Notion's aesthetic: generous whitespace, black text on white (and white on near-black for dark mode), thin 1px borders instead of heavy shadows, small quiet icons, no gratuitous color. The UI should feel like a clean document/tool, not a flashy dashboard. This also directly supports the accessibility requirements above (high contrast is native to this palette).

### 9.2 Color Palette

| Token | Light Mode | Dark Mode | Usage |
|---|---|---|---|
| `--bg-primary` | `#FFFFFF` | `#191919` | Page background |
| `--bg-secondary` | `#F7F7F5` | `#202020` | Panels, cards, sidebar |
| `--bg-hover` | `#EFEFED` | `#2A2A2A` | Hover states |
| `--border` | `#E9E9E7` | `#2F2F2F` | 1px dividers/borders |
| `--text-primary` | `#191919` | `#EDEDED` | Primary text |
| `--text-secondary` | `#787774` | `#9B9B9B` | Secondary/meta text |
| `--text-disabled` | `#B4B4B0` | `#5A5A5A` | Disabled labels |
| `--accent` | `#191919` | `#FFFFFF` | Primary buttons (inverted, not a "color" — stays black/white) |
| `--focus-ring` | `#2383E2` (blue, WCAG-safe) | `#5EA5F0` | Focus outline only — the *one* deliberate color, used solely for focus/selection states so keyboard users always have a visible, high-contrast, non-monochrome cue |
| `--success` | `#2F9E44` | `#4FCE5D` | Success text/icon, always paired with a checkmark icon |
| `--error` | `#E03131` | `#FF6B6B` | Error text/icon, always paired with a warning icon |

Rule: color is used *only* for focus, success, and error — everything else is grayscale. This is a deliberate constraint, not an oversight; confirm every new component against it before adding color.

### 9.3 Typography
- **Font:** System UI stack for fast load and zero font-licensing/CDN overhead: `-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif`. A CDN web font may be added later if desired since the app now assumes internet access, but system fonts remain the recommended default for speed and the minimal Notion-style aesthetic.
- **Scale:** 14px base body, 1.5 line-height. Headings: 24px / 18px / 16px, weight 600, tight line-height (1.25).
- **Monospace** (for any technical/debug info, e.g. file names, coordinates): system mono stack (`ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`).

### 9.4 Layout & Components

**Overall shell:**
- Left sidebar (240px, collapsible), `--bg-secondary`, containing: app logo/name, "New Conversion," recent session history (local only), settings.
- Main content area, `--bg-primary`, max-width content column (~960px) centered, generous padding (32–48px).
- No heavy drop shadows; use 1px `--border` to separate regions. Rounded corners: 6–8px, consistently, on all cards/buttons/inputs.

**Screen 1 — Upload:**
- Centered dashed-border drop zone (`--border`, 2px dashed), large upload icon, "Drag a worksheet photo here, or click to upload" text, "Supports PNG, JPG" caption in `--text-secondary`.
- Below: small privacy note ("Your upload is used only to generate this preview and is automatically deleted after your session ends — see our privacy policy") — since data now travels over the internet to a cloud backend, the UI must be explicit about retention rather than implying full on-device privacy.

**Screen 2 — Review & Adjust (core screen):**
- Dual-pane layout: left pane = original image (labeled "Original"), right pane = live tactile SVG preview (labeled "Tactile Preview"), both in bordered cards with a thin divider between them. Stack vertically on narrow/mobile widths.
- Right-hand control rail (or bottom panel on mobile) with three labeled sliders, each as: label → native range slider → adjacent numeric input → live value announced via `aria-live`:
  1. **Edge Sensitivity** (0–100)
  2. **Stroke Weight** (1.5–2.0 pt, matches Section 6.4 hard bounds — slider cannot exceed this range without an explicit "Advanced / override BANA defaults" toggle, off by default)
  3. **Label Placement offset** (fine nudge, small/medium/large presets + custom)
- A quiet warning banner (icon + text, not color-only) appears if the spatial-kerning check detects lines too close together, with an "Auto-fix spacing" button.
- Sticky top-right primary button: **Export for Embosser** (black filled button, white text in light mode / inverted in dark mode — the one "loud" element on the page, deliberately, since it's the primary action).

**Screen 3 — Export confirmation:**
- Simple confirmation card: filename, file size, "Open folder" / "Download again" actions, and a "Start new conversion" link back to Screen 1.

### 9.5 States & Feedback
- Every async action (upload, processing, export) shows an inline, textual, low-motion progress indicator — a simple determinate/indeterminate bar in `--text-secondary`, with a text status ("Analyzing lines…", "Converting labels to Braille…") — not just a spinning icon, so screen readers and low-vision users both get the same information.
- Empty/error states always paired with a next action ("Try a clearer photo," "Retry").

### 9.6 Dark Mode
- Implement via a `prefers-color-scheme` default plus a manual toggle in settings (stored in local storage / a local config file, not a server account). All tokens above already have dark-mode values — no separate design pass needed, just apply the token table.

## 10. Data Model (Cloud DB + Object Storage, Session-Scoped)

```
Session
 ├── session_id (uuid)
 ├── created_at
 ├── expires_at              # created_at + TTL, enforced by cleanup job
 ├── original_image_url      # object storage, not public
 ├── processing_params { edge_sensitivity, stroke_weight, label_offset }
 ├── detected_shapes []      # vector primitives with type, coordinates
 ├── detected_labels []      # {text, braille, bbox, anchor_vertex}
 ├── preview_svg_url
 └── export_svg_url (nullable, set on export)
```

### 10.1 Storage & Retention Policy (important — read before building)
Even though the infra is cloud-hosted (per the scope decision in Section 4), sessions are **not** turned into permanent user history in v1:
- Session metadata lives in a managed cloud DB (e.g. Postgres); images/SVGs live in object storage (e.g. S3-compatible bucket), never inline in the DB.
- Every session row gets a **TTL** (e.g. 24 hours after last activity) after which a scheduled cleanup job deletes both the object-storage files and the DB row. This preserves the original "don't retain student data" intent of the project even though the app is now online.
- No login/account is required to use the tool in v1 — a session is identified only by its `session_id` (e.g. stored in a signed cookie or returned to the client), so there's no persistent link between a session and an identifiable teacher/school unless auth is added later.
- If accounts/persistent history are added in a future version, this TTL-based model should be treated as the default retention for anonymous/guest sessions, with an explicit opt-in required to keep anything longer.

## 11. API Contract (FastAPI, cloud-hosted, HTTPS only)

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/sessions` | Upload image, create a session, return `session_id` |
| `POST` | `/api/sessions/{id}/process` | Run the CV/OCR/Braille pipeline with given params, return preview SVG + detected shape/label data |
| `PATCH` | `/api/sessions/{id}/params` | Update sliders (edge sensitivity, stroke weight, label offset), re-run only the affected stage, return updated preview |
| `POST` | `/api/sessions/{id}/export` | Finalize tactile rules (stroke bounds, kerning check, texture fills), produce the export-ready SVG, return a download URL |
| `GET` | `/api/sessions/{id}` | Fetch session state (for reload/history) |
| `GET` | `/api/health` | Local health check, used by the frontend to confirm the backend is reachable |

All responses are JSON except the SVG download itself. All endpoints require HTTPS, validate uploaded file type/size before processing, and are rate-limited per client IP/session to prevent abuse. CORS should be locked to the deployed frontend origin(s) — do not use a wildcard `*` origin in production.

## 12. Implementation Roadmap

**Milestone 1 — Core Pipeline**
- FastAPI scaffold + `/api/sessions` upload endpoint
- OpenCV adaptive thresholding + morphological filtering
- Raw SVG path conversion from contours/Hough lines
- Basic React upload screen (Screen 1) wired to the backend

**Milestone 2 — OCR & Braille**
- Integrate EasyOCR/Tesseract
- Spatial mapping of labels to vertices
- Liblouis integration for UEB conversion (capitals, numbers)
- Collision offset logic for Braille markers

**Milestone 3 — Editor & UX Polish**
- Full Review & Adjust screen (Screen 2) with live sliders and dual-pane preview
- Tactile rules engine: deterministic stroke width, texture-fill library, spatial-kerning auto-fix
- Export flow (Screen 3) and download handling
- Full accessibility pass: keyboard nav, `axe-core` CI tests, screen-reader labels, contrast audit against Section 9.2 tokens
- Dark mode toggle
- U-Net refinement path for complex/hand-drawn shapes (can slip to a fast-follow v1.1 if time-constrained — flag explicitly rather than silently dropping)

## 13. Non-Functional Requirements

- **Connectivity requirement:** v1 requires an active internet connection to use the tool — this is a deliberate trade-off versus the original offline-first concept, and should be called out clearly in any submission/pitch materials, since low-connectivity rural schools (a use case named in Section 1) will not be able to use this version without internet access. Flag this explicitly to stakeholders rather than letting it pass silently.
- **Performance:** preview re-render on slider change should complete in <1–2s round-trip over a typical school internet connection; full pipeline (upload → first preview) target <8s including network latency.
- **Security & privacy:** all traffic over HTTPS/TLS; images encrypted at rest in object storage; session data auto-deleted per the TTL policy in Section 10.1; no analytics/telemetry beyond basic operational logging (errors, latency) with no student-identifying content in logs.
- **Scalability:** backend should autoscale (or be easy to scale manually) to handle concurrent classroom usage across multiple schools; object storage and DB choices should be managed/serverless where possible to avoid ops burden on a small team.
- **Cross-browser:** modern evergreen browsers (Chrome, Firefox, Edge, Safari) — no IE support needed.

## 14. Key Differentiators (context for the coding agent, not a build task)

- **Cloud-hosted, zero-install:** teachers can use it from any browser with no local setup — trades away the original offline/rural-connectivity guarantee (see Section 13) in exchange for easier deployment and updates.
- **Teacher-in-the-loop:** AI automates ~90% of conversion, but sliders keep the teacher in control for each student's specific tactile-reading level.
- **Standards-compliant:** BANA & ICEB alignment means output is pedagogically usable, not just a simplified image.
- **Scalable & low-cost:** SVG output scales cleanly from index-card to whiteboard size without quality loss.

---

*End of PRD. Build order: Section 5 (architecture) → Section 6 (pipeline) → Section 10–11 (data/API) → Section 9 (UI) → Section 8 (accessibility, woven in throughout, not bolted on at the end) → Section 12 (roadmap milestones), with CI accessibility and offline-network tests wired in from Milestone 1 onward.*


# TactileGeo — Gemini Dual-Architecture Integration

## Purpose

TactileGeo currently contains a deterministic/structured visual-to-tactile
processing pipeline.

A previous Gemini-based prototype also exists separately.

The objective of this phase is NOT to replace the existing TactileGeo
pipeline.

Instead, the existing Gemini prototype will be studied, reconstructed where
necessary, and integrated as an optional second interpretation path
(Model B) using a new provider API key.

The existing TactileGeo pipeline remains Model A and remains the controlled
source for final tactile geometry and SVG generation.

---

# 1. Existing Architecture — Model A

Model A is the current TactileGeo processing pipeline.

High-level flow:

Image Upload
    ↓
Image Preprocessing
    ↓
Geometry / Shape Detection
    ↓
OCR
    ↓
Semantic Geometry
    ↓
Label Association
    ↓
Braille Translation
    ↓
Tactile Simplification
    ↓
Braille Layout
    ↓
Tactile QA
    ↓
SVG Generation

Model A is responsible for:
- geometric accuracy
- coordinates
- detected shapes
- detected lines
- OCR results
- semantic relationships
- Braille conversion
- tactile simplification
- spatial layout
- tactile QA
- final SVG generation

Model A must remain functional independently of Model B.

---

# 2. Gemini Prototype — Model B

A previous Gemini prototype exists separately from the current
production TactileGeo implementation.

The prototype must be inspected before integration.

Do not assume that its implementation is correct or production-ready.

The following must be extracted from the prototype:

- Gemini model used
- Gemini API invocation method
- prompt structure
- system instructions
- image encoding method
- request payload
- response format
- structured output/schema, if any
- detected entities
- detected relationships
- detected text
- detected geometry
- confidence information
- error handling
- API key handling
- frontend/backend responsibilities
- dependencies
- limitations

The prototype is a reference implementation, not automatically a production
component.

---

# 3. Provider API Key

The provider API key used for the integrated implementation MUST be different
from any key that may have existed in the original prototype.

The key must NEVER be hardcoded in:
- React source code
- TypeScript source
- Python source
- Git history
- PROJECT.md
- README.md
- screenshots
- client-side JavaScript

The key must be stored using environment configuration.

For example:

OPENROUTER_API_KEY=<secret>

The frontend must never receive the provider API key.

All provider API requests must originate from the backend.

---

# 4. Dual Architecture

The intended architecture is:

                    Worksheet Image
                           |
                           |
                 Image Preprocessing
                           |
             +-------------+-------------+
             |                           |
             v                           v
       MODEL A                       MODEL B
    TactileGeo CV                OpenRouter API
       Pipeline                    Interpretation
             |                           |
             |                           |
             v                           v
    Structured Geometry          AI Interpretation
             |                           |
             +-------------+-------------+
                           |
                           v
                    Normalization Layer
                           |
                           v
                   Comparison / Fusion
                           |
                           v
                  Semantic Representation
                           |
                           v
                 Tactile Optimization
                           |
                           v
                    Braille + QA
                           |
                           v
                      SVG Output

Model B is optional.

If the provider is unavailable, Model A must continue to work.

---

# 5. Model B Responsibilities

Model B should primarily provide higher-level visual interpretation.

Potential outputs include:

- diagram type
- geometric entities
- labels
- spatial relationships
- semantic interpretation
- contextual understanding
- ambiguous or difficult-to-detect relationships
- possible missing entities
- possible corrections to Model A

Model B must NOT directly generate the final tactile SVG.

Model B output must pass through validation and normalization.

---

# 6. Model A Responsibilities

Model A remains responsible for deterministic geometric processing.

The following should not be replaced merely because Model B can describe them:

- exact coordinates
- geometric primitives
- line positions
- shape boundaries
- tactile stroke generation
- Braille positioning
- collision avoidance
- tactile spacing
- QA constraints
- SVG generation

---

# 7. Normalized Representation

Model A and Model B should not communicate using arbitrary independent
structures.

Both should be converted into a common internal representation.

Example:

{
  "elements": [
    {
      "id": "shape_001",
      "type": "triangle",
      "geometry": {},
      "label": "ABC",
      "confidence": 0.94,
      "source": "model_a"
    }
  ],
  "relationships": [
    {
      "type": "parallel",
      "source": "line_001",
      "target": "line_002",
      "confidence": 0.88,
      "source": "model_b"
    }
  ]
}

The exact schema must be determined after inspecting the existing code.

Do NOT introduce this exact schema blindly.

---

# 8. Comparison / Fusion

Model A and Model B outputs must be compared.

Possible cases:

### Agreement

Model A:
    Triangle ABC

Model B:
    Triangle ABC

→ Increased confidence.

### Partial agreement

Model A:
    Triangle ABC

Model B:
    Triangle ABC + angle marker

→ Preserve both if geometrically valid.

### Disagreement

Model A:
    Triangle

Model B:
    Quadrilateral

→ Flag for review rather than silently choosing one.

### Model B provider unavailable

Model B:
    unavailable

→ Continue using Model A.

### Model A unavailable

Model B:
    available

→ Do NOT automatically generate final tactile geometry unless the
implementation explicitly validates that the required geometric data exists.

---

# 9. Teacher-in-the-Loop

The teacher remains the final reviewer.

The system should expose model disagreement or low-confidence interpretation
rather than silently making potentially incorrect changes.

The UI may show:

- Model A interpretation
- Model B interpretation
- agreement/disagreement
- confidence
- suggested interpretation
- teacher correction

Teacher corrections should be applied to the canonical semantic representation,
not permanently modify either model.

---

# 10. Failure Isolation

Model B must be treated as an optional external dependency.

Possible failures:

- API timeout
- invalid API key
- quota exceeded
- rate limit
- network failure
- malformed provider response
- unexpected schema
- safety/filter response
- unavailable model

None of these failures should prevent Model A from functioning.

---

# 11. Security

Never expose OPENROUTER_API_KEY to the browser.

The architecture must be:

Frontend
   ↓
TactileGeo Backend
   ↓
OpenRouter API

NOT:

Frontend
   ↓
OpenRouter API

API keys must be loaded through environment variables or the deployment
platform's secret-management system.

---

# 12. Performance

Model B should not unnecessarily duplicate expensive processing already
performed by Model A.

The integration should investigate:

- image resizing
- request size
- prompt size
- response size
- model latency
- timeout
- caching
- whether Model B should run synchronously or asynchronously
- whether Model B should be enabled only when needed

The existing TactileGeo performance targets should not be degraded without
measurement.

---

# 13. Observability

For development/debugging, record:

- Model A execution time
- Model B execution time
- provider success/failure
- response validation status
- agreement/disagreement
- fusion result
- teacher correction

Do not log:
- API keys
- sensitive credentials
- unnecessary image data
- raw confidential provider responses in production

---

# 14. Testing

The Model B integration must have independent tests.

Required categories:

1. provider API request construction
2. API response parsing
3. malformed response handling
4. timeout handling
5. invalid API key handling
6. rate-limit handling
7. Model A-only operation
8. Model B-only interpretation
9. Model A + Model B agreement
10. Model A + Model B disagreement
11. teacher correction after disagreement
12. final tactile output remains valid
13. regression tests for existing pipeline

Existing Model A tests must continue to pass.

---

# 15. Integration Principle

The Model B integration must be additive.

DO NOT:

- replace Model A
- rewrite the existing CV pipeline
- make Model B mandatory
- move provider API calls to the frontend
- hardcode API keys
- allow unvalidated Model B output to directly control SVG generation
- remove existing QA
- remove teacher review
- duplicate existing services unnecessarily

The desired result is:

Model A = geometric/control pipeline

Model B = semantic/AI interpretation

Fusion = controlled reconciliation

Teacher = final reviewer

Tactile engine = final authority for tactile output

---

# 16. Development Rule

Before modifying code:

1. Inspect current TactileGeo repository.
2. Inspect Gemini prototype.
3. Map both architectures.
4. Identify reusable prototype components.
5. Identify incompatible components.
6. Identify dependencies.
7. Identify security risks.
8. Identify API integration points.
9. Propose normalized schemas.
10. Propose fusion strategy.
11. Propose test strategy.

No implementation should begin until this analysis is complete.
---

# 17. Implemented Architecture (as built)

This section describes what is actually implemented, as distinct from the
sections above, which state requirements.

## Model A — deterministic geometry authority

`backend/app/services/pipeline.py` is the only producer of the geometry that is
embossed on the page. It is deterministic, structured, and independent of any
external service:

    image -> OpenCV contour/Hough detection -> OCR -> semantic normalisation
          -> simplified geometry -> tactile QA -> sanitised SVG

Model A owns exact pixel coordinates, line endpoints, radii, Braille placement,
stroke widths, printable-area checks, the final SVG, and the pass/fail decision
on whether a page is print-ready. Model A runs with no network access and no
credentials. It is not modified by anything described in this section.

## Model B — vision-language interpretation layer (optional, advisory)

Model B adds interpretation, not geometry:

    image -> prepared frame -> OpenRouter (one call, structured output)
          -> parse -> validate against the Model B contract -> normalise
          -> advisory regions, relationships, text, uncertainty

Model B may contribute: entity suggestions, geometry *type* suggestions,
normalised bounding regions, labels, relationships, diagram relations such as
tangency or right-angle markers, contextual interpretation, uncertainty, and
confidence with provenance.

Model B must never become the authority for exact pixel coordinates, line
endpoints, radii, angles, final tactile geometry, Braille placement, stroke
width, printable area, or the final SVG. Model B returns **regions and types
only**; it deliberately emits no point, radius, centre, or angle, so there is
no value it could supply that would be mistaken for a measurement.

## The boundary

| Concern | Authority |
| --- | --- |
| Exact coordinates, endpoints, radii, angles | Model A |
| Braille, stroke width, printable area, final SVG | Model A |
| Blocking QA and export decision | Model A |
| Relationship and contextual interpretation | Model B (advisory) |
| Uncertainty and ambiguity reporting | Model B (advisory) |
| Region/type suggestions for teacher review | Model B (advisory) |

Reconciliation is read-only and lives in `backend/app/model_b/fusion.py`. It
produces agreements, candidate additions (each requiring teacher approval), and
explicit disagreements. It cannot mutate Model A geometry, ids, QA severity, or
export state, and a test asserts those invariants on every fusion.

## Structured output

The response contract is defined once, as Pydantic models in
`backend/app/model_b/schema.py`. The JSON Schema is *derived* from those models by
`backend/app/model_b/json_schema.py`, so there is exactly one source of truth and
no second competing schema. The schema is sent on every request via
`response_format.json_schema` with `strict: true`.

Responses are rejected, not repaired, when they violate the contract: malformed
JSON, a missing required field, an invalid enum value, an impossible bounding
region, or an invalid confidence band all fail the job with a typed error. The
raw model output is never allowed to reach geometry or the browser.

## Security

`OPENROUTER_API_KEY` is read from the backend environment only. It is never sent
to the browser, never written to source, tests, documentation, prompts, or logs,
and never included in an error message. The frontend learns only whether Model B
is usable, via `GET /api/model-b/status`, which returns a boolean, a
human-readable reason, the provider name, and the configured model or router id
— never a key or quota information.

A credential must never be stored in a `.env` file that reaches the frontend, or
in a `VITE_*` variable: Vite inlines those into the shipped bundle, which would
publish the key. `backend/tests/test_model_b_live_boundary.py` fails the build if
a credential-shaped literal or a Vite-exposed key read appears anywhere in
`frontend/src`.

The supported architecture is:

    Frontend  ->  TactileGeo backend  ->  OpenRouter API

The frontend never calls the provider directly. The model or router is configurable
through `OPENROUTER_MODEL`; no model name is hardcoded in business logic. The
default is `openrouter/free`, which selects a different model on each call, so
the configured id and the model that actually served a request are recorded and
reported separately — a result is always attributable to a real model.

## Failure behaviour

Model B is an optional external dependency. Every failure mode — missing key,
invalid key, authentication failure, permission denied, quota exceeded, rate
limit, timeout, network error, provider unavailable, malformed output, schema
validation failure, empty result — is mapped to a distinct typed error and an
HTTP status the teacher can act on. Retryability is reported explicitly so the
UI can offer a retry only when retrying can help.

**A Model B failure never prevents Model A from functioning.** Model A requires
no key, no network, and no Model B code path. `MODEL_B_ENABLED` defaults to
false, and a missing or unusable key degrades to an unavailable advisory panel
rather than an error state.

---

# 18. Measured Status and Known Gaps

Recorded from the implementation as built. Timings are from an idle Windows
workstation running the backend venv directly, using the generated
`triangle_worksheet` fixture.

## Performance: the section 13 target is not met

Section 13 sets a target of under 8 seconds for upload to first preview,
including network latency. Measured Model A execution time:

| Run | Wall clock |
| --- | --- |
| Cold start (first request, includes EasyOCR/Torch model load) | ~42 s |
| Steady state (requests 2-5) | ~15-18 s |

Steady state is roughly **2x over the 8 s target**, and cold start is over 5x
over. This is a real gap, not a measurement artifact: the figures were taken
with no other load on the machine, and a full 442-test backend run completed in
4 minutes on the same host. The section 13 Model A execution-time logging added
in this implementation is what made the gap measurable; before it, the cost was
invisible.

Optimising this is a separate piece of work. The likely candidates, in rough
order of expected payoff, are:

1. Loading the OCR model once at process start rather than per request, which
   should remove most of the cold-start cost.
2. Caching the compiled regex tables and lookup structures in the Braille and
   semantic-normalisation services, which are rebuilt per call.
3. Reducing redundant geometry passes in `build_full_analysis`, which currently
   recomputes derived structures across stage boundaries.

Until that work is done, the 8 s figure in section 13 should be read as a target
rather than a description of current behaviour, and stakeholder-facing material
should not claim it.

## Section 13 observability coverage

| Required signal | Status |
| --- | --- |
| Model A execution time | Recorded in `app/api/sessions.py` and `app/api/demo.py` |
| Model B execution time | Recorded in `app/api/model_b.py` |
| Provider success/failure | Recorded, with HTTP status and typed error, never the message body |
| Response validation status | Recorded, plus the count of validation warnings |
| Agreement/disagreement | Recorded, plus fusion candidate and review counts |
| Fusion result | Recorded |
| Teacher correction | Not instrumented; corrections are persisted on the session but not separately counted in operational logs |

## Live provider status

The request path is implemented and exercised against the real gateway, and
every automated test uses a faked transport, so the suite needs no key, no
network, and costs nothing. One real call remains opt-in and separate:
`scripts/model_b_live_smoke.py`, which refuses to run without a key and without
`MODEL_B_ENABLED=true`, never prints the key, generates its worksheet in memory
rather than writing it to disk, and prints only aggregate counts and short quoted
statements. It runs the full `image -> prepare -> OpenRouter -> structured JSON
-> validate -> normalise` path and deliberately does not touch Model A.

The typed error mapping, the status-code classification, the retry bounds, and
the redaction behaviour are verified with a mocked transport; the live script
supplies the one thing a mock cannot, which is evidence that a real model
returns a document this contract accepts.
