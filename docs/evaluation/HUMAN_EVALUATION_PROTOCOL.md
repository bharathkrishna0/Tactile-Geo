# Human evaluation protocol (not yet run)

No participant data exists. This protocol defines how TactileGeo output should
be evaluated with people before any claim about learning outcomes is made. All
numbers in the accuracy report are machine metrics against synthetic ground
truth; they say nothing about whether a blind or low-vision student can use a
sheet.

## 1. Question

Can students who read tactile graphics answer mathematics questions from a
TactileGeo sheet as correctly and as quickly as from a hand-made reference
tactile graphic of the same diagram, and how does that compare with sighted
peers using the original printed diagram?

## 2. Conditions (within-subject, counterbalanced)

| Code | Material | Who |
|---|---|---|
| T-GEO | TactileGeo SVG, teacher-reviewed, embossed / swell paper | tactile readers |
| T-REF | Reference tactile graphic made by a qualified transcriber | tactile readers |
| V-ORIG | Original printed diagram | sighted comparison group (optional) |

Each participant sees each diagram once, in one condition. Diagram-to-condition
assignment and order follow a Latin square so no diagram is always first or
always in one condition.

## 3. Materials

* 12 diagrams drawn from the math benchmark categories A-L, one per category,
  from the held-out test split or the real-worksheet slice, never from data used
  to tune Model A.
* Every TactileGeo sheet must pass tactile QA, and the teacher-review state used
  (edits, accepted Model B findings) is recorded with the sheet.
* Same paper, embosser / swell settings and Braille code (UEB Grade 2, the
  pipeline's Liblouis table) across conditions.

## 4. Tasks per diagram

Two or three short questions with one correct answer each, written before the
sheets are produced, covering: identification ("which shape is labelled ABC?"),
relationship ("which line is perpendicular to AB?"), quantity ("what is the
length of side AC?") and, for graphs, reading a value.

## 5. Measures

| Measure | Definition | Instrument |
|---|---|---|
| Task correctness | proportion of questions answered correctly | answer sheet, scored by a second person blind to condition |
| Completion time | seconds from first touch / first look to final answer, per question | stopwatch or video timestamps |
| Hints | count of standardised prompts given ("check the left edge"), from a fixed hint list | facilitator log |
| Interpretation errors | wrong answers coded as: misread label, missed element, wrong relationship, Braille misread, confusion from clutter | two coders, agreement reported (Cohen's kappa) |
| Confidence | 1-5 rating after each question | verbal rating |
| Perceived clarity | 1-5 rating per sheet, plus "what was hardest to find?" | short interview |

## 6. Participants and ethics

* Target at least 8 tactile readers; report the actual number, age range,
  Braille experience and onset of vision loss. Small samples mean results are
  descriptive, not significance claims.
* Informed consent (and guardian consent for minors) through the school or a
  disability organisation; participants may stop at any time; no identifying
  data in shared results.
* A qualified teacher of the visually impaired is present.

## 7. Analysis

Per condition: mean and median correctness, completion time, hints, confidence
and clarity, each with the per-participant spread. Paired comparisons
T-GEO vs T-REF per participant. Every interpretation error is listed with the
diagram and the benchmark failure category it maps to (e.g. a misread label ->
FAIL-OCR), so human findings feed back into `docs/evaluation/FAILURE_ANALYSIS.md`.

## 8. What not to claim

* No result from this protocol exists yet; nothing in this repository reports
  participant outcomes.
* Machine metrics are not a proxy for usability and must not be presented as one.
* Passing tactile QA means the sheet meets the configured physical thresholds;
  it is not a certification against any tactile-graphics standard.
