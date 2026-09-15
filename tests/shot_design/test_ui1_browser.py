"""Browser number and wording contracts, executed by Node without dependencies."""

import json
import subprocess
from pathlib import Path

from shot_design.labels.event_sources import LEGACY_HULL_CAVEAT
from shot_design.mcp import tools
from shot_design.retrieval import phenomena as ph

APP = Path(__file__).resolve().parents[2] / "src/shot_design/ui/static/app.js"

# Also the before/after wording ledger for the task report. Shared sources stay intact.
WORDING_CASES = [
    (ph.RUN_SCOPE_TEXT, "Run-level text, not shot-specific"),
    (ph.NO_COVERAGE, "Coverage unrecorded; absence unmeasured"),
    (ph.NO_DETECTOR.format(id="rwm"), "rwm: no detector; text/database evidence only"),
    (ph.COVERAGE_UNPROCESSED.format(title="ELM"),
     "ELM: detectors not run; absence unmeasured"),
    (ph.COVERAGE_OUTSIDE_WINDOW.format(title="ELM", segment="flat_top"),
     "ELM: no coverage of flat_top; absence unmeasured"),
    (ph.COVERAGE_PARTIAL.format(title="Transient activity", segment="flat_top",
                              covered="[0.9873090865135192, 3.797149896621704] s"),
     "Transient activity: covered 0.987–3.797 s of flat_top only; outside unmeasured"),
    (ph.COVERAGE_GAPS.format(title="ELM", segment="flat_top", n=2),
     "ELM: 2 gaps in flat_top coverage; see covered intervals"),
    (ph.TRANSIENT_NOT_CLASSIFIED,
     "Class-agnostic transient; may be ELM-like, a sawtooth or a disruption precursor"),
    (ph.UNCLASSIFIED_KIND.format(n=3, kinds="human"),
     "3 human rows; neither observations nor forecasts"),
    (ph.LABEL_BELOW_FLOOR.format(key="tm/p", p=.123, floor=.5),
     "tm/p: 0.123, below evidence floor 0.500"),
    (ph.QUOTE_UNRELATED.format(title="ELM"), "Shot logbook quote; does not mention ELM"),
    (ph.ALL_FORECASTS.format(n=44), "44 indexed rows, all forecasts; no observations"),
    (ph.TEXT_ONLY, "Text only"),
    (ph.FORECAST_ONLY, "Ranked on forecasts (model estimates)"),
    (ph.LABEL_ONLY.format(p=.9), "Ranked on model labels (0.900); no diagnostic evidence"),
    (ph.DATABASE_ONLY, "Curated list only; no detector, model or logbook evidence"),
    (ph.NO_OBSERVED, "No detector evidence for this phenomenon"),
    (ph.NO_LABEL_MODEL, "No label model for this phenomenon"),
    (ph.LABEL_UNAVAILABLE, "Label unavailable: not run or no valid samples"),
    (ph.NO_TEXT, "No shot-specific operator mention"),
    (ph.NEGATIVE_CLAIM.format(title="ELM"), "Operator reports no ELM"),
    (ph.NO_SEGMENT.format(segment="flat_top"), "No flat_top segment; searched whole record"),
    (tools._FORECAST_CAVEAT.format(n=44), "44 forecasts (model estimates) shown separately"),
    (tools._DATABASE_CAVEAT.format(n=2),
     "2 database intervals (curated lists); coverage unknown; absence unmeasured"),
    (tools._TEXT_CAVEAT.format(n=3), "3 logbook word matches; not observations"),
    (tools._TIMELESS_CAVEAT.format(n=2), "2 rows excluded: time unknown, not outside window"),
    (tools._UNKNOWN_COVERAGE_CAVEAT.format(n=1, shot=199607,
                                         names="actuator/ech_power_total"),
     "actuator/ech_power_total: coverage unknown"),
    (tools._UNPROCESSED_CAVEAT.format(shot=199607),
     "Shot 199607: no completed covering detector; unprocessed, not quiet"),
    (tools._UNCOVERED_UNKNOWN_CAVEAT.format(shot=199607, window=" over [1.0, 2.0] s"),
     "Shot 199607: coverage unknown over 1.000–2.000 s; absence unmeasured"),
    (tools._NO_DETECTION_CAVEAT.format(n=2, shot=199607, window=" over [1.0, 2.0] s"),
     "Shot 199607: 2 sources, 0 detections within coverage over 1.000–2.000 s"),
    ("1 source(s) FAILED on shot 199607: whatever they would have seen is missing from this reply",
     "Shot 199607: 1 sources failed; evidence missing"),
    (LEGACY_HULL_CAVEAT, "Legacy coverage hull; interior gaps unknown"),
    ("elm_clock: " + LEGACY_HULL_CAVEAT,
     "elm_clock: Legacy coverage hull; interior gaps unknown"),
    ("actuator/ech: ran; coverage unknown; absence is not evidence",
     "actuator/ech: coverage unknown; absence unmeasured"),
    (ph.DROPPED_UNSCORED.format(n=2, limit=.5),
     "2 events excluded: confidence unrecorded; minimum 0.500"),
    (ph.AVOID_NO_COVERAGE.format(token="phenomenon:elm", title="ELM"),
     "Kept with avoid phenomenon:elm: ELM unexamined; absence unmeasured"),
    (ph.AVOID_UNPROCESSED.format(token="phenomenon:elm", title="ELM"),
     "Kept with avoid phenomenon:elm: ELM detectors not run; absence unmeasured"),
    (ph.AVOID_UNCOVERED.format(token="phenomenon:elm", title="ELM"),
     "Kept with avoid phenomenon:elm: ELM coverage outside window; absence unmeasured"),
    (ph.AVOID_PARTIAL.format(token="phenomenon:elm", title="ELM"),
     "Kept with avoid phenomenon:elm: ELM coverage partial; outside unmeasured"),
    (ph.AVOID_DROPPED.format(token="phenomenon:elm", n=2, title="ELM"),
     "Avoid phenomenon:elm: excluded 2 shots with observed ELM"),
    (ph.AVOID_DROPPED_CAVEATED.format(n=2, caveat=ph.TRANSIENT_NOT_CLASSIFIED),
     "2 excluded shots: Class-agnostic transient; may be ELM-like, a sawtooth or a disruption precursor"),
    ("--avoid phenomenon:elm: excluded 5 flat_top segment(s) with unprocessed coverage; absence is not evidence",
     "Avoid phenomenon:elm: excluded 5 flat_top segments; unprocessed coverage, absence unmeasured"),
    ("--avoid phenomenon:elm: detectors covered 85.5% of the flat_top window; absence outside that coverage is unmeasured",
     "Avoid phenomenon:elm: covered 85.5% of flat_top; outside unmeasured"),
    ("the window [7.0, 8.0] s is outside every source's coverage of shot 199607, whose display hull is 0.0 to 6.0 s; covered intervals: [0.0, 2.0] s, [3.0, 6.0] s -- nobody looked there, so an empty result says nothing about the window you asked about",
     "Shot 199607: 7.000–8.000 s outside coverage; hull 0.000–6.000 s; covered intervals: 0.000–2.000 s, 3.000–6.000 s; requested window unmeasured"),
    ("the frame-code cache for shot 199607 has no provenance sidecar: the device and thread count it was encoded on are not recorded, and the codes are not bit-reproducible across either",
     "Shot 199607: frame-code device and thread count unrecorded; codes vary with both"),
    ("shot 199607 has no ramp_up segment; the description falls back to `full`",
     "Shot 199607: no ramp_up segment scalars"),
    ("no channel had anything to search on -- give ref_shot, text, constraints or actuators",
     "Enter a reference shot, text, constraints or actuators"),
    ("excluded for having no recorded value: ip_mean (5 shots)",
     "Missing values excluded: ip_mean (5 shots)"),
    ("unknown future caveat: preserve me", "unknown future caveat: preserve me"),
]


def run_js(expression, value):
    script = """
const fs = require('fs'), vm = require('vm');
const context = {input: JSON.parse(fs.readFileSync(0, 'utf8'))};
let code = fs.readFileSync(process.argv[1], 'utf8');
code = code.slice(0, code.lastIndexOf('\\ninit().catch'));
vm.createContext(context);
vm.runInContext(code, context);
process.stdout.write(JSON.stringify(vm.runInContext(process.argv[2], context)));
"""
    result = subprocess.run(["node", "-e", script, str(APP), expression],
                            input=json.dumps(value), capture_output=True, text=True,
                            check=True, timeout=20)
    return json.loads(result.stdout)


def test_browser_wording_preserves_scope_and_missing_coverage():
    assert run_js("input.map(caption)", [a for a, _b in WORDING_CASES]) == [
        b for _a, b in WORDING_CASES
    ]


def test_shared_number_formatter_boundaries_missing_values_and_identifiers():
    cases = [
        [7.13e14, "7.13e14"], [892400, "8.924e5"], [.00012, "1.2e-4"],
        [2.82, "2.82"], [.945678, "0.9457"], [892, "892"], [100000, "1e5"],
        [.001, "0.001"], [0, "0"], [-.00012, "-1.2e-4"],
        [99999.99, "100000"], [None, "—"], ["NaN", "—"],
    ]
    assert run_js("input.map(formatNumber)", [a for a, _b in cases]) == [b for _a, b in cases]
    assert run_js("[display(input, 'shot'), display(input, 'run_id'), "
                  "display(input, 'n_events'), display(input)]", 199607) == [
                      "199607", "199607", "199607", "1.996e5",
                  ]
    assert run_js("[display(input, 't0_s'), display(input, 'confidence')]", 1.32123) == [
        "1.321", "1.321",
    ]


def test_rendered_disclosures_summary_scalars_and_event_tooltips():
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.attrs = {}; this.events = {};
    this.classList = {toggle: (k, on) => {
      const names = new Set((this.attrs.class || '').split(' '));
      if (on) names.add(k); else names.delete(k);
      this.attrs.class = [...names].join(' ');
    }};
  }
  get id() { return this.attrs.id; }
  setAttribute(k, v) { this.attrs[k] = v; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  set textContent(text) { this.children = [text]; }
  addEventListener(k, fn) { this.events[k] = fn; }
}
const targets = {};
const context = {Node: Element, URLSearchParams, document: {
  createElement: (tag) => new Element(tag),
  querySelector: (id) => targets[id] ??= new Element('div'),
}};
let code = fs.readFileSync(process.argv[1], 'utf8');
code = code.slice(0, code.lastIndexOf('init().catch'));
vm.createContext(context); vm.runInContext(code, context);
const run = (code) => vm.runInContext(code, context);
const all = (el) => el instanceof Element ? [el, ...el.children.flatMap(all)] : [];
const text = (el) => el instanceof Element ? el.children.map(text).join('') : String(el);
const byClass = (el, cls) => all(el).filter(n => n.attrs.class?.split(' ').includes(cls));
let stopped = 0;
const event = {stopPropagation: () => stopped++};
const long = run(`longText('Full quote for shot 199607. '.repeat(15))`);
const button = all(long).find(n => n.tag === 'button');
assert(button.attrs['aria-controls']);
assert(text(long).includes('…')); button.events.click(event);
assert(text(long).includes('Full quote for shot 199607. '.repeat(15)));
assert.equal(button.attrs['aria-expanded'], 'true');
button.events.click(event); assert.equal(button.attrs['aria-expanded'], 'false');
assert.equal(stopped, 2); assert(byClass(long, 'shot-number').length);
const section = run(`collapsible(el('div', {}, 'Tall content'))`);
const toggle = all(section).find(n => n.tag === 'button');
toggle.events.click(event); assert.equal(text(toggle), 'Show less');
toggle.events.click(event); assert.equal(text(toggle), 'Show all');
run(`renderShot({description: 'NEVER_PARSE_OR_DISPLAY_THIS', units: {ip_mean: 'A'},
 record: {shot: 199607, summary: 'Goal. Outcome. Finding.', segments: [
 {name: 'flat_top', t0_ms: 1321, t1_ms: 1792, raw: {ip_mean: 892400}}]},
 describe_parts: {header: 'Shot 199607.', scalars: [{name:'ip_mean', value:892400, units:'A'}],
 segment:{name:'flat_top',t0_s:1.321,t1_s:1.792}, labels:{}, outcome:{}, phenomena:[], caveats:[]}})`);
const shot = targets['#shot-record'];
assert.equal(shot.children[0].attrs.class, 'summary-block');
assert(text(shot).includes('8.924e5 A')); assert(text(shot).includes('1.321–1.792 s'));
assert(!text(shot).includes('NEVER_PARSE'));
run(`renderShot({record: {summary:null, segments:[]}})`);
assert.equal(byClass(shot, 'summary-block').length, 0);
run(`api = async () => ({data: {record: {human: {}}}})`);
const results = run(`resultsTable([{shot:199607, score:.945678, summary:'Offline summary', caveats:[]}], 'flat_top')`);
assert(text(results).includes('Summary')); assert(!text(results).includes('Quote'));
assert(text(results).includes('199607')); assert(text(results).includes('0.9457'));
const hit = run(`renderHit({shot:199607, score:.5, phenomenon:'tearing', summary:'Offline summary', intervals:[
 {source:'detector', evidence_kind:'detector', t0_s:1.32123, t1_s:1.79221, confidence:.945678}]}, 'flat_top')`);
assert(text(hit).includes('Offline summary'));
const bar = byClass(hit, 'mark')[0];
assert(bar.attrs.title.includes('1.321–1.792 s'));
assert(bar.attrs.title.includes('phenomenon: tearing'));
assert(bar.attrs.title.includes('confidence: 0.946'));
assert.equal(byClass(hit, 'lane-details').length, 0);
process.stdout.write('PASS');
"""
    result = subprocess.run(["node", "-e", script, str(APP)], capture_output=True,
                            text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "PASS"
