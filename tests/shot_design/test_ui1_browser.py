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


def test_flag_measurements_use_raw_numbers_and_keep_database_counts():
    flags = [
        {"message": "ip_mean = 8.92e+05 > 1e+05: configured limit", "value": 892400,
         "limit": 100000, "source": "config"},
        {"message": "ip_mean = 8.92e+05 is above the observed 1e+03-1e+05 range of 199607 shots in the database -- outside what has been run, not necessarily outside what is possible",
         "value": 892400, "limit": 100000, "source": "envelope"},
        {"message": "rule 2026 skipped: missing value", "value": None, "limit": None},
    ]
    assert run_js("input.map(formatFlag)", flags) == [
        "ip_mean = 8.924e5 > 1e5: configured limit",
        "ip_mean = 8.924e5; above observed range 1000–1e5 (199607 shots); not an operating limit",
        "rule 2026 skipped: missing value",
    ]


DOM_HARNESS = r"""
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
"""


def run_dom(script):
    result = subprocess.run(
        ["node", "-e", DOM_HARNESS + "\n(async () => {\n" + script +
         "\n})().catch(error => { console.error(error); process.exitCode = 1; });", str(APP)],
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_cells_have_one_toggle_for_all_items_including_async_notes():
    run_dom(r"""
run(`api = async () => ({data: {record: {human: {
  run_title: 'Run title '.repeat(40), mp_title: 'MP title '.repeat(40)}},
  caveats: ['API caveat '.repeat(40)], error: 'Partial title data'}})`);
const table = run(`resultsTable([{shot:199607, score:.03062, run_id:'r2026',
  blurb:'Summary '.repeat(60), caveats:['First caveat '.repeat(40), 'Second caveat '.repeat(40)],
  flags:[{message:'Flag detail '.repeat(40)}]}], 'flat_top')`);
await new Promise(resolve => setImmediate(resolve));
const cells = all(table).filter(n => n.tag === 'td');
assert.equal(cells.length, 5);
let stopped = 0;
for (const cell of cells.slice(2)) {
  const buttons = byClass(cell, 'text-toggle');
  assert.equal(buttons.length, 1, 'one toggle per whole cell');
  assert.equal(byClass(cell, 'section-toggle').length, 0);
  assert.equal((text(cell).match(/…/g) || []).length, 1);
  const button = buttons[0];
  assert(button.attrs['aria-controls']);
  button.events.click({stopPropagation: () => stopped++});
  assert.equal(button.attrs['aria-expanded'], 'true');
  assert.equal(text(button), 'less');
  assert(!text(cell).includes('…'));
}
assert.equal(stopped, 3);
assert(text(cells[2]).includes('r2026\n' + 'Run title '.repeat(40) + '\n' + 'MP title '.repeat(40)));
for (const expected of ['First caveat ', 'Second caveat ', 'Flag detail ', 'API caveat ']) {
  assert(text(cells[4]).includes(expected.repeat(40)));
}
assert(text(cells[4]).includes('Partial title data'));
run(`api = async () => { throw new Error('Title request failed'); }`);
const failed = run(`resultsTable([{shot:199607, caveats:['Existing note '.repeat(40)]}], 'flat_top')`);
await new Promise(resolve => setImmediate(resolve));
const noteCell = all(failed).filter(n => n.tag === 'td')[4];
assert.equal(byClass(noteCell, 'text-toggle').length, 1);
byClass(noteCell, 'text-toggle')[0].events.click({stopPropagation() {}});
assert(text(noteCell).includes('Title request failed'));
assert(text(noteCell).includes('Existing note '.repeat(40)));
""")


def test_phenomena_cell_expands_every_interval_and_all_coverage_notes():
    run_dom(r"""
const table = run(`phenomenaTable([{id:'elm', title:'ELM', n_observed:244, n_forecast:44,
  first_intervals:[{t0_s:0, t1_s:.01}],
  intervals:Array.from({length:244}, (_, i) => ({t0_s:i/100, t1_s:i/100+.01})),
  coverage_note:'observed', coverage_windows:[[0, 1], [2, 4]], coverage_partial:true,
  caveats:['First coverage note '.repeat(30), 'Last coverage note '.repeat(30)]}])`);
const cells = all(table).filter(n => n.tag === 'td');
for (const index of [2, 4]) {
  const buttons = byClass(cells[index], 'text-toggle');
  assert.equal(buttons.length, 1);
  buttons[0].events.click({stopPropagation() {}});
}
assert(!text(table).includes('+241 more'));
assert(text(cells[2]).includes('2.430–2.440 s'));
assert.equal((text(cells[2]).match(/ s/g) || []).length, 244);
assert(text(cells[4]).includes('observed\n0.000–1.000 s\n2.000–4.000 s'));
assert(text(cells[4]).includes('Partial coverage; outside unmeasured'));
assert(text(cells[4]).includes('Last coverage note '.repeat(30)));
assert.equal(byClass(cells[1], 'numeric').length, 1);
assert.equal(byClass(cells[3], 'numeric').length, 1);
assert(byClass(cells[2], 'numeric').some(n => text(n) === '2.430–2.440 s'));
""")


def test_text_clamp_limits_and_short_content_hide_toggle():
    run_dom(r"""
assert.equal(run('CELL_TEXT_LIMIT'), 320);
assert.equal(run('BLOCK_TEXT_LIMIT'), 600);
for (const [expression, limit] of [["cellText([input])", 320], ["longText(input)", 600]]) {
  for (const size of [140, limit, limit+1]) {
    context.input = 'x'.repeat(size);
    const block = run(expression), content = byClass(block, 'text-content')[0];
    const toggle = byClass(block, 'text-toggle')[0];
    assert.equal(toggle.hidden, size <= limit);
    assert.equal(text(content), size <= limit ? context.input : 'x'.repeat(limit) + '…');
    if (size > limit) {
      toggle.events.click({stopPropagation() {}});
      assert.equal(text(content), context.input);
    }
  }
}
run(`renderShot({record:{blurb:'x'.repeat(500), segments:[]}, describe_parts:{
  header:'Shot 199607.', scalars:[], labels:{}, outcome:{fault_strings:['y'.repeat(650)]},
  phenomena:[], operator_quote:{text:'z'.repeat(650)}, caveats:['a'.repeat(350), 'b'.repeat(350)]}})`);
const shot = targets['#shot-record'];
assert(byClass(shot, 'summary-block').some(n => text(n).includes('x'.repeat(500))));
for (const ch of ['y', 'z', 'a']) assert(text(shot).includes(ch.repeat(350)));
assert(text(shot).includes('y'.repeat(600) + '…'));
assert(text(shot).includes('z'.repeat(600) + '…'));
assert.equal(byClass(byClass(shot, 'caveats')[0], 'text-toggle').length, 1);
""")


def test_table_css_preserves_headers_numbers_word_boundaries_and_local_scroll():
    run_dom(r"""
const css = fs.readFileSync(process.argv[1].replace('app.js', 'style.css'), 'utf8');
const rules = [...css.replace(/\/\*[\s\S]*?\*\//g, '').matchAll(/([^{}]+)\{([^{}]*)\}/g)];
const style = (selector) => Object.fromEntries(rules.filter(([, selectors]) =>
  selectors.split(',').map(s => s.trim()).includes(selector)).flatMap(([, , body]) =>
  body.split(';').filter(s => s.includes(':')).map(s => s.split(':').map(s => s.trim()))));
assert.equal(style('th')['white-space'], 'nowrap');
assert.equal(style('th')['overflow-wrap'], 'normal');
assert.equal(style('td')['overflow-wrap'], 'normal');
assert.equal(style('td')['word-break'], 'normal');
assert.equal(style('.numeric')['white-space'], 'nowrap');
assert.equal(style('.numeric')['font-variant-numeric'], 'tabular-nums');
assert.equal(style('table')['table-layout'], 'auto');
assert.equal(style('.table-wrap')['overflow-x'], 'auto');
assert.equal(style('.table-wrap')['max-width'], '100%');
assert.equal(style('.table-wrap')['min-width'], '0');
assert.equal(style('.view')['min-width'], '0');
assert.equal(style('.text-content')['overflow-wrap'], 'break-word');
assert.equal(style('.text-content')['-webkit-line-clamp'], '8');
assert.equal(style('.cell-text > .text-content')['-webkit-line-clamp'], '5');
run(`api = async () => ({data:{}})`);
const table = run(`resultsTable([{shot:199607, score:.03062}], 'flat_top')`);
assert.equal(table.attrs.class, 'table-wrap');
const cells = all(table).filter(n => n.tag === 'td');
assert.equal(text(cells[1]), '0.03062');
assert.equal(byClass(cells[1], 'numeric').length, 1);
const hit = run(`renderHit({shot:199607, score:.03062}, 'flat_top')`);
assert(byClass(hit, 'numeric').some(n => text(n) === 'score 0.03062'));
const scalars = run(`fields({ip_mean:892400, confidence:.9456, t0_s:1.321})`);
assert.equal(byClass(scalars, 'numeric').filter(n => n.tag === 'dd').length, 3);
""")


def test_timeline_uses_shot_domain_clips_true_spans_and_keeps_top_ticks_visible():
    run_dom(r"""
run(`renderEvents({status:'observed', domain:{t0_s:-2,t1_s:8,source:'full segment'},
  events:[{source:'detector',phenomenon:'elm',evidence_kind:'detector',confidence:.9,t0_s:1,t1_s:7},
    {source:'early',t0_s:-5,t1_s:-3}, {source:'late',t0_s:9,t1_s:10},
    {source:'point',t0_s:8,t1_s:8}],
  forecasts:[{source:'model',evidence_kind:'forecast',t0_s:7,t1_s:10}],
  database_intervals:[{source:'database',evidence_kind:'database',t0_s:-3,t1_s:0}],
  coverage:{sources:[{source:'actuator',status:'ok',intervals:[[-10,94.857]]}]}})`);
for (const id of ['event', 'forecast', 'database', 'coverage']) {
  const root = targets['#'+id+'-lanes'], axes = byClass(root, 'axis');
  assert.equal(axes.length, 2);
  const nodes = all(root), lane = byClass(root, 'lane')[0];
  assert(nodes.indexOf(axes[0]) < nodes.indexOf(lane));
  assert(nodes.indexOf(axes[1]) > nodes.indexOf(byClass(root, 'lane').at(-1)));
  assert(!byClass(root, 'collapse-body').some(body => all(body).includes(axes[0])));
  assert.deepEqual(axes[0].children.map(text), ['-2 s','-1 s','0 s','1 s','2 s','3 s','4 s','5 s','6 s','7 s','8 s']);
  const zero = byClass(axes[0], 'zero')[0];
  assert.equal(text(zero), '0 s'); assert(zero.attrs.style.includes('left:20%'));
  for (const mark of byClass(root, 'mark')) {
    const [left, width] = [...mark.attrs.style.matchAll(/(?:left|width):([\d.]+)%/g)].map(m => +m[1]);
    assert(left >= 0 && width >= 0 && left+width <= 100);
  }
}
const eventMarks = byClass(targets['#event-lanes'], 'mark');
assert(eventMarks[0].attrs.style.startsWith('left:30%;width:60%;'));
assert.equal(byClass(targets['#event-lanes'], 'clipped-left').length, 1);
assert.equal(byClass(targets['#event-lanes'], 'clipped-right').length, 1);
assert(eventMarks[1].attrs.title.includes('-5.000–-3.000 s, drawn -2–-2 s'));
assert(eventMarks[2].attrs.title.includes('9.000–10.000 s, drawn 8–8 s'));
const mark = byClass(targets['#coverage-lanes'], 'mark')[0];
assert(mark.attrs.class.includes('clipped-left') && mark.attrs.class.includes('clipped-right'));
assert(mark.attrs.title.includes('coverage -10.000–94.857 s, drawn -2–8 s'));
assert.equal(mark.attrs['aria-label'], mark.attrs.title);
assert(byClass(targets['#forecast-lanes'], 'mark')[0].attrs.class.includes('clipped-right'));
assert(byClass(targets['#database-lanes'], 'mark')[0].attrs.class.includes('clipped-left'));
assert(byClass(targets['#forecast-lanes'], 'mark')[0].attrs.title.includes('evidence_kind: forecast'));
// Missing data uses the documented default, never the available coverage hull.
run(`renderEvents({coverage:{sources:[{source:'actuator',status:'ok',intervals:[[-10,95]]}]}})`);
assert.equal(text(byClass(targets['#coverage-lanes'], 'axis')[0].children.at(-1)), '8 s');
""")


def test_locate_timelines_use_the_supplied_full_segment_domain():
    run_dom(r"""
const hit = run(`renderHit({shot:199607, domain:{t0_s:-4,t1_s:12,source:'full segment'},
  intervals:[{source:'detector',evidence_kind:'detector',t0_s:9,t1_s:10}],
  forecasts:[{source:'model',evidence_kind:'forecast',t0_s:9,t1_s:10}]}, 'flat_top')`);
const marks = byClass(hit, 'mark');
assert.equal(marks.length, 2);
for (const mark of marks) {
  assert(!mark.attrs.class.includes('clipped'));
  assert(mark.attrs.style.startsWith('left:81.25%;width:6.25%;'));
}
for (const axis of byClass(hit, 'axis')) {
  assert.equal(text(axis.children[0]), '-4 s');
  assert.equal(text(axis.children.at(-1)), '12 s');
}
""")


def test_rendered_disclosures_summary_scalars_and_event_tooltips():
    run_dom(r"""
let stopped = 0;
const event = {stopPropagation: () => stopped++};
const long = run(`longText('Full quote for shot 199607. '.repeat(30))`);
const button = all(long).find(n => n.tag === 'button');
assert(button.attrs['aria-controls']);
assert(text(long).includes('…')); button.events.click(event);
assert(text(long).includes('Full quote for shot 199607. '.repeat(30)));
assert.equal(button.attrs['aria-expanded'], 'true');
button.events.click(event); assert.equal(button.attrs['aria-expanded'], 'false');
assert.equal(stopped, 2); assert(byClass(long, 'shot-number').length);
const section = run(`collapsible(el('div', {}, 'Tall content'))`);
const toggle = all(section).find(n => n.tag === 'button');
toggle.events.click(event); assert.equal(text(toggle), 'Show less');
toggle.events.click(event); assert.equal(text(toggle), 'Show all');
run(`renderShot({description: 'NEVER_PARSE_OR_DISPLAY_THIS', units: {ip_mean: 'A'},
 record: {shot: 199607, blurb: 'Goal. Outcome. Finding.', blurb_source: 'template', segments: [
 {name: 'flat_top', t0_ms: 1321, t1_ms: 1792, raw: {ip_mean: 892400}}]},
 describe_parts: {header: 'Shot 199607.', scalars: [{name:'ip_mean', value:892400, units:'A'}],
 segment:{name:'flat_top',t0_s:1.321,t1_s:1.792}, labels:{}, outcome:{}, phenomena:[], caveats:[]}})`);
const shot = targets['#shot-record'];
assert.equal(shot.children[0].attrs.class, 'summary-block');
assert(text(shot.children[0]).includes('Goal. Outcome. Finding.'));
assert.equal(text(byClass(shot.children[0], 'blurb-auto')[0]), 'auto');
assert(text(shot).includes('8.924e5 A')); assert(text(shot).includes('1.321–1.792 s'));
assert(!text(shot).includes('NEVER_PARSE'));
run(`renderShot({record: {blurb:null, blurb_source:'template', segments:[]}})`);
assert.equal(byClass(shot, 'summary-block').length, 0);
run(`api = async () => ({data: {record: {human: {}}}})`);
const results = run(`resultsTable([{shot:199607, score:.945678, blurb:'Offline summary', blurb_source:'llm', caveats:[]}], 'flat_top')`);
assert(text(results).includes('Summary')); assert(!text(results).includes('Quote'));
assert(text(results).includes('199607')); assert(text(results).includes('0.9457'));
assert(text(results).includes('Offline summary')); assert.equal(byClass(results, 'blurb-auto').length, 0);
for (const source of ['llm', 'template', null, undefined, '']) {
  const row = JSON.stringify({shot:199607, blurb:'Stored summary', blurb_source:source, segments:[]});
  run(`renderShot({record: ${row}})`);
  const cell = run(`resultsTable([${row}], 'flat_top')`);
  const located = run(`renderHit(${row}, 'flat_top')`);
  for (const root of [shot.children[0], cell, located]) {
    assert(text(root).includes('Stored summary'));
    const tags = byClass(root, 'blurb-auto');
    assert.equal(tags.length, source === 'template' ? 1 : 0);
    if (tags.length) {
      assert.equal(text(tags[0]), 'auto');
      assert(tags[0].attrs.class.includes('muted'));
      assert(tags[0].attrs.class.includes('small'));
      assert.equal(tags[0].attrs.title, 'Deterministic header + outcome; no model summary yet');
    }
  }
}
for (const blurb of [null, undefined, '', '  \t\n']) {
  const row = JSON.stringify({shot:199607, blurb, blurb_source:'template', segments:[]});
  run(`renderShot({record: ${row}})`);
  assert.equal(byClass(shot, 'summary-block').length, 0);
  const cell = byClass(run(`resultsTable([${row}], 'flat_top')`), 'summary-cell')[0];
  assert.equal(text(cell), '—');
  assert.equal(byClass(cell, 'blurb-auto').length, 0);
  assert.equal(byClass(run(`renderHit(${row}, 'flat_top')`), 'blurb-auto').length, 0);
}
const hit = run(`renderHit({shot:199607, score:.5, phenomenon:'tearing', blurb:'Offline summary', blurb_source:'llm', intervals:[
 {source:'detector', evidence_kind:'detector', t0_s:1.32123, t1_s:1.79221, confidence:.945678}]}, 'flat_top')`);
assert(text(hit).includes('Offline summary'));
const bar = byClass(hit, 'mark')[0];
assert(bar.attrs.title.includes('1.321–1.792 s'));
assert(bar.attrs.title.includes('phenomenon: tearing'));
assert(bar.attrs.title.includes('confidence: 0.946'));
assert.equal(byClass(hit, 'lane-details').length, 0);
""")
