"""Executable browser contracts for the plain-JavaScript waveform editor."""

import subprocess
from pathlib import Path

STATIC = Path(__file__).resolve().parents[2] / "src/shot_design/ui/static"
DESIGN = STATIC / "design.js"
APP = STATIC / "app.js"


HARNESS = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');

class Element {
  constructor(tag, id = '') {
    this.tag = tag;
    this.id = id;
    this.children = [];
    this.attrs = {};
    this.events = {};
    this.value = '';
    this.hidden = false;
    this.disabled = false;
    this.parent = null;
    this.capturedPointers = new Set();
    this.classList = {
      toggle: (name, on) => {
        const names = new Set((this.attrs.class || '').split(' ').filter(Boolean));
        if (on) names.add(name); else names.delete(name);
        this.attrs.class = [...names].join(' ');
      },
      add: name => this.classList.toggle(name, true),
      remove: name => this.classList.toggle(name, false),
      contains: name => (this.attrs.class || '').split(' ').includes(name),
    };
  }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  getAttribute(name) { return this.attrs[name] ?? null; }
  removeAttribute(name) { delete this.attrs[name]; }
  append(...children) {
    for (const child of children) if (child instanceof Element) child.parent = this;
    this.children.push(...children);
  }
  replaceChildren(...children) {
    for (const child of this.children) if (child instanceof Element) child.parent = null;
    this.children = [];
    this.append(...children);
  }
  addEventListener(name, fn) { this.events[name] = fn; }
  set textContent(value) { this.children = [String(value)]; }
  get textContent() { return this.children.map(text).join(''); }
  getBoundingClientRect() { return {left: 0, top: 0, width: 600, height: 260}; }
  get isConnected() { return Boolean(this.id || this.parent?.isConnected); }
  setPointerCapture(pointerId) {
    if (!this.isConnected) throw new Error('InvalidStateError: detached pointer-capture target');
    this.capturedPointers.add(pointerId);
  }
  releasePointerCapture(pointerId) { this.capturedPointers.delete(pointerId); }
  hasPointerCapture(pointerId) { return this.capturedPointers.has(pointerId); }
}

const ids = [
  'design-reference', 'design-start', 'design-end',
  'design-notes', 'design-preview', 'design-save', 'design-revisions',
  'design-reload', 'design-channel', 'design-reset', 'design-point-time',
  'design-point-value', 'design-apply-point', 'design-svg', 'design-status',
  'design-errors', 'design-warnings', 'design-selection', 'design-json',
  'design-ignite', 'design-dirty', 'design-channel-details',
  'design-legend', 'design-prepare', 'design-feedback', 'design-prepare-status',
  'design-delete-point', 'design-simplify', 'design-merge', 'design-merge-status',
];
const targets = Object.fromEntries(ids.map(id => [`#${id}`, new Element('div', id)]));
for (const id of ['design-reference', 'design-start',
  'design-end', 'design-notes', 'design-revisions', 'design-channel',
  'design-point-time', 'design-point-value']) targets[`#${id}`].tag = 'input';

const text = node => node instanceof Element ? node.children.map(text).join('') : String(node);
const all = node => node instanceof Element ? [node, ...node.children.flatMap(all)] : [];
const byClass = (node, name) => all(node).filter(child =>
  (child.attrs.class || '').split(' ').includes(name));
const document = {
  querySelector: selector => targets[selector] || null,
  createElement: tag => new Element(tag),
  createElementNS: (_namespace, tag) => new Element(tag),
};
const timers = new Map();
let nextTimer = 0;
const context = {
  assert, console, document, Node: Element, location: {hash: ''},
  controlTimers: false,
  setTimeout: fn => {
    const id = ++nextTimer;
    if (context.controlTimers) timers.set(id, fn); else fn();
    return id;
  },
  clearTimeout: id => timers.delete(id),
  flushTimers: () => {
    const pending = [...timers.values()];
    timers.clear();
    for (const fn of pending) fn();
  },
  pendingTimers: () => timers.size,
};
context.globalThis = context;
context.window = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const run = source => vm.runInContext(source, context);
"""


FIXTURE = r"""
const preview = (overrides = {}) => ({
  program: {
    schema_version: 'shot-design/1', id: null, created: null,
    reference_shot: 199607, comparison_shots: [199608], start_s: 1,
    end_s: 1.15, notes: 'Initial plan', reference_digest: 'digest-a', edits: {},
    units: {'pinj[0]': 'kW', 'nbi.total': 'kW'},
    ...(overrides.program || {}),
  },
  channels: [{
    key: 'pinj[0]', label: 'Neutral beam 1', units: 'kW', editable: true,
    reason: null,
    reference: [
      {t_s: 0, y: 80}, {t_s: .95, y: 90}, {t_s: 1, y: 100},
      {t_s: 1.05, y: 110}, {t_s: 1.1, y: 100},
    ],
    vertices: [{t_s: 1, y: 100}, {t_s: 1.05, y: 110}, {t_s: 1.1, y: 100}],
    comparisons: [{shot: 199608, vertices: [
      {t_s: 1, y: 95}, {t_s: 1.05, y: 105}, {t_s: 1.1, y: 115},
    ]}],
  }, {
    key: 'nbi.total', label: 'NBI total', units: 'kW', editable: false,
    reason: 'No active member channels', reference: [], vertices: [], comparisons: [],
  }],
  validation: {errors: [], warnings: ['Provisional check'], can_export: true},
  context_start_s: 0, seed_frames: 20, frame_s: .05,
  ...overrides,
});
"""


def run_design(script):
    result = subprocess.run(
        ["node", "-e", HARNESS + "\n(async () => {\n" + FIXTURE + script +
         "\n})().catch(error => { console.error(error); process.exitCode = 1; });",
         str(DESIGN)],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_numeric_edit_sends_only_changed_channel_and_invalidates_downloads():
    run_design(r"""
const calls = [];
const api = async (path, options = {}) => {
  calls.push([path, options]);
  if (path === '/api/design') return {data: []};
  const sent = JSON.parse(options.body || '{}');
  return {data: preview({program: {...preview().program, ...sent,
    units: preview().program.units}})};
};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openDesign(199607);
const circles = byClass(targets['#design-svg'], 'design-point');
assert.equal(circles.length, 3);
let prevented = 0;
circles[0].events.keydown({key: 'Enter', preventDefault() { prevented++; }});
assert.equal(prevented, 1);
assert.equal(targets['#design-point-value'].value, '100');
assert.equal(targets['#design-point-value'].disabled, false);
byClass(targets['#design-svg'], 'design-point')[1].events.click({stopPropagation() {}});
assert.equal(targets['#design-point-time'].value, '1.05');
assert.equal(targets['#design-point-value'].value, '110');
assert.equal(targets['#design-point-time'].disabled, false);
assert.equal(targets['#design-point-value'].disabled, false);
assert.equal(targets['#design-apply-point'].disabled, false);

// A saved revision may export until a real local edit is made.
targets['#design-json'].hidden = false;
targets['#design-json'].setAttribute('href', '/api/design/saved/json');
targets['#design-ignite'].hidden = false;
targets['#design-ignite'].setAttribute('href', '/api/design/saved/ignite');
targets['#design-point-value'].value = '175.5';
await targets['#design-apply-point'].events.click({preventDefault() {}});
const body = JSON.parse(calls.filter(([path]) => path === '/api/design/preview').at(-1)[1].body);
assert.deepEqual(Object.keys(body.edits), ['pinj[0]']);
assert.deepEqual(body.units, {'pinj[0]': 'kW', 'nbi.total': 'kW'});
assert.equal(body.edits['pinj[0]'][1].y, 175.5);
assert.equal(body.edits['pinj[0]'][1].t_s, 1.05);
assert.equal(targets['#design-json'].hidden, true);
assert.equal(targets['#design-json'].getAttribute('href'), null);
assert.equal(targets['#design-ignite'].hidden, true);
assert.equal(targets['#design-dirty'].textContent, 'Unsaved changes');
assert(text(targets['#design-channel-details']).includes('Neutral beam 1'));
assert(text(targets['#design-channel-details']).includes('kW'));
assert.equal(byClass(targets['#design-svg'], 'design-comparison-trace').length, 1);
assert(byClass(targets['#design-svg'], 'design-x-tick-label').length >= 3);
assert(byClass(targets['#design-svg'], 'design-y-tick-label').length >= 3);
assert(text(targets['#design-svg']).includes('5 s'));
assert(text(targets['#design-legend']).includes('Reference 199608'));
""")


def test_primary_reference_can_be_entered_from_empty_design_view():
    run_design(r"""
const requests = [];
const api = async (path, options = {}) => {
  if (path === '/api/design') return {data: []};
  requests.push(JSON.parse(options.body));
  return {data: preview()};
};
await context.ShotDesign.initDesign({api});
targets['#design-reference'].value = '199607, 199608';
targets['#design-start'].value = '1';
targets['#design-end'].value = '5';
await targets['#design-preview'].events.click({preventDefault() {}});
assert.equal(requests[0].reference_shot, 199607);
assert.deepEqual(requests[0].comparison_shots, [199608]);
assert.equal(byClass(targets['#design-svg'], 'design-point').length, 3);
""")


def test_reset_removes_edit_and_restores_reference_vertices():
    run_design(r"""
const requests = [];
const api = async (path, options = {}) => {
  if (path === '/api/design') return {data: []};
  const body = JSON.parse(options.body || '{}');
  requests.push(body);
  return {data: preview({program: {...preview().program, ...body}})};
};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openDesign(199607);
byClass(targets['#design-svg'], 'design-point')[0].events.click({stopPropagation() {}});
targets['#design-point-value'].value = '140';
await targets['#design-apply-point'].events.click({preventDefault() {}});
assert(requests.at(-1).edits['pinj[0]']);
await targets['#design-reset'].events.click({preventDefault() {}});
assert.deepEqual(requests.at(-1).edits, {});
assert.equal(byClass(targets['#design-svg'], 'design-point').length, 3);
assert(text(targets['#design-selection']).includes('Reference waveform restored'));
""")


def test_save_reopen_restores_program_and_only_current_revision_downloads():
    run_design(r"""
const saved = preview({program: {...preview().program, id: 'design-abc',
  created: '2026-09-18T12:00:00Z', notes: 'Saved plan'}});
const calls = [];
const api = async (path, options = {}) => {
  calls.push([path, options]);
  if (path === '/api/design' && options.method === 'POST') return {data: saved};
  if (path === '/api/design') return {data: [{id: 'design-abc', reference_shot: 199607,
    created: '2026-09-18T12:00:00Z', start_s: 1, end_s: 1.15}]};
  if (path === '/api/design/design-abc') return {data: saved};
  return {data: preview()};
};
await context.ShotDesign.initDesign({api});
assert(text(targets['#design-revisions']).includes('Shot 199607 · 1–1.15 s · design-a'));
await context.ShotDesign.openDesign(199607);
targets['#design-notes'].value = 'Saved plan';
targets['#design-notes'].events.input({});
await targets['#design-save'].events.click({preventDefault() {}});
assert.equal(JSON.parse(calls.find(([path, options]) =>
  path === '/api/design' && options.method === 'POST')[1].body).notes, 'Saved plan');
assert.equal(targets['#design-json'].attrs.href, '/api/design/design-abc/json');
assert.equal(targets['#design-ignite'].attrs.href, '/api/design/design-abc/ignite');
assert.equal(targets['#design-ignite'].hidden, false);
assert.equal(targets['#design-dirty'].textContent, 'Saved revision design-abc');

targets['#design-notes'].value = 'Changed after save';
targets['#design-notes'].events.input({});
assert.equal(targets['#design-json'].hidden, true);
byClass(targets['#design-svg'], 'design-point')[0].events.click({stopPropagation() {}});
assert.equal(targets['#design-point-value'].value, '100');
targets['#design-revisions'].value = 'design-abc';
await targets['#design-reload'].events.click({preventDefault() {}});
assert.equal(targets['#design-notes'].value, 'Saved plan');
assert.equal(targets['#design-point-time'].value, '');
assert.equal(targets['#design-point-value'].value, '');
assert.equal(targets['#design-json'].hidden, false);
assert(calls.some(([path]) => path === '/api/design/design-abc'));
""")


def test_stale_preview_cannot_replace_a_newly_saved_revision():
    run_design(r"""
const saved = preview({program: {...preview().program, id: 'design-new',
  created: '2026-09-18T13:00:00Z', notes: 'Saved while check ran'}});
let previewCount = 0, resolveCheck;
const api = async (path, options = {}) => {
  if (path === '/api/design' && options.method === 'POST') return {data: saved};
  if (path === '/api/design') return {data: []};
  if (path === '/api/design/preview' && ++previewCount === 2) {
    return new Promise(resolve => { resolveCheck = resolve; });
  }
  return {data: preview()};
};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openDesign(199607);
byClass(targets['#design-svg'], 'design-point')[1].events.click({stopPropagation() {}});
targets['#design-point-value'].value = '160';
targets['#design-apply-point'].events.click({preventDefault() {}});
targets['#design-notes'].value = 'Saved while check ran';
await targets['#design-save'].events.click({preventDefault() {}});
resolveCheck({data: preview({program: {...preview().program, id: null,
  notes: 'stale check'}})});
await new Promise(resolve => setImmediate(resolve));
assert.equal(targets['#design-dirty'].textContent, 'Saved revision design-new');
assert.equal(targets['#design-json'].attrs.href, '/api/design/design-new/json');
assert.equal(targets['#design-json'].hidden, false);
""")


def test_save_response_cannot_replace_edits_made_while_save_is_in_flight():
    run_design(r"""
const saved = preview({program: {...preview().program, id: 'design-old',
  created: '2026-09-18T13:00:00Z', notes: 'Snapshot being saved'}});
let resolveSave, resolvePointCheck;
const previewBodies = [];
const api = async (path, options = {}) => {
  if (path === '/api/design' && options.method === 'POST') {
    return new Promise(resolve => { resolveSave = resolve; });
  }
  if (path === '/api/design') return {data: []};
  if (path === '/api/design/preview') {
    const body = JSON.parse(options.body);
    previewBodies.push(body);
    if (previewBodies.length > 1) {
      return new Promise(resolve => { resolvePointCheck = resolve; });
    }
    return {data: preview()};
  }
  throw new Error(path);
};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openDesign(199607);
targets['#design-notes'].value = 'Snapshot being saved';
targets['#design-notes'].events.input({});
const saving = targets['#design-save'].events.click({preventDefault() {}});

targets['#design-notes'].value = 'Newer local note';
targets['#design-notes'].events.input({});
byClass(targets['#design-svg'], 'design-point')[1].events.click({stopPropagation() {}});
targets['#design-point-value'].value = '177';
targets['#design-apply-point'].events.click({preventDefault() {}});
assert.equal(previewBodies.at(-1).notes, 'Newer local note');
assert.equal(previewBodies.at(-1).edits['pinj[0]'][1].y, 177);

resolveSave({data: saved});
await saving;
assert.equal(targets['#design-notes'].value, 'Newer local note');
assert.equal(targets['#design-dirty'].textContent, 'Unsaved changes');
assert.equal(targets['#design-json'].hidden, true);
assert.equal(targets['#design-json'].getAttribute('href'), null);
assert.equal(targets['#design-ignite'].hidden, true);
assert(resolvePointCheck, 'the newer local point edit starts its own check');
""")


def test_preview_response_cannot_replace_newer_form_input():
    run_design(r"""
let previewCount = 0, resolveCheck;
const api = async (path, options = {}) => {
  if (path === '/api/design') return {data: []};
  if (path === '/api/design/preview' && ++previewCount === 2) {
    return new Promise(resolve => { resolveCheck = resolve; });
  }
  return {data: preview()};
};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openDesign(199607);
const checking = targets['#design-preview'].events.click({preventDefault() {}});
targets['#design-notes'].value = 'Typed after check started';
targets['#design-notes'].events.input({});
resolveCheck({data: preview({program: {...preview().program, notes: 'Older note'}})});
await checking;
assert.equal(targets['#design-notes'].value, 'Typed after check started');
assert.equal(targets['#design-dirty'].textContent, 'Unsaved changes');
assert.equal(targets['#design-json'].hidden, true);
assert.equal(targets['#design-ignite'].hidden, true);
""")


def test_structural_form_change_requires_preview_before_waveform_edit():
    run_design(r"""
const previewBodies = [];
const api = async (path, options = {}) => {
  if (path === '/api/design') return {data: []};
  const body = JSON.parse(options.body);
  previewBodies.push(body);
  return {data: preview({program: {...preview().program, ...body}})};
};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openDesign(199607);
byClass(targets['#design-svg'], 'design-point')[1].events.click({stopPropagation() {}});
targets['#design-reference'].value = '199609';
targets['#design-reference'].events.input({});
assert.equal(targets['#design-point-time'].disabled, true);
assert.equal(targets['#design-point-value'].disabled, true);
assert.equal(targets['#design-apply-point'].disabled, true);
const count = previewBodies.length;
targets['#design-point-value'].value = '180';
await targets['#design-apply-point'].events.click({preventDefault() {}});
assert.equal(previewBodies.length, count);
assert(text(targets['#design-errors']).includes('Check preview after changing'));

await targets['#design-preview'].events.click({preventDefault() {}});
assert.equal(previewBodies.at(-1).reference_shot, 199609);
assert.equal(targets['#design-apply-point'].disabled, true,
  'accepted structural preview clears the old point selection');
""")


def test_reset_reopened_edited_revision_restores_reference_vertices():
    run_design(r"""
const editedVertices = [{t_s: 1, y: 135}, {t_s: 1.05, y: 145}, {t_s: 1.1, y: 155}];
const base = preview();
const edited = preview({
  program: {...base.program, id: 'design-edited', created: '2026-09-18T14:00:00Z',
    edits: {'pinj[0]': editedVertices}},
  channels: [{...base.channels[0], vertices: editedVertices}, base.channels[1]],
});
const previewBodies = [];
const api = async (path, options = {}) => {
  if (path === '/api/design') return {data: [{id: 'design-edited', reference_shot: 199607,
    created: '2026-09-18T14:00:00Z', start_s: 1, end_s: 1.15}]};
  if (path === '/api/design/design-edited') return {data: edited};
  if (path === '/api/design/preview') {
    const body = JSON.parse(options.body);
    previewBodies.push(body);
    return {data: preview({program: {...base.program, ...body}, channels: base.channels})};
  }
  throw new Error(path);
};
await context.ShotDesign.initDesign({api});
targets['#design-revisions'].value = 'design-edited';
await targets['#design-reload'].events.click({preventDefault() {}});
assert.equal(targets['#design-reset'].disabled, false);
await targets['#design-reset'].events.click({preventDefault() {}});
assert.deepEqual(previewBodies.at(-1).edits, {});
byClass(targets['#design-svg'], 'design-point')[0].events.click({stopPropagation() {}});
assert.equal(targets['#design-point-value'].value, '100');
assert(text(targets['#design-selection']).includes('Joint 1'));
""")


def test_invalid_form_value_reports_error_without_losing_draft():
    run_design(r"""
const api = async (path, options = {}) => path === '/api/design' ? {data: []} :
  {data: preview()};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openDesign(199607);
targets['#design-notes'].value = 'Keep this draft';
targets['#design-notes'].events.input({});
targets['#design-reference'].value = 'not-a-shot';
await targets['#design-save'].events.click({preventDefault() {}});
assert(text(targets['#design-errors']).includes('Reference must be a whole shot number'));
assert.equal(targets['#design-notes'].value, 'Keep this draft');
assert.equal(targets['#design-dirty'].textContent, 'Unsaved changes');
""")


def test_drag_updates_vertex_without_clamping_physical_value():
    run_design(r"""
const requests = [];
const api = async (path, options = {}) => {
  if (path === '/api/design') return {data: []};
  const body = JSON.parse(options.body || '{}');
  requests.push(body);
  return {data: preview({program: {...preview().program, ...body}})};
};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openDesign(199607);
const point = byClass(targets['#design-svg'], 'design-point')[1];
const pointX = Number(point.attrs.cx) * 600 / 760;
point.events.pointerdown({pointerId: 1, clientX: pointX, clientY: 130,
  preventDefault() {}, stopPropagation() {}});
assert.equal(targets['#design-svg'].hasPointerCapture(1), true);
targets['#design-svg'].events.pointermove({pointerId: 1, clientX: pointX, clientY: -130,
  preventDefault() {}});
await targets['#design-svg'].events.pointerup({pointerId: 1});
assert.equal(targets['#design-svg'].hasPointerCapture(1), false);
const edit = requests.at(-1).edits['pinj[0]'][1];
assert(edit.y > 120, 'value follows pointer beyond the plotted reference range');
assert.equal(edit.t_s, 1.05);
const count = requests.length;
const nextPoint = byClass(targets['#design-svg'], 'design-point')[0];
nextPoint.events.pointerdown({pointerId: 2, clientX: 100, clientY: 100,
  preventDefault() {}, stopPropagation() {}});
assert.equal(targets['#design-svg'].hasPointerCapture(2), true);
await targets['#design-svg'].events.pointercancel({pointerId: 2});
assert.equal(targets['#design-svg'].hasPointerCapture(2), false);
assert.equal(requests.length, count, 'a cancelled drag with no movement creates no edit');
""")


def test_queued_point_preview_is_cancelled_by_newer_note_input():
    run_design(r"""
context.controlTimers = true;
const previewBodies = [];
const api = async (path, options = {}) => {
  if (path === '/api/design') return {data: []};
  const body = JSON.parse(options.body);
  previewBodies.push(body);
  return {data: preview({program: {...preview().program, ...body}})};
};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openDesign(199607);
byClass(targets['#design-svg'], 'design-point')[1].events.click({stopPropagation() {}});
targets['#design-point-value'].value = '166';
targets['#design-apply-point'].events.click({preventDefault() {}});
assert.equal(context.pendingTimers(), 1);
const count = previewBodies.length;
targets['#design-notes'].value = 'Note typed during debounce';
targets['#design-notes'].events.input({});
context.flushTimers();
await new Promise(resolve => setImmediate(resolve));
assert.equal(previewBodies.length, count);
assert.equal(targets['#design-notes'].value, 'Note typed during debounce');
assert.equal(targets['#design-dirty'].textContent, 'Unsaved changes');
""")


def test_queued_point_preview_is_cancelled_by_structural_input():
    run_design(r"""
context.controlTimers = true;
const previewBodies = [];
const api = async (path, options = {}) => {
  if (path === '/api/design') return {data: []};
  const body = JSON.parse(options.body);
  previewBodies.push(body);
  return {data: preview({program: {...preview().program, ...body}})};
};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openDesign(199607);
byClass(targets['#design-svg'], 'design-point')[1].events.click({stopPropagation() {}});
targets['#design-point-value'].value = '166';
targets['#design-apply-point'].events.click({preventDefault() {}});
assert.equal(context.pendingTimers(), 1);
const count = previewBodies.length;
targets['#design-reference'].value = '199609';
targets['#design-reference'].events.input({});
context.flushTimers();
await new Promise(resolve => setImmediate(resolve));
assert.equal(previewBodies.length, count);
assert.equal(targets['#design-reference'].value, '199609');
assert.equal(byClass(targets['#design-svg'], 'design-point').length, 0);
assert.equal(targets['#design-apply-point'].disabled, true);
""")


def test_missing_channel_is_visible_but_cannot_be_edited():
    run_design(r"""
const api = async (path, options = {}) => path === '/api/design' ? {data: []} :
  {data: preview()};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openDesign(199607);
targets['#design-channel'].value = 'nbi.total';
targets['#design-channel'].events.change({});
assert.equal(targets['#design-point-time'].disabled, true);
assert.equal(targets['#design-point-value'].disabled, true);
assert.equal(targets['#design-reset'].disabled, true);
assert(text(targets['#design-channel-details']).includes('No active member channels'));
assert(text(targets['#design-status']).includes('Ready to export'));
assert(text(targets['#design-warnings']).includes('Provisional check'));
""")


def test_editor_prefers_an_editable_total_channel_returned_by_server():
    run_design(r"""
const total = {...preview().channels[0], key: 'nbi.total', label: 'NBI total'};
const api = async (path, options = {}) => path === '/api/design' ? {data: []} :
  {data: preview({channels: [preview().channels[0], total]})};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openDesign(199607);
assert.equal(targets['#design-channel'].value, 'nbi.total');
assert(text(targets['#design-channel-details']).includes('NBI total'));
""")


def test_existing_views_offer_use_as_reference_without_static_imports():
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
class Element {
  constructor(tag) { this.tag=tag; this.children=[]; this.attrs={}; this.events={};
    this.classList={toggle(){}}; }
  setAttribute(k,v){this.attrs[k]=String(v)} removeAttribute(k){delete this.attrs[k]}
  append(...x){this.children.push(...x)} replaceChildren(...x){this.children=x}
  addEventListener(k,v){this.events[k]=v} set textContent(v){this.children=[String(v)]}
}
const targets={};
const context={Node:Element, URLSearchParams, location:{hash:''}, document:{
  createElement:t=>new Element(t), querySelector:id=>targets[id]??=new Element('div')}};
let code=fs.readFileSync(process.argv[1],'utf8');
code=code.slice(0,code.lastIndexOf('init().catch'));
vm.createContext(context); vm.runInContext(code,context);
const run=s=>vm.runInContext(s,context);
const all=n=>n instanceof Element?[n,...n.children.flatMap(all)]:[];
const text=n=>n instanceof Element?n.children.map(text).join(''):String(n);
const controls=n=>all(n).filter(x=>(x.attrs.class||'').split(' ').includes('use-reference'));
run(`api = async () => ({data:{record:{human:{}},caveats:[]}})`);
const table=run(`resultsTable([{shot:199607,score:.5}], 'flat_top')`);
const button=controls(table)[0]; assert(button); assert(text(button).includes('Use as reference'));
button.events.click({preventDefault(){},stopPropagation(){}});
assert.equal(context.location.hash,'#design/199607');
run(`S.shot=199608; renderShot({record:{shot:199608,human:{},segments:[]}})`);
const detail=controls(targets['#shot-record'])[0]; assert(detail);
detail.events.click({preventDefault(){},stopPropagation(){}});
assert.equal(context.location.hash,'#design/199608');
"""
    result = subprocess.run(
        ["node", "-e", script, str(APP)], capture_output=True, text=True,
        timeout=20, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_uncached_preview_offers_preparation_without_blocking_edit_or_save():
    run_design(r"""
const requests = [];
const api = async (path, options = {}) => {
  requests.push(path);
  if (path === '/api/design' && !options.method) return {data: []};
  const program = {...preview().program, ...JSON.parse(options.body || '{}')};
  if (path === '/api/design/prepare') return {data: preview({program,
    validation: {errors: [], warnings: [], can_save: true, can_export: true, needs_seed: false}})};
  return {data: preview({program, validation: {errors: ['No seed cache yet'],
    warnings: [], can_save: true, can_export: false, needs_seed: true}})};
};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openDesign(199597);
assert.equal(targets['#design-prepare'].hidden, false);
assert.equal(targets['#design-save'].disabled, false);
assert(text(targets['#design-feedback']).includes('edit and save'));
assert.equal(byClass(targets['#design-svg'], 'design-point').length, 3);
await targets['#design-prepare'].events.click({preventDefault() {}});
assert(requests.includes('/api/design/prepare'));
assert.equal(targets['#design-prepare'].hidden, true);
assert(text(targets['#design-feedback']).includes('ready'));
""")


def test_unavailable_source_explained_next_to_preview_button():
    run_design(r"""
const api = async path => path === '/api/design' ? {data: []} : {data: preview({
  channels: [], validation: {errors: ['No actuator data for shot 199597'],
    warnings: [], can_save: false, can_export: false, needs_seed: false}})};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openDesign(199597);
assert(text(targets['#design-feedback']).includes('No actuator data'));
assert.equal(targets['#design-save'].disabled, true);
assert.equal(targets['#design-prepare'].hidden, true);
""")


def test_preparation_failure_remains_visible_after_another_preview():
    run_design(r"""
let failPreparation;
const api = async (path, options = {}) => {
  if (path === '/api/design') return {data: []};
  if (path === '/api/design/prepare') return await new Promise((_resolve, reject) => { failPreparation = reject; });
  return {data: preview({validation: {errors: ['No seed cache yet'], warnings: [],
    can_save: true, can_export: false, needs_seed: true}})};
};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openDesign(199597);
const preparing = targets['#design-prepare'].events.click({preventDefault() {}});
await targets['#design-preview'].events.click({preventDefault() {}});
failPreparation(new Error('Codec unavailable'));
await preparing;
assert(text(targets['#design-prepare-status']).includes('Codec unavailable'));
assert(!text(targets['#design-feedback']).includes('Loading'));
assert.equal(targets['#design-prepare'].disabled, false);
""")


def test_flat_and_linear_reference_runs_have_only_corner_handles():
    run_design(r"""
const points = Array.from({length: 80}, (_, i) => ({t_s: 1+i*.05,
  y: i < 30 ? 100 + (i % 2)*.1 : i < 50 ? 100+(i-30)*5 : 200}));
let savedBody;
const api = async (path, options = {}) => {
  if (path === '/api/design' && !options.method) return {data: []};
  if (path === '/api/design' && options.method) savedBody = JSON.parse(options.body);
  const p = preview();
  return {data: preview({program: {...p.program, end_s: 5}, channels: [
    {...p.channels[0], reference: points, vertices: points},
  ]})};
};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openDesign(199607);
assert.equal(byClass(targets['#design-svg'], 'design-point').length, 4);
await targets['#design-save'].events.click({preventDefault() {}});
assert.deepEqual(savedBody.edits, {}, 'display simplification must not change saved controls');
""")


def test_click_adds_joint_delete_removes_it_and_history_stays_locked():
    run_design(r"""
const sent = [];
const p = preview();
const flat = Array.from({length: 80}, (_, i) => ({t_s: 1+i*.05,y:100}));
const api = async (path, options = {}) => {
  if (path === '/api/design') return {data: []};
  const body = JSON.parse(options.body);
  sent.push(body);
  return {data: preview({program: {...p.program,...body}, context_start_s:0,
    channels:[{...p.channels[0],reference:flat,vertices:body.edits?.['pinj[0]'] || flat}]})};
};
await context.ShotDesign.initDesign({api});
// Normal controls loaded for a 1–5 second prediction.
targets['#design-reference'].value='199607';
targets['#design-start'].value='1'; targets['#design-end'].value='5';
await targets['#design-preview'].events.click({preventDefault() {}});
assert.equal(byClass(targets['#design-svg'], 'design-point').length,2);
const svg=targets['#design-svg'];
// Synthetic DOM SVG is 600x260; x=370 maps to interior shot time ~3s.
await svg.events.click({clientX:370,clientY:110,preventDefault(){}});
assert.equal(byClass(svg,'design-point').length,3);
assert.equal(Object.keys(sent.at(-1).edits).length,1);
const inserted=sent.at(-1).edits['pinj[0]'][1];
assert(inserted.t_s>1 && inserted.t_s<4.95);
assert.equal(targets['#design-delete-point'].disabled,false);
await targets['#design-delete-point'].events.click({preventDefault(){}});
assert.equal(sent.at(-1).edits['pinj[0]'].length,2);
// Clicking the locked history cannot add a point or issue a new preview.
const count=sent.length;
await svg.events.click({clientX:70,clientY:110,preventDefault(){}});
assert.equal(sent.length,count);
byClass(svg,'design-point')[0].events.click({stopPropagation(){}});
assert.equal(targets['#design-delete-point'].disabled,true);
assert.equal(targets['#design-point-time'].disabled,true);
""")


def test_single_reference_field_restores_order_and_switches_seed_shot():
    run_design(r"""
const calls=[];
const api=async(path,options={})=>{
  if(path==='/api/design')return {data:[]};
  const body=JSON.parse(options.body);calls.push(body);
  return {data:preview({program:{...preview().program,...body}})};
};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openDesign(199607);
assert.equal(targets['#design-reference'].value,'199607');
targets['#design-reference'].value='199608, 199607, 199609, 199608';
targets['#design-reference'].events.input({});
await targets['#design-preview'].events.click({preventDefault(){}});
assert.equal(calls.at(-1).reference_shot,199608);
assert.deepEqual(calls.at(-1).comparison_shots,[199607,199609]);
assert.equal(calls.at(-1).reference_digest,null);
assert.equal(targets['#design-reference'].value,'199608, 199607, 199609');
""")


def test_saved_collinear_joints_survive_and_keyboard_delete_keeps_anchors():
    run_design(r"""
const points=[{t_s:1,y:100},{t_s:1.05,y:100},{t_s:1.1,y:100}];
const revision=preview({program:{...preview().program,id:'saved',edits:{'pinj[0]':points}},
  channels:[{...preview().channels[0],vertices:points}]});
let body;
const api=async(path,options={})=>{
  if(path==='/api/design')return {data:[]};
  if(path==='/api/design/saved')return {data:revision};
  body=JSON.parse(options.body);
  return {data:preview({program:body,channels:[{...preview().channels[0],vertices:body.edits['pinj[0]']}]})};
};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openSavedDesign('saved');
let handles=byClass(targets['#design-svg'],'design-point');
assert.equal(handles.length,3,'explicit collinear joint must survive reopening');
handles[0].events.keydown({key:'Delete',preventDefault(){}});
assert.equal(body,undefined,'anchors cannot be deleted');
handles[1].events.keydown({key:'Backspace',preventDefault(){}});
assert.equal(body.edits['pinj[0]'].length,2);
""")


def test_merging_references_replaces_edits_and_reset_keeps_average_baseline():
    run_design(r"""
const calls=[];
const mean=[{t_s:1,y:150},{t_s:1.1,y:150}];
let program=preview().program;
const api=async(path,options={})=>{
  if(path==='/api/design')return {data:[]};
  const body=JSON.parse(options.body);calls.push({path,body});
  program={...program,...body};
  if(path==='/api/design/merge')program.proposal={method:'average',skipped_channels:{'gas_raw[0]':'missing'}};
  return {data:preview({program,channels:[{...preview().channels[0],
    baseline_vertices:program.proposal?mean:preview().channels[0].vertices,
    vertices:program.edits['pinj[0]'] || (program.proposal?mean:preview().channels[0].vertices)}]})};
};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openDesign(199607);
targets['#design-reference'].value='199607, 199608';
await targets['#design-merge'].events.click({preventDefault(){}});
assert.equal(calls.at(-1).path,'/api/design/merge');
assert.deepEqual(calls.at(-1).body.edits,{});
assert(text(targets['#design-merge-status']).includes('1 unavailable'));
byClass(targets['#design-svg'],'design-point')[0].events.click({stopPropagation(){}});
targets['#design-point-value'].value='175';
await targets['#design-apply-point'].events.click({preventDefault(){}});
await targets['#design-reset'].events.click({preventDefault(){}});
assert.equal(calls.at(-1).body.proposal.method,'average');
assert.deepEqual(calls.at(-1).body.edits,{});
assert.equal(byClass(targets['#design-svg'],'design-point').length,2);
assert(text(targets['#design-selection']).includes('Averaged waveform restored'));
targets['#design-reference'].value='199607';
targets['#design-reference'].events.input({});
await targets['#design-preview'].events.click({preventDefault(){}});
assert.equal(calls.at(-1).body.proposal,null,'removing donors returns to first reference');
assert.deepEqual(calls.at(-1).body.edits,{});
assert.deepEqual(calls.at(-1).body.comparison_shots,[]);
assert(text(targets['#design-merge-status']).includes('were cleared'));
""")


def test_pointer_selection_does_not_create_an_extra_joint():
    run_design(r"""
let calls=0;
const api=async(path)=>{
  if(path==='/api/design')return {data:[]};
  calls+=1; return {data:preview()};
};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openDesign(199607);
const svg=targets['#design-svg'];
byClass(svg,'design-point')[1].events.pointerdown({pointerId:1,preventDefault(){},stopPropagation(){}});
svg.events.pointerup({pointerId:1});
svg.events.click({clientX:550,clientY:110});
assert.equal(calls,1,'click retargeted after rerender only selects a handle');
assert.equal(byClass(svg,'design-point').length,3);
""")
