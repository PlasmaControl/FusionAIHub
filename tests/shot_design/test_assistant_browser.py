"""Executable browser contracts for the natural-language shot design workflow."""

import subprocess
from pathlib import Path

ASSISTANT = (
    Path(__file__).resolve().parents[2]
    / "src/shot_design/ui/static/assistant.js"
)


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
    this.focused = false;
  }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  getAttribute(name) { return this.attrs[name] ?? null; }
  removeAttribute(name) { delete this.attrs[name]; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener(name, fn) {
    (this.events[name] ||= []).push(fn);
  }
  async fire(name, values = {}) {
    const event = {preventDefault() {}, ...values};
    for (const fn of this.events[name] || []) await fn(event);
  }
  focus() { this.focused = true; }
  set textContent(value) { this.children = [String(value)]; }
  get textContent() { return this.children.map(text).join(''); }
  set innerHTML(_value) { throw Error('Unsafe HTML insertion'); }
}
const ids = [
  'assistant-form', 'assistant-prompt', 'assistant-model', 'assistant-submit',
  'assistant-status', 'assistant-stages', 'assistant-error', 'assistant-retry',
  'assistant-result', 'assistant-summary', 'assistant-explanation',
  'assistant-checks', 'assistant-edit', 'assistant-download', 'assistant-examples',
];
const targets = Object.fromEntries(ids.map(id => [id, new Element('div', id)]));
const node = id => targets[id];
const text = node => node instanceof Element ? node.children.map(text).join('') : String(node);
const all = node => node instanceof Element ? [node, ...node.children.flatMap(all)] : [];
const document = {
  querySelector: selector => targets[selector.slice(1)] || null,
  createElement: tag => new Element(tag),
};
const timers = new Map();
let nextTimer = 0;
const context = {
  document, console, AbortController,
  setTimeout: fn => { const id = ++nextTimer; timers.set(id, fn); return id; },
  clearTimeout: id => timers.delete(id),
};
context.globalThis = context;
context.window = context;
vm.createContext(context);
if (fs.existsSync(process.argv[1])) {
  vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
}
assert(context.ShotDesignAssistant, 'Natural-language shot design controller is missing');
const tick = async () => {
  const pending = [...timers.values()];
  timers.clear();
  await Promise.all(pending.map(fn => fn()));
};
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
};
const snapshot = (status = 'running', overrides = {}) => ({
  id: 'job-17', status,
  stages: [
    {key: 'interpret', label: 'Thinking about your needs', status: 'complete',
      detail: 'Tearing-mode control between 1 and 5 seconds.'},
    {key: 'retrieve', label: 'Searching for reference shots', status: 'running',
      detail: 'Searching observed tearing-mode evidence.'},
    {key: 'propose', label: 'Choosing actuation', status: 'pending', detail: ''},
    {key: 'validate', label: 'Checking waveforms', status: 'pending', detail: ''},
    {key: 'save', label: 'Saving HDF5', status: 'pending', detail: ''},
  ],
  result: null, error: null, ...overrides,
});
const complete = (overrides = {}) => snapshot('complete', {
  stages: snapshot().stages.map(stage => ({...stage, status: 'complete'})),
  result: {
    design_id: 'design-88', reference_shot: 199607, comparison_shots: [199608],
    explanation: 'Reduced beam power after 2 seconds.',
    checks: {errors: [], warnings: ['Review the proposed power ramp.'],
      can_export: false, can_save: true, needs_seed: true, hdf5_valid: true,
      available_channels: 12, total_channels: 88},
    artifact_path: '/private/generated/design-88.h5',
    hdf5_url: '/api/design-assistant/job-17/hdf5',
    model: 'quality', baseline: false, goal: {phenomenon: 'tearing'},
    ...overrides,
  },
});
"""


def run_assistant(script):
    result = subprocess.run(
        [
            "node", "-e",
            HARNESS + "\n(async () => {\n" + script
            + "\n})().catch(error => { console.error(error); process.exitCode = 1; });",
            str(ASSISTANT),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_enter_launches_once_and_displays_only_reported_progress():
    run_assistant(r"""
const calls = [], opened = [], submission = deferred();
let current = snapshot();
const api = async (path, options = {}) => {
  calls.push([path, options]);
  if (options.method === 'POST') return submission.promise;
  return {data: current};
};
context.ShotDesignAssistant.init({api, onOpenDesign: id => opened.push(id)});
node('assistant-prompt').value = '  Design a shot to control tearing modes.  ';
let prevented = 0;
const launch = node('assistant-prompt').fire('keydown', {
  key: 'Enter', preventDefault() { prevented++; },
});
assert.equal(prevented, 1);
assert.equal(node('assistant-submit').disabled, true);
assert.equal(node('assistant-result').hidden, true);
assert.equal(node('assistant-download').getAttribute('href'), null);
assert.equal(node('assistant-stages').children.length, 0);
await node('assistant-form').fire('submit');
assert.equal(calls.length, 1, 'Enter and form submit must not launch twice');
assert.equal(calls[0][0], '/api/design-assistant');
assert.deepEqual(JSON.parse(calls[0][1].body), {
  prompt: 'Design a shot to control tearing modes.', model: 'quality',
});
submission.resolve({data: current});
await launch;
let stages = node('assistant-stages').children;
assert.equal(stages.length, 5);
assert.equal(stages[1].getAttribute('data-state'), 'running');
assert.equal(stages[2].getAttribute('data-state'), 'pending');
assert(text(stages[1]).includes('Searching observed tearing-mode evidence.'));
await tick();
assert.equal(node('assistant-stages').children[2].getAttribute('data-state'), 'pending');
assert.equal(node('assistant-result').hidden, true);
current = complete();
await tick();
assert.equal(node('assistant-submit').disabled, false);
assert.equal(node('assistant-result').hidden, false);
assert(text(node('assistant-summary')).includes('199607'));
assert(text(node('assistant-summary')).includes('199608'));
assert(text(node('assistant-checks')).includes('12'));
assert(text(node('assistant-checks')).includes('88'));
assert(text(node('assistant-checks')).includes('Review the proposed power ramp.'));
assert(text(node('assistant-checks')).includes('seed'));
assert.equal(node('assistant-download').getAttribute('href'),
  '/api/design-assistant/job-17/hdf5');
assert.equal(node('assistant-download').hidden, false);
assert(!text(node('assistant-result')).includes('/private/'));
await node('assistant-edit').fire('click');
assert.deepEqual(opened, ['design-88']);
assert.equal(timers.size, 0);
""")


def test_prompt_validation_and_multiline_input_do_not_launch_jobs():
    run_assistant(r"""
const calls = [];
context.ShotDesignAssistant.init({api: async (path, options) => {
  calls.push([path, options]);
  return {data: snapshot()};
}});
node('assistant-prompt').value = '   ';
await node('assistant-form').fire('submit');
assert.equal(calls.length, 0);
assert(text(node('assistant-error')).length > 0);
assert.equal(node('assistant-prompt').focused, true);
node('assistant-prompt').value = 'x'.repeat(4001);
await node('assistant-form').fire('submit');
assert.equal(calls.length, 0);
assert(text(node('assistant-error')).includes('4000'));
node('assistant-prompt').value = 'Design an ELM-control shot.';
await node('assistant-prompt').fire('keydown', {key: 'Enter', shiftKey: true});
await node('assistant-prompt').fire('keydown', {key: 'Enter', isComposing: true});
assert.equal(calls.length, 0);
node('assistant-model').value = 'fast';
await node('assistant-form').fire('submit');
assert.equal(JSON.parse(calls[0][1].body).model, 'fast');
assert.equal(text(node('assistant-error')), '');
""")


def test_start_failure_can_retry_after_configuration_is_fixed():
    run_assistant(r"""
const calls = [];
let configured = false;
context.ShotDesignAssistant.init({api: async (path, options) => {
  calls.push([path, options]);
  if (!configured) throw Error('Gemma unavailable. Configure SHOT_DESIGN_LLM_URL.');
  return {data: complete()};
}});
node('assistant-prompt').value = 'Control tearing modes.';
await node('assistant-form').fire('submit');
assert(text(node('assistant-error')).includes('SHOT_DESIGN_LLM_URL'));
assert.equal(node('assistant-submit').disabled, false);
assert.equal(node('assistant-retry').hidden, false);
assert.equal(node('assistant-result').hidden, true);
assert.equal(timers.size, 0);
configured = true;
await node('assistant-retry').fire('click');
assert.equal(calls.length, 2);
assert.equal(JSON.parse(calls[1][1].body).prompt, 'Control tearing modes.');
assert.equal(node('assistant-result').hidden, false);
assert.equal(node('assistant-retry').hidden, true);
assert.equal(text(node('assistant-error')), '');
""")


def test_failed_stage_is_shown_without_fabricated_completion_or_download():
    run_assistant(r"""
const failed = snapshot('failed', {
  stages: snapshot().stages.map(stage => stage.key === 'retrieve' ?
    {...stage, status: 'failed', detail: '<script>unsafe()</script>'} : stage),
  error: 'No reference shot has the required waveforms. Try a different time window.',
});
context.ShotDesignAssistant.init({api: async () => ({data: failed})});
node('assistant-prompt').value = 'Control tearing modes.';
await node('assistant-form').fire('submit');
assert.equal(node('assistant-stages').children[1].getAttribute('data-state'), 'failed');
assert(text(node('assistant-stages')).includes('<script>unsafe()</script>'));
assert.equal(node('assistant-stages').children[4].getAttribute('data-state'), 'pending');
assert(text(node('assistant-error')).includes('Try a different time window'));
assert.equal(node('assistant-submit').disabled, false);
assert.equal(node('assistant-retry').hidden, false);
assert.equal(node('assistant-result').hidden, true);
assert.equal(node('assistant-download').getAttribute('href'), null);
assert.equal(timers.size, 0);
""")


def test_lost_connection_rechecks_existing_job_without_starting_duplicate():
    run_assistant(r"""
const calls = [];
let online = false;
context.ShotDesignAssistant.init({api: async (path, options = {}) => {
  calls.push([path, options]);
  if (options.method === 'POST') return {data: snapshot()};
  if (!online) throw Error('Network request failed');
  return {data: complete()};
}});
node('assistant-prompt').value = 'Control Alfven eigenmodes.';
await node('assistant-form').fire('submit');
await tick();
assert.equal(node('assistant-submit').disabled, false);
assert.equal(node('assistant-retry').hidden, false);
assert(text(node('assistant-error')).includes('Network request failed'));
assert(text(node('assistant-status')).includes('running'));
assert.equal(timers.size, 0);
online = true;
await node('assistant-retry').fire('click');
assert.equal(calls.filter(([_path, options]) => options.method === 'POST').length, 1);
assert.equal(calls[2][0], '/api/design-assistant/job-17');
assert.equal(node('assistant-result').hidden, false);
assert.equal(node('assistant-retry').hidden, true);
""")


def test_reinitialization_aborts_old_poll_and_ignores_its_late_response():
    run_assistant(r"""
const pending = deferred();
let oldSignal;
context.ShotDesignAssistant.init({api: async (_path, options = {}) => {
  if (options.method === 'POST') return {data: snapshot()};
  oldSignal = options.signal;
  return pending.promise;
}});
node('assistant-prompt').value = 'Old goal';
await node('assistant-form').fire('submit');
const oldPoll = tick();
let newCalls = 0;
context.ShotDesignAssistant.init({api: async () => {
  newCalls++;
  return {data: {...complete({design_id: 'new-design', reference_shot: 200001}),
    id: 'new-job'}};
}});
assert.equal(oldSignal.aborted, true);
node('assistant-prompt').value = 'New goal';
await node('assistant-form').fire('submit');
assert.equal(newCalls, 1, 'Reinitializing must not duplicate event handlers');
pending.resolve({data: complete({explanation: 'STALE RESULT'})});
await oldPoll;
assert(text(node('assistant-summary')).includes('200001'));
assert(!text(node('assistant-explanation')).includes('STALE RESULT'));
assert.equal(node('assistant-download').getAttribute('href'),
  '/api/design-assistant/new-job/hdf5');
assert.equal(timers.size, 0);
""")


def test_examples_fill_prompt_without_starting_and_results_remain_literal_text():
    run_assistant(r"""
let requests = 0;
context.ShotDesignAssistant.init({api: async () => {
  requests++;
  return {data: complete({explanation: '<img src=x onerror=alert(1)>',
    hdf5_url: 'javascript:alert(1)'})};
}, onOpenDesign: async () => { throw Error('Revision could not be loaded'); }});
const examples = node('assistant-examples').children;
assert.equal(examples.length, 3);
await examples[1].fire('click');
assert(node('assistant-prompt').value.toLowerCase().includes('eigenmode'));
assert.equal(node('assistant-prompt').focused, true);
assert.equal(requests, 0);
await node('assistant-form').fire('submit');
assert.equal(text(node('assistant-explanation')), '<img src=x onerror=alert(1)>');
assert.equal(node('assistant-download').getAttribute('href'),
  '/api/design-assistant/job-17/hdf5');
await node('assistant-edit').fire('click');
assert(text(node('assistant-error')).includes('Revision could not be loaded'));
assert.equal(node('assistant-edit').disabled, false);
assert.equal(node('assistant-download').hidden, false);
""")


def test_invalid_completion_keeps_download_hidden_and_stops_polling():
    run_assistant(r"""
context.ShotDesignAssistant.init({api: async () => ({data: snapshot('complete')})});
node('assistant-prompt').value = 'Control ELMs.';
await node('assistant-form').fire('submit');
assert.equal(node('assistant-result').hidden, true);
assert.equal(node('assistant-download').getAttribute('href'), null);
assert.equal(node('assistant-submit').disabled, false);
assert(text(node('assistant-error')).length > 0);
assert.equal(timers.size, 0);
""")
