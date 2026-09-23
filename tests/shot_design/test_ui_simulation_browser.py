"""Simulation results remain attached to saved design revisions."""

import json

from .test_design_browser import run_design
from .test_ui_metrics import METRICS

SETUP = r"""
context.controlTimers = true;
context.confirm = () => true;
context.metrics = METRICS;
const calls = [];
let simulationState = {state:'complete'};
let revision = 'a'.repeat(32);
const api = async (path, options = {}) => {
  calls.push([path, options]);
  if (path === '/api/design') return {data:[]};
  if (path.endsWith('/simulate/metrics')) return {data:context.metrics};
  if (path.endsWith('/simulate')) {
    if (options.method === 'POST') {
      simulationState = {state:'queued',submitted:new Date().toISOString()};
      return {data:{job_id:'42'}};
    }
    return {data:simulationState};
  }
  if (path === '/api/design/preview') return {data:preview({program:JSON.parse(options.body)})};
  return {data:preview({program:{...preview().program,id:revision}})};
};
await context.ShotDesign.initDesign({api, pollMs:5000});
""".replace('METRICS', json.dumps(METRICS))


def test_reopen_renders_f6_metrics_negative_skill_unresolved_effect_and_panels():
    run_design(SETUP + r"""
await context.ShotDesign.openSavedDesign(revision);
assert(calls.some(([path]) => path === `/api/design/${revision}/simulate`));
const results = targets['#design-simulation-results'];
for (const term of ['1.21','1.05','-0.08','0.12','0.1','10-60 kHz band power',
  'Error vs measured','Hold-last-frame error','Skill vs holding the last frame',
  'Edit effect','Run-to-run noise','Effect resolved?','No — unresolved',
  'worse than the hold-last-frame baseline',revision]) assert(text(results).includes(term),term);
assert.equal((text(results).match(/co2/g) || []).length,1);
assert(text(results).includes('not simulated (diagnostic absent)'));
const images = all(results).filter(n => n.tag === 'img');
assert.equal(images.length,1);
assert.equal(images[0].attrs.src,`/api/design/${revision}/simulate/panels/mhr.png`);
assert.equal(targets['#design-simulate'].textContent,'Run again');
assert.equal(targets['#design-simulate-report'].hidden,false);
""")


def test_edits_preserve_results_with_revision_and_disable_submission():
    run_design(SETUP + r"""
await context.ShotDesign.openSavedDesign(revision);
targets['#design-notes'].value = 'Changed plan';
targets['#design-notes'].events.input();
assert(text(targets['#design-simulation-results']).includes(revision));
assert(text(targets['#design-simulation-results']).includes('Unsaved edits are not included'));
assert.equal(targets['#design-simulate-report'].hidden,false);
assert.equal(targets['#design-simulate'].disabled,true);
await targets['#design-simulate'].events.click({preventDefault(){}});
assert(!calls.some(([,options]) => options.method === 'POST'));
// An async preview no longer erases the displayed result either.
await targets['#design-preview'].events.click({preventDefault(){}});
assert(text(targets['#design-simulation-results']).includes('-0.08'));
assert.equal(targets['#design-simulate'].disabled,true);
""")


def test_rerun_needs_confirmation_and_prevents_double_submission():
    run_design(SETUP + r"""
await context.ShotDesign.openSavedDesign(revision);
let confirmations = 0;
context.confirm = message => { confirmations++; assert(message.includes(revision)); return false; };
await targets['#design-simulate'].events.click({preventDefault(){}});
assert(!calls.some(([,options]) => options.method === 'POST'));
context.confirm = () => { confirmations++; return true; };
await targets['#design-simulate'].events.click({preventDefault(){}});
await targets['#design-simulate'].events.click({preventDefault(){}});
assert.equal(confirmations,2);
assert.equal(calls.filter(([,options]) => options.method === 'POST').length,1);
assert.equal(targets['#design-simulate'].disabled,true);
assert(text(targets['#design-simulate-status']).includes('Queued'));
assert(text(targets['#design-simulation-results']).includes('-0.08'));
""")


def test_running_elapsed_time_and_other_revision_do_not_share_results():
    run_design(SETUP + r"""
await context.ShotDesign.openSavedDesign(revision);
revision = 'b'.repeat(32);
simulationState = {state:'running',started:new Date(Date.now()-125000).toISOString()};
await context.ShotDesign.openSavedDesign(revision);
assert.equal(targets['#design-simulate'].disabled,true);
assert(text(targets['#design-simulate-status']).includes('2m 5s elapsed'));
assert(!text(targets['#design-simulation-results']).includes('-0.08'));
assert.equal(targets['#design-simulate-report'].hidden,true);
revision = 'c'.repeat(32); simulationState = {state:'not_started'};
await context.ShotDesign.openSavedDesign(revision);
assert.equal(targets['#design-simulate'].textContent,'Simulate');
assert.equal(targets['#design-simulate'].disabled,false);
assert.equal(context.pendingTimers(),0);
""")


def test_missing_metrics_are_explained_without_hiding_report():
    run_design(SETUP + r"""
await context.ShotDesign.initDesign({api:async (path, options) => {
  if (path.endsWith('/metrics')) throw Error('No simulation metrics for this design yet');
  return api(path,options);
}});
await context.ShotDesign.openSavedDesign(revision);
assert(text(targets['#design-simulation-results']).includes('No simulation metrics for this design yet'));
assert.equal(targets['#design-simulate-report'].hidden,false);
""")


def test_unreadable_status_does_not_offer_to_submit_an_unknown_job():
    run_design(SETUP + r"""
await context.ShotDesign.initDesign({api:async (path, options) => {
  if (path.endsWith('/simulate')) throw Error('Connection lost');
  return api(path,options);
}});
await context.ShotDesign.openSavedDesign(revision);
assert(text(targets['#design-simulate-status']).includes('Status check failed: Connection lost'));
assert.equal(targets['#design-simulate'].disabled,true);
""")


def test_old_status_reply_cannot_replace_another_revision():
    run_design(SETUP + r"""
let resolveOld;
const oldId = revision;
await context.ShotDesign.initDesign({api:async (path, options) => {
  if (path === `/api/design/${oldId}/simulate`) return new Promise(resolve => resolveOld = resolve);
  return api(path,options);
}});
const opening = context.ShotDesign.openSavedDesign(oldId);
await new Promise(resolve => setImmediate(resolve));
revision = 'b'.repeat(32); simulationState = {state:'not_started'};
await context.ShotDesign.openSavedDesign(revision);
resolveOld({data:{state:'complete'}}); await opening;
assert.equal(targets['#design-simulation-results'].hidden,true);
assert.equal(targets['#design-simulate-report'].hidden,true);
assert.equal(targets['#design-simulate'].textContent,'Simulate');
assert.equal(targets['#design-simulate'].disabled,false);
""")


def test_submission_failure_keeps_previous_result_and_allows_retry():
    run_design(SETUP + r"""
await context.ShotDesign.initDesign({api:async (path, options) => {
  if (options?.method === 'POST') throw Error('Queue unavailable');
  return api(path,options);
}});
await context.ShotDesign.openSavedDesign(revision);
await targets['#design-simulate'].events.click({preventDefault(){}});
assert(text(targets['#design-simulate-status']).includes('Submit failed: Queue unavailable'));
assert.equal(targets['#design-simulate'].disabled,false);
assert.equal(targets['#design-simulate'].textContent,'Run again');
assert(text(targets['#design-simulation-results']).includes('-0.08'));
""")


def test_pending_rerun_shows_submitting_instead_of_previous_completion():
    run_design(SETUP + r"""
let submitted;
await context.ShotDesign.initDesign({api:async (path, options) => {
  if (options?.method === 'POST') return new Promise(resolve => submitted = resolve);
  return api(path,options);
}});
await context.ShotDesign.openSavedDesign(revision);
const pending = targets['#design-simulate'].events.click({preventDefault(){}});
assert.equal(targets['#design-simulate-status'].textContent,'Submitting…');
assert.equal(targets['#design-simulate'].disabled,true);
assert(text(targets['#design-simulation-results']).includes('-0.08'));
simulationState = {state:'queued',submitted:new Date().toISOString()};
submitted({data:{job_id:'42'}}); await pending;
assert(text(targets['#design-simulate-status']).includes('Queued'));
""")
