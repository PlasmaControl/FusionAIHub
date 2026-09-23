"""Assistant result actions and reload reattachment."""

from .test_assistant_browser import run_assistant
from .test_ui1_browser import run_dom


def test_exportable_result_simulates_and_shot_references_are_links():
    run_assistant(r"""
const simulated = [], opened = [], jobs = [];
context.ShotDesignAssistant.init({api:async () => ({data:complete({
  checks:{can_export:true,can_save:true,hdf5_valid:true,needs_seed:false}})}),
  onOpenDesign:id => opened.push(id), onSimulateDesign:id => simulated.push(id),
  onJobId:id => jobs.push(id)});
node('assistant-prompt').value = 'Control tearing modes';
await node('assistant-form').fire('submit');
assert.equal(node('assistant-edit').textContent,'Simulate');
await node('assistant-edit').fire('click');
assert.deepEqual(simulated,['design-88']);
assert.deepEqual(opened,[]);
assert.deepEqual(jobs,['job-17']);
const links = all(node('assistant-summary')).filter(n => n.tag === 'a');
assert.deepEqual(links.map(n => n.attrs.href),['#shot/199607','#shot/199608']);
assert.deepEqual(links.map(text),['199607','199608']);
""")


def test_unexportable_result_keeps_editor_action():
    run_assistant(r"""
const opened = [], simulated = [];
context.ShotDesignAssistant.init({api:async () => ({data:complete()}),
  onOpenDesign:id => opened.push(id), onSimulateDesign:id => simulated.push(id)});
node('assistant-prompt').value = 'Control tearing modes';
await node('assistant-form').fire('submit');
assert.equal(node('assistant-edit').textContent,'Open in actuator editor');
await node('assistant-edit').fire('click');
assert.deepEqual(opened,['design-88']);
assert.deepEqual(simulated,[]);
""")


def test_reload_attaches_running_job_then_displays_completed_result_without_post():
    run_assistant(r"""
const calls = [];
let reply = snapshot('running',{prompt:'Restore my experiment'});
context.ShotDesignAssistant.init({api:async (path, options = {}) => {
  calls.push([path,options.method]); return {data:reply};
}, onOpenDesign:() => {}});
await context.ShotDesignAssistant.attach('job-17');
assert.equal(node('assistant-submit').disabled,true);
assert.equal(node('assistant-prompt').value,'Restore my experiment');
assert.equal(node('assistant-stages').children.length,5);
reply = complete(); await tick();
assert.equal(node('assistant-result').hidden,false);
assert.equal(node('assistant-submit').disabled,false);
assert(calls.every(([path,method]) => path === '/api/design-assistant/job-17' && method !== 'POST'));
assert.equal(timers.size,0);
""")


def test_missing_job_is_explained_and_stops_polling():
    run_assistant(r"""
context.ShotDesignAssistant.init({api:async () => {
  throw Object.assign(Error('Unknown design assistant job'),{status:404});
}});
await context.ShotDesignAssistant.attach('gone');
assert(text(node('assistant-error')).includes('no longer on the server'));
assert.equal(node('assistant-submit').disabled,false);
assert.equal(node('assistant-retry').hidden,true);
assert.equal(node('assistant-result').hidden,true);
assert.equal(timers.size,0);
""")


def test_old_attachment_cannot_replace_newer_job():
    run_assistant(r"""
const pending = deferred();
context.ShotDesignAssistant.init({api:async path => {
  if (path.endsWith('/old')) return pending.promise;
  return {data:{...complete(),id:'new'}};
}});
const old = context.ShotDesignAssistant.attach('old');
await context.ShotDesignAssistant.attach('new');
pending.resolve({data:{...snapshot('running'),id:'old'}}); await old;
assert.equal(node('assistant-result').hidden,false);
assert.equal(node('assistant-download').attrs.href,'/api/design-assistant/new/hdf5');
assert.equal(timers.size,0);
""")


def test_assistant_job_hash_routes_and_preserves_other_views():
    run_dom(r"""
context.location = {hash:'#create/job-17'};
context.history = {replaceState(_state,_title,hash) {context.location.hash=hash;}};
context.document.querySelectorAll = () => [];
const attached = [];
context.ShotDesignAssistant = {attach:async id => attached.push(id)};
await run('route()');
assert.deepEqual(attached,['job-17']);
run("rememberAssistantJob('job-18')");
assert.equal(context.location.hash,'#create/job-18');
context.location.hash='#shot/199607';
run("rememberAssistantJob('job-19')");
assert.equal(context.location.hash,'#shot/199607');
""")


def test_simulation_action_opens_saved_design_once_without_reload_resubmission():
    run_dom(r"""
context.location = {hash:'#create/job-17'};
context.history = {pushState(_state,_title,hash) {context.location.hash=hash;}};
context.document.querySelectorAll = () => [];
const calls = [];
context.ShotDesign = {
  openSavedDesign:async id => calls.push(['open',id]),
  simulateDesign:async id => calls.push(['simulate',id]),
};
await run("simulateAssistantDesign('saved-id')");
assert.equal(context.location.hash,'#design-revision/saved-id');
assert.deepEqual(calls,[['open','saved-id'],['simulate','saved-id']]);
await run('route()');
assert.deepEqual(calls.at(-1),['open','saved-id']);
assert.equal(calls.filter(([action]) => action === 'simulate').length,1);
""")


def test_api_errors_keep_http_status_for_missing_job_messages():
    run_dom(r"""
context.fetch = async () => ({ok:false,status:404,text:async () =>
  JSON.stringify({detail:'Unknown design assistant job'})});
await assert.rejects(run("api('/api/design-assistant/gone')"), error =>
  error.status === 404 && error.message === 'Unknown design assistant job');
""")
