"""Search-to-design behavior, using the browser's real controllers."""

from .test_design_browser import run_design
from .test_ui1_browser import run_dom


def test_result_selection_order_limit_explanations_and_no_row_fetches():
    run_dom(r"""
context.location = {hash: '#search'};
run(`api = async () => { throw Error('Search rows must not fetch shots'); }`);
const table = run(`resultsTable(Array.from({length:7}, (_, i) => ({
  shot:199600+i, score:.031234, run_id:'run', blurb:'Steady plasma',
  explanation:{channel_ranks:{bm25:2, scalar_knn:null, text_knn:1},
    matched_constraints:['ip_mean within range'], text_highlight:'low torque',
    top_similar:[['ip_mean',1e6,1.1e6]], top_different:[['q95',4,6]]},
  proposal_flags:[{message:'Review requested power'}], flags:[{message:'Recorded limit'}],
})), 'flat_top', 'Study low torque')`);
const boxes = all(table).filter(n => n.tag === 'input' && n.attrs.type === 'checkbox');
assert.equal(boxes.length,7);
const action = all(table).find(n => n.tag === 'button' && text(n) === 'Design from selected');
assert(action.disabled);
const choose = i => { boxes[i].checked = true; boxes[i].events.change(); };
choose(2); choose(0);
assert.equal(action.disabled,false);
for (const i of [1,3,4,5]) choose(i);
assert.equal(boxes[6].disabled,true);
boxes[0].checked = false; boxes[0].events.change();
assert.equal(boxes[6].disabled,false);
choose(6);
action.events.click();
const params = new URLSearchParams(context.location.hash.split('?')[1]);
assert.equal(context.location.hash.split('?')[0],'#design/199602');
assert.equal(params.get('comparisons'),'199601,199603,199604,199605,199606');
assert.equal(params.get('notes'),'Study low torque');
assert(!text(table).includes('.031234'));
for (const term of ['Rank','ip_mean within range','low torque','Review requested power',
                    'Recorded limit','q95']) assert(text(table).includes(term),term);
assert.equal(byClass(table,'retrieval-channel').length,14);
assert(!text(table).includes('scalar_knn'));
""")


def test_constraint_builder_validates_ranges_and_restores_hash_filters():
    run_dom(r"""
context.location = {hash:'#search'};
context.document.querySelectorAll = () => [];
const form = targets['#search-form'] = new Element('form');
form.elements = Object.fromEntries(['text','ref_shot','segment','n','require_labels','avoid_labels']
  .map(name => [name, new Element(name.includes('labels') ? 'select' : 'input')]));
run(`initSearchFilters({constraint_fields:[{name:'ip_mean',units:'A'},{name:'q95',units:''}],
  labels:['H','dud']})`);
const body = {text:'low torque & ECH',ref_shot:199607,segment:'full',n:7,
  constraints:{ip_mean:{lo:1e6,hi:2e6},q95:{hi:5}},require_labels:['H'],avoid_labels:['dud']};
context.body = body;
const hash = run('searchHash(body)');
context.location.hash = hash;
const calls = [];
context.fetchSearch = async (path, options) => {
  calls.push([path,JSON.parse(options.body)]); return {data:{results:[]}};
};
run('api = fetchSearch');
await run('route()');
assert.deepEqual(calls,[['/api/search',body]]);
assert.equal(form.elements.text.value,body.text);
assert.equal(form.elements.ref_shot.value,'199607');
assert.equal(form.elements.segment.value,'full');
assert.equal(form.elements.require_labels.children.find(n => n.attrs.value === 'H').selected,true);
assert.equal(form.elements.avoid_labels.children.find(n => n.attrs.value === 'dud').selected,true);
assert.deepEqual(JSON.parse(run('JSON.stringify(readConstraints())')),body.constraints);
const inputs = all(targets['#search-constraints']).filter(n => n.tag === 'input');
inputs[0].value = '3e6';
assert.throws(() => run('readConstraints()'), /Maximum must be at least minimum/);
inputs[0].value = 'NaN';
assert.throws(() => run('readConstraints()'), /Minimum must be a number/);
// Reload restores the same result request.
await run('route()');
assert.deepEqual(calls[1],calls[0]);
""")


def test_search_opens_editor_with_ordered_comparisons_and_query_notes():
    run_design(r"""
let sent;
const api = async (path, options = {}) => {
  if (path === '/api/design') return {data:[]};
  sent = JSON.parse(options.body);
  return {data:preview({program:sent})};
};
await context.ShotDesign.initDesign({api});
await context.ShotDesign.openDesign(199602, [199600,199605], 'Study low torque');
assert.equal(sent.reference_shot,199602);
assert.deepEqual(sent.comparison_shots,[199600,199605]);
assert.equal(sent.notes,'Study low torque');
assert.equal(targets['#design-reference'].value,'199602, 199600, 199605');
assert.equal(targets['#design-notes'].value,'Study low torque');
""")
