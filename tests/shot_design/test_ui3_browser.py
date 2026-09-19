"""UI3 rendered lanes, scoring, and numeric form submission regressions."""

from html.parser import HTMLParser

import pytest

from .test_ui1_browser import APP, run_dom, run_js


class Forms(HTMLParser):
    def __init__(self):
        super().__init__()
        self.form = None
        self.fields = {}
        self.tabs = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form":
            self.form = attrs["id"]
        if tag == "input":
            self.fields[self.form, attrs["name"]] = attrs
        if "data-view" in attrs:
            self.tabs.append(attrs["data-view"])


def test_numeric_form_fields_are_text_with_explicit_keyboard_modes():
    forms = Forms()
    forms.feed(APP.with_name("index.html").read_text())
    identifiers = [("search-form", "ref_shot"), ("shot-form", "shot")]
    decimals = [("search-form", "n"), ("events-form", "t0_s"),
                ("events-form", "t1_s"), ("events-form", "min_confidence"),
                ("locate-form", "n"), ("locate-form", "min_confidence")]
    for key in identifiers + decimals:
        field = forms.fields[key]
        assert field["type"] == "text", key
        assert field["inputmode"] == ("numeric" if key in identifiers else "decimal")
        if key in identifiers:
            assert field["pattern"] == "[0-9]*" and field["autocomplete"] == "off"
    assert not any(field.get("type") == "number" for field in forms.fields.values())
    assert forms.tabs == ["create", "search", "shot", "locate", "design", "info"]


@pytest.mark.parametrize("prefix", ["", "context-"])
def test_phenomenon_lanes_precede_sources_and_forecasts_stay_separate(prefix):
    run_dom(r"""
const prefix = PREFIX;
run(`renderEvents({status:'observed', domain:{t0_s:-2,t1_s:8},
  events:[{source:'track',phenomenon:'coherent_mode',evidence_kind:'detector',t0_s:1,t1_s:3}],
  forecasts:[{source:'risk',phenomenon:'tearing',evidence_kind:'forecast',t0_s:3,t1_s:4}],
  phenomena:[{id:'eho',title:'Edge harmonic oscillation',intervals:[
    {source:'track',evidence_kind:'detector',t0_s:-4,t1_s:3,f0_khz:2,f1_khz:20,confidence:.89123}],
    forecast_intervals:[],caveats:['Coverage unrecorded; absence unmeasured']},
    {id:'tearing',title:'Tearing mode',intervals:[],forecast_intervals:[
      {source:'risk',evidence_kind:'forecast',t0_s:3,t1_s:10,confidence:.7}],caveats:[]},
    {id:'rwm',title:'Resistive wall mode',intervals:[],forecast_intervals:[],caveats:[]}]
}, '${prefix}')`);
const events = targets[`#${prefix}event-lanes`];
const forecast = targets[`#${prefix}forecast-lanes`];
const groups = byClass(events, 'phenomenon-lanes');
assert.equal(groups.length, 1);
assert(text(groups[0]).includes('Phenomena'));
const lanes = byClass(events, 'lane');
assert.equal(lanes.length, 4, 'three phenomenon lanes, then one source lane');
assert(text(lanes[0]).includes('Edge harmonic oscillation'));
assert(text(byClass(lanes[0], 'muted')[0]).includes('eho'));
assert(text(lanes[3]).includes('track'));
assert.equal(byClass(lanes[0], 'mark').length, 1);
assert.equal(byClass(lanes[1], 'mark').length, 0, 'forecast is never observed');
const mark = byClass(lanes[0], 'mark')[0];
for (const word of ['Edge harmonic oscillation', 'track', 'detector', '-4.000–3.000 s',
                    '0.891', '2–20 kHz', 'Coverage unrecorded']) assert(mark.attrs.title.includes(word), word);
assert(mark.attrs.class.includes('clipped-left'));
assert(mark.attrs.title.includes('drawn -2–3 s'));
assert(!text(lanes[0]).includes('kHz'), 'frequency is tooltip-only');
const forecastLanes = byClass(forecast, 'lane');
assert.equal(forecastLanes.length, 2);
assert(text(forecastLanes[0]).includes('Tearing mode'));
assert(text(forecastLanes[1]).includes('risk'));
assert(byClass(forecastLanes[0], 'mark')[0].attrs.title.includes('forecast'));
assert(byClass(forecastLanes[0], 'mark')[0].attrs.class.includes('clipped-right'));
assert.equal(byClass(events, 'axis').length, 2);
assert.equal(byClass(events, 'timeline')[0].children[0].attrs.class, 'axis');
assert.equal(byClass(events, 'section-toggle').length, 1);
""".replace("PREFIX", repr(prefix)))


def test_scoring_tab_renders_endpoint_values_including_default_weights():
    run_dom(r"""
const fixture = {method:'Weighted reciprocal rank fusion',formula:'score = Σ_c w_c / (k0 + rank_c)',
  k0:41,score_range:[.0123,.0432],channels:[
    {name:'scalar_knn',weight:1.7,compares:'Segment scalar embedding of the reference shot'},
    {name:'text_knn',weight:2.5,compares:'MiniLM text'},
    {name:'bm25',weight:.3,compares:'Exact words'},
    {name:'ignite_knn',weight:.8,compares:'IGNITE codec embeddings'},
    {name:'phenomenon',weight:1.3,compares:'Resolved evidence'},
    {name:'new_channel',weight:1,weight_source:'default',compares:'New channel'}],
  dedup_threshold:.82,run_diversity_decay:.72,outcome_penalty:.31,
  hard_filters:['constraints','segment','require labels','avoid labels'],
  phenomenon:{formula:'score = label × max_p + event × (1 − exp(−n_events / saturation_n)) + text × tanh(hits / 2) + database',
    weights:{label:1.1,event:1.7,text:.6,database:1.2},saturation_n:7.5,
    class_order:['OBSERVED','LABELLED','FORECAST','DATABASE','TEXTUAL'],text_only_ceiling:.23},
  db:{n_shots:123,shot_range:[199607,200003],git_sha:'build-abc',built:'2026-09-15'}};
context.fixture = fixture;
run('renderScoring(fixture)');
const root = targets['#info-content'];
for (const heading of ['Search score','Phenomenon score','Database']) assert(text(root).includes(heading));
const rows = all(root).filter(n => n.tag === 'tr');
for (const channel of fixture.channels) {
  const row = rows.find(row => text(row).includes(channel.name));
  assert(row, channel.name);
  assert(text(row).includes(String(channel.weight)));
  assert(text(row).includes(channel.compares));
}
assert(text(rows.find(row => text(row).includes('new_channel'))).includes('default'));
for (const value of ['41','.82','.72','.31','7.5','.23','build-abc','199607–200003','2026-09-15']) {
  assert(text(root).includes(value), value);
}
assert(text(root).includes('OBSERVED > LABELLED > FORECAST > DATABASE > TEXTUAL'));
assert(text(root).includes('rank_c'));
assert(text(root).includes('saturation_n'));
assert(!text(root).includes('k0 = 60'));
for (const w of Object.values(fixture.phenomenon.weights)) assert(text(root).includes(String(w)));
context.location = {hash:'#info'};
context.document.querySelectorAll = () => [];
run(`api = async path => { if(path !== '/api/scoring') throw Error(path); return {data:fixture}; }`);
await run('route()');
assert(text(targets['#info-content']).includes('build-abc'));
""")


def test_explicit_numeric_parsing_accepts_decimals_and_rejects_invalid_identifiers():
    run_dom(r"""
const parse = (value, options = {}) => {
  context.value = value; context.options = options;
  return run("parseNumberField(value, 'Shot', options)");
};
for (const [value, expected] of [['-1.25',-1.25],['.75',.75],['1e-3',.001],['0',0]]) {
  assert.equal(parse(value), expected);
}
assert.equal(parse('  '), null);
assert.equal(parse('199607', {identifier:true}), 199607);
for (const value of ['199607.1','199607.0','2e5','-1','+2','1x','9007199254740993']) {
  assert.throws(() => parse(value,{identifier:true}), /Shot must be a whole number/);
}
for (const value of ['NaN','Infinity','0x10','1,5','1abc','1e999']) {
  assert.throws(() => parse(value), /Shot must be a number/);
}
assert.throws(() => parse('',{required:true}), /Shot is required/);
assert.throws(() => parse('1.5',{integer:true}), /Shot must be a whole number/);
assert.throws(() => parse('0',{min:1}), /Shot must be at least 1/);
assert.throws(() => parse('1.1',{min:0,max:1}), /Shot must be between 0 and 1/);
""")


def test_form_submission_parses_values_reports_errors_and_never_rewrites_typing():
    run_dom(r"""
// Minimal form controls around the real init/bindForm/route/loadEvents functions.
const form = (id, values) => {
  const node = targets[id] = new Element('form');
  node.elements = Object.fromEntries(Object.entries(values).map(([name,value]) => [name,{value}]));
  node.button = new Element('button'); node.error = new Element('p');
  node.querySelector = sel => sel === '.form-error' ? node.error : node.button;
  return node;
};
const search = form('#search-form', {text:'',ref_shot:'00199607',segment:'flat_top',n:'12',
  constraints:'',require_labels:'',avoid_labels:''});
const shot = form('#shot-form', {shot:'199607',segment:'flat_top'});
const events = form('#events-form', {phenomenon:'tearing',t0_s:'-1.25',t1_s:'2.5',min_confidence:'.75'});
const locate = form('#locate-form', {phenomenon:'tearing',segment:'flat_top',n:'7',min_confidence:'.625',avoid:''});
context.FormData = class extends Map {
  constructor(form) { super(Object.entries(form.elements).map(([key,field]) => [key,field.value])); }
};
context.location = {hash:'#search'};
context.window = {addEventListener(){}};
context.document.querySelectorAll = () => [];
const calls = [];
let pending;
context.fetchApi = async (path, options = {}) => {
  calls.push([path,options]);
  if (path === '/api/meta') return {data:{segments:[]}};
  if (path === '/api/phenomena') return {data:[]};
  if (path === '/api/search' && pending) return new Promise(resolve => pending.resolve = resolve);
  return {data:path.startsWith('/api/locate') ? [] : {events:[],forecasts:[]}};
};
run('api = fetchApi');
await run('init()'); calls.length = 0;
const submit = node => node.events.submit({target:node,preventDefault(){}});
pending = {};
const searching = submit(search);
assert.equal(JSON.parse(calls[0][1].body).ref_shot, 199607);
assert.equal(JSON.parse(calls[0][1].body).n, 12);
assert.equal(search.elements.ref_shot.value, '00199607');
search.elements.ref_shot.value = '199608';
pending.resolve({data:{results:[]}}); await searching; pending = null;
assert.equal(search.elements.ref_shot.value, '199608', 'response must not overwrite typing');
for (const [node, key, value, message] of [
  [search,'ref_shot','12.5','Reference shot must be a whole number'],
  [shot,'shot','1e5','Shot must be a whole number'],
  [locate,'n','2.5','Hits must be a whole number'],
  [locate,'min_confidence','NaN','Minimum confidence must be a number'],
]) {
  const original = node.elements[key].value; node.elements[key].value = value;
  const count = calls.length;
  await submit(node);
  assert.equal(calls.length, count, 'invalid input must not make a request');
  assert.equal(text(node.error), message);
  assert.equal(node.elements[key].value, value);
  assert.equal(node.button.disabled, false);
  node.elements[key].value = original;
}
await submit(locate);
const params = new URLSearchParams(calls.at(-1)[0].split('?')[1]);
assert.equal(params.get('n'),'7'); assert.equal(params.get('min_confidence'),'0.625');
assert.equal(locate.elements.min_confidence.value,'.625');
run('S.shot = 199607');
await submit(events);
for (const [path] of calls.slice(-2)) {
  const params = new URLSearchParams(path.split('?')[1]);
  assert.equal(params.get('t0_s'),'-1.25'); assert.equal(params.get('t1_s'),'2.5');
  assert.equal(params.get('min_confidence'),'0.75');
}
assert.equal(events.elements.min_confidence.value,'.75');
events.elements.t1_s.value = '-2';
const count = calls.length; await submit(events);
assert.equal(calls.length,count); assert.equal(text(events.error),'End must be after start');
assert.equal(events.elements.t1_s.value,'-2');
""")


def test_filtered_confidence_caveats_use_shared_confidence_formatter():
    assert run_js("input.map(caption)", [
        "2 events excluded: confidence below 0.1234567 or unrecorded",
        "3 forecasts excluded: confidence below 1e-05 or unrecorded",
    ]) == [
        "2 events excluded: confidence below 0.123 or unrecorded",
        "3 forecasts excluded: confidence below 0.000 or unrecorded",
    ]
