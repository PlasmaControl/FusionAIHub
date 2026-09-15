"""Exercise the actual timeline renderer without a server or external browser assets."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_rendered_source_bars_do_not_fill_a_gap_or_an_empty_interval_set():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to exercise the JavaScript renderer")
    app = Path(__file__).resolve().parents[2] / "src/shot_design/ui/static/app.js"
    script = r"""
const fs = require('fs');
const vm = require('vm');
class Element {
  constructor() { this.children = []; this.attrs = {}; }
  setAttribute(k, v) { this.attrs[k] = v; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener() {}
}
const targets = {};
const context = {Node: Element, document: {
  createElement: () => new Element(),
  querySelector: (id) => targets[id] ??= new Element(),
}};
let code = fs.readFileSync(process.argv[1], 'utf8');
code = code.slice(0, code.lastIndexOf('\ninit().catch'));
vm.createContext(context);
vm.runInContext(code, context);
vm.runInContext(`renderEvents({status: 'observed', coverage: {sources: [
  {source: 'elm_clock', status: 'ok', t_cov0_s: 0, t_cov1_s: 4,
   intervals: [[0, 1], [2, 4]], min_gap_s: .003},
  {source: 'empty', status: 'ok', t_cov0_s: 0, t_cov1_s: 4, intervals: []},
  {source: 'skipped', status: 'skipped', t_cov0_s: 0, t_cov1_s: 4, intervals: [[0, 4]]}
]}})`, context);
function marks(item) {
  if (!(item instanceof Element)) return [];
  return [ ...(item.attrs.class?.startsWith('mark') ? [item.attrs.style] : []),
           ...item.children.flatMap(marks) ];
}
process.stdout.write(JSON.stringify(marks(targets['#coverage-lanes'])));
"""
    result = subprocess.run([node, "-e", script, str(app)], capture_output=True, text=True,
                            check=True, timeout=20)
    styles = json.loads(result.stdout)
    assert len(styles) == 2
    assert styles[0].startswith("left:0%;width:25%;")
    assert styles[1].startswith("left:50%;width:50%;")
