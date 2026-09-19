// Natural-language shot design. Progress comes exclusively from the server job.
(function (scope) {
  "use strict";

  const MAX_PROMPT_LENGTH = 4000;
  const EXAMPLES = [
    ["Tearing mode control", "Design a shot to study tearing mode control. Find suitable reference shots and propose actuator waveforms."],
    ["Alfvén eigenmode control", "Design a shot to study Alfvén eigenmode control. Find suitable reference shots and propose actuator waveforms."],
    ["ELM control", "Design a shot to study edge-localized mode (ELM) control. Find suitable reference shots and propose actuator waveforms."],
  ];
  const state = {
    api: null,
    onOpenDesign: null,
    initialized: false,
    busy: false,
    generation: 0,
    controller: null,
    timer: null,
    jobId: null,
    result: null,
    retryMode: null,
    examples: [],
  };
  const node = (id) => document.querySelector(`#assistant-${id}`);

  function element(tag, text, className) {
    const result = document.createElement(tag);
    result.textContent = text;
    if (className) result.setAttribute("class", className);
    return result;
  }

  function setBusy(busy) {
    state.busy = busy;
    node("form").setAttribute("aria-busy", String(busy));
    node("submit").disabled = busy;
    node("model").disabled = busy;
    node("retry").disabled = busy;
    for (const button of state.examples) button.disabled = busy;
  }

  function clearResult() {
    state.result = null;
    node("result").hidden = true;
    node("download").hidden = true;
    node("download").removeAttribute("href");
    node("edit").disabled = true;
  }

  function renderStages(stages) {
    node("stages").replaceChildren();
    for (const stage of stages || []) {
      const row = element("li", "", "assistant-stage");
      row.setAttribute("data-state", stage.status);
      const heading = element("div", "", "assistant-stage-heading");
      heading.append(element("strong", stage.label),
        element("span", stage.status, "assistant-stage-state"));
      row.append(heading);
      if (stage.detail) row.append(element("p", stage.detail, "assistant-stage-detail"));
      node("stages").append(row);
    }
  }

  function renderResult(result) {
    state.result = result;
    const comparisons = result.comparison_shots || [];
    node("summary").textContent = `Reference shots: ${[result.reference_shot, ...comparisons].join(", ")}` +
      ` · ${result.reference_shot} supplies the initial state`;
    node("explanation").textContent = result.explanation || "";
    const checks = result.checks || {};
    const list = element("ul", "", "assistant-checks-list");
    for (const error of checks.errors || []) list.append(element("li", error, "assistant-error"));
    for (const warning of checks.warnings || []) list.append(element("li", warning, "assistant-warning"));
    if (Number.isFinite(checks.available_channels) && Number.isFinite(checks.total_channels)) {
      list.append(element("li", `${checks.available_channels} of ${checks.total_channels} actuator channels available.`));
    }
    if (checks.hdf5_valid) list.append(element("li", "HDF5 saved and checked.", "assistant-ok"));
    if (checks.needs_seed) list.append(element("li",
      "IGNITE simulation export still needs a prepared seed. Open the waveform editor to prepare it.",
      "assistant-warning"));
    node("checks").replaceChildren(list);
    node("download").setAttribute("href", `/api/design-assistant/${encodeURIComponent(state.jobId)}/hdf5`);
    node("download").hidden = checks.hdf5_valid === false;
    if (checks.hdf5_valid === false) node("download").removeAttribute("href");
    node("edit").disabled = !state.onOpenDesign;
    node("result").hidden = false;
  }

  function receive(snapshot, generation) {
    if (generation !== state.generation) return;
    if (!snapshot || typeof snapshot.id !== "string" || !snapshot.id ||
      !["queued", "running", "complete", "failed"].includes(snapshot.status) ||
      !Array.isArray(snapshot.stages) || (state.jobId && snapshot.id !== state.jobId)) {
      throw new Error("The server returned an invalid design status. Retry to check the workflow.");
    }
    state.jobId = snapshot.id;
    renderStages(snapshot.stages);
    node("error").textContent = "";
    node("retry").hidden = true;
    state.retryMode = null;
    if (snapshot.status === "complete") {
      if (!snapshot.result?.design_id) {
        fail("The workflow finished without a saved revision. Retry the design.", generation, "start");
        return;
      }
      renderResult(snapshot.result);
      node("status").textContent = "Your shot design is ready. Review the waveforms or download the HDF5 file.";
      setBusy(false);
      return;
    }
    if (snapshot.status === "failed") {
      fail(snapshot.error || "The design workflow failed. Update your request and retry.", generation, "start");
      return;
    }
    const active = snapshot.stages?.find((stage) => stage.status === "running");
    node("status").textContent = active?.label || "Waiting for the design workflow to start.";
    state.timer = setTimeout(() => poll(generation), 1000);
  }

  async function poll(generation) {
    if (generation !== state.generation) return;
    state.timer = null;
    try {
      const { data } = await state.api(`/api/design-assistant/${encodeURIComponent(state.jobId)}`,
        { signal: state.controller.signal });
      receive(data, generation);
    } catch (error) {
      fail(error.message || String(error), generation, "poll");
    }
  }

  function fail(message, generation, retryMode) {
    if (generation !== state.generation) return;
    if (state.timer !== null) clearTimeout(state.timer);
    state.timer = null;
    state.retryMode = retryMode;
    node("error").textContent = message;
    node("status").textContent = retryMode === "poll" ?
      "Progress updates stopped. The design may still be running; reconnect to check its status." :
      "The design could not finish. Review the message, adjust your request if needed, and retry.";
    node("retry").textContent = retryMode === "poll" ? "Reconnect to design" : "Retry design";
    node("retry").hidden = false;
    setBusy(false);
  }

  function newRequest() {
    state.generation += 1;
    state.controller?.abort();
    if (state.timer !== null) clearTimeout(state.timer);
    state.timer = null;
    state.controller = new AbortController();
    return state.generation;
  }

  async function submit(event) {
    event?.preventDefault();
    if (state.busy) return;
    const prompt = node("prompt").value.trim();
    if (!prompt || prompt.length > MAX_PROMPT_LENGTH) {
      node("error").textContent = !prompt ? "Describe the shot you want to design." :
        `Keep your request within ${MAX_PROMPT_LENGTH} characters.`;
      node("prompt").focus();
      return;
    }
    const generation = newRequest();
    state.jobId = null;
    state.retryMode = null;
    clearResult();
    node("stages").replaceChildren();
    node("error").textContent = "";
    node("retry").hidden = true;
    node("status").textContent = "Starting your shot design…";
    setBusy(true);
    try {
      const { data } = await state.api("/api/design-assistant", {
        method: "POST", signal: state.controller.signal,
        body: JSON.stringify({ prompt, model: node("model").value === "fast" ? "fast" : "quality" }),
      });
      receive(data, generation);
    } catch (error) {
      fail(error.message || String(error), generation, "start");
    }
  }

  async function retry(event) {
    event.preventDefault();
    if (state.busy) return;
    if (state.retryMode !== "poll" || !state.jobId) return submit(event);
    const generation = newRequest();
    node("error").textContent = "";
    node("retry").hidden = true;
    node("status").textContent = "Reconnecting to your design…";
    setBusy(true);
    return poll(generation);
  }

  async function openDesign(event) {
    event.preventDefault();
    if (!state.result?.design_id || !state.onOpenDesign || node("edit").disabled) return;
    const generation = state.generation;
    node("edit").disabled = true;
    node("error").textContent = "";
    try {
      await state.onOpenDesign(state.result.design_id);
    } catch (error) {
      if (generation !== state.generation) return;
      node("error").textContent = `${error.message || String(error)}. You can retry opening the saved revision.`;
    } finally {
      if (generation === state.generation) node("edit").disabled = false;
    }
  }

  function renderExamples() {
    if (!node("examples")) return;
    state.examples = EXAMPLES.map(([label, prompt]) => {
      const button = element("button", label, "assistant-example");
      button.setAttribute("type", "button");
      button.addEventListener("click", () => {
        if (state.busy) return;
        node("prompt").value = prompt;
        node("prompt").focus();
        node("error").textContent = "";
      });
      return button;
    });
    node("examples").replaceChildren(...state.examples);
  }

  function init({ api, onOpenDesign }) {
    if (!node("form")) return;
    newRequest();
    state.api = api;
    state.onOpenDesign = onOpenDesign;
    state.jobId = null;
    state.retryMode = null;
    clearResult();
    node("stages").replaceChildren();
    node("error").textContent = "";
    node("status").textContent = "Describe your experiment, then press Enter to design a shot.";
    node("prompt").setAttribute("maxlength", MAX_PROMPT_LENGTH);
    node("retry").hidden = true;
    if (!node("model").value) node("model").value = "quality";
    renderExamples();
    setBusy(false);
    if (state.initialized) return;
    state.initialized = true;
    node("form").addEventListener("submit", submit);
    node("prompt").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) return submit(event);
    });
    node("retry").addEventListener("click", retry);
    node("edit").addEventListener("click", openDesign);
  }

  scope.ShotDesignAssistant = { init };
})(globalThis);
