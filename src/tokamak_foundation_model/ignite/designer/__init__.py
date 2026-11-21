"""IGNITE shot designer: tools an AI agent uses to design DIII-D actuator trajectories.

The designer has no model and no machine or physics knowledge of its own. An agent (any
LLM with tool use) does all the reasoning, from its own knowledge and anything it looks
up; the designer gives it data and mechanics:

1. IGNITE's actuator channels (:mod:`.machine`; names and signal paths from the data
   config) and what the database shows DIII-D ran, per era and overall;
2. the measured actuators of any tokenized shot (:mod:`.shots`);
3. trajectory construction in physical units (:mod:`.plan`): switch beams, balance the
   torque, set, scale, ramp, shift or draw waveforms, copy another shot's actuators;
4. feasibility against the database (:mod:`.validate`) and export to an
   IGNITE rollout (:mod:`.simulate`, actuator mode ``file:<plan.npz>``);
5. precedents: measured shots that match conditions on actuators and decoded plasma
   quantities (:mod:`.precedents`, index built by :mod:`.index`).

Entry points: ``python -m tokamak_foundation_model.ignite.designer`` (CLI),
``python -m tokamak_foundation_model.ignite.designer.mcp_server`` (MCP over stdio) and
``analysis/shot_designer.ipynb`` (notebook GUI with IGNITE simulation).
"""
