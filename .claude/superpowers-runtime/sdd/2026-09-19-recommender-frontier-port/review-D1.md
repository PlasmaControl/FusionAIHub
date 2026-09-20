## Task D1 review (round 0, head 6b1e7d9)

Verdict: Needs fixes. Important: (1) unrestored in-place cfg.maskgit_decode_steps mutation (model.cfg is cfg); (2) load_dynamics untested though a tmp_path checkpoint test is feasible; (3) F < k0+n_predict truncates gt → opaque RuntimeError; (4) actuators None → bare TypeError. Minor: alignment precondition undocumented; missing-modality KeyError.
