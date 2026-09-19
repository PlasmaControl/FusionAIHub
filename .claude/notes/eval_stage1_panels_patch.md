# Stage 1 eval — 4-panel plotting wire-up

The big plotting helpers (`HexbinAccumulator`, `PercentileSampleCache`,
`collect_demo_shot_trajectory`, `_best_improvement_channel`,
`plot_ts_4panel`) have already landed in `eval_e2e_stage1.py`. Three remaining
edits, all in `main()` (lines ~1100–1240) and `parse_args` (lines ~595–620).

Apply by hand or `git apply` the diff at the bottom.

## Edit 1 — parse_args: add two CLI flags

In `parse_args()` (currently around line 605–615), **add two new
arguments** just before `return p.parse_args()`:

```python
    p.add_argument(
        "--hexbin_cap", type=int, default=50_000,
        help="Max (pred, target) pairs per modality reservoir-sampled "
             "for the Panel C scatter.",
    )
    p.add_argument(
        "--pct_cache_batches", type=int, default=8,
        help="Number of leading batches whose tensors are cached on CPU "
             "for Panel D best/median/worst-MAE percentile selection.",
    )
```

## Edit 2 — main: replace plot_cache with the new accumulators

Find the block at the start of the eval loop (starts with
`# ── Eval loop ──`, currently line 1101). Replace this:

```python
    # ── Eval loop ────────────────────────────────────────────────────
    accum = GlobalAccumulator(diag_names)
    per_chan = PerChannelAccumulator(diag_names)
    plot_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    rng = random.Random(args.seed)
    n_processed = 0
    for i, batch in enumerate(loader):
        if args.max_batches is not None and i >= args.max_batches:
            break
        predictions, diag_inputs, targets, masks = forward_one_batch(
            model, batch, device
        )
        for cfg in model.diagnostics:
            n = cfg.name
            copy_pred, copy_target, copy_mask = copy_baseline_for_modality(
                cfg, batch, device
            )
            # ctx for direction/magnitude is the diag input, in the same
            # space as predictions and targets (video already standardised).
            ctx = diag_inputs[n]
            accum.update_modality(
                n,
                pred=predictions[n],
                target=targets[n],
                ctx=ctx,
                mask=masks[n],
                copy_pred=copy_pred,
                min_disp_norm=args.min_disp_norm,
            )
            per_chan.update_modality(
                n,
                pred=predictions[n],
                copy_pred=copy_pred,
                target=targets[n],
                mask=masks[n],
            )
        accum.step()
        n_processed += 1

        # Cache the first batch's tensors for plotting (CPU).
        if i == 0:
            for cfg in model.diagnostics:
                n = cfg.name
                plot_cache[n] = {
                    "pred": predictions[n].detach().cpu(),
                    "target": targets[n].detach().cpu(),
                    "ctx": diag_inputs[n].detach().cpu(),
                    "kind": cfg.kind,
                }

        if (i + 1) % 10 == 0:
            logger.info(f"  batch {i + 1} processed")
```

with this:

```python
    # ── Eval loop ────────────────────────────────────────────────────
    accum = GlobalAccumulator(diag_names)
    per_chan = PerChannelAccumulator(diag_names)
    hexbin = HexbinAccumulator(diag_names, cap=args.hexbin_cap)
    pct_cache = PercentileSampleCache(
        diag_names, n_batches=args.pct_cache_batches
    )
    # Video modalities still use the old single-batch image plot path.
    video_first_batch_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    rng = random.Random(args.seed)
    n_processed = 0
    for i, batch in enumerate(loader):
        if args.max_batches is not None and i >= args.max_batches:
            break
        predictions, diag_inputs, targets, masks = forward_one_batch(
            model, batch, device
        )
        for cfg in model.diagnostics:
            n = cfg.name
            copy_pred, copy_target, copy_mask = copy_baseline_for_modality(
                cfg, batch, device
            )
            ctx = diag_inputs[n]
            accum.update_modality(
                n,
                pred=predictions[n],
                target=targets[n],
                ctx=ctx,
                mask=masks[n],
                copy_pred=copy_pred,
                min_disp_norm=args.min_disp_norm,
            )
            per_chan.update_modality(
                n,
                pred=predictions[n],
                copy_pred=copy_pred,
                target=targets[n],
                mask=masks[n],
            )
            if cfg.kind != "video":
                hexbin.update(n, predictions[n], targets[n], masks[n])
                pct_cache.maybe_update(
                    i, n, predictions[n], targets[n], ctx, masks[n]
                )
        accum.step()
        n_processed += 1

        if i == 0:
            for cfg in model.diagnostics:
                if cfg.kind == "video":
                    video_first_batch_cache[cfg.name] = {
                        "pred": predictions[cfg.name].detach().cpu(),
                        "target": targets[cfg.name].detach().cpu(),
                        "ctx": diag_inputs[cfg.name].detach().cpu(),
                    }

        if (i + 1) % 10 == 0:
            logger.info(f"  batch {i + 1} processed")
```

## Edit 3 — main: collect demo shot, replace final plot loop

Find the final plotting block (starts with `# ── Plots ──`, currently
around line 1215). Replace this:

```python
    # ── Plots ────────────────────────────────────────────────────────
    for cfg in diagnostics:
        cache = plot_cache.get(cfg.name)
        if cache is None:
            continue
        out_path = plots_dir / f"{cfg.name}.png"
        try:
            if cache["kind"] == "video":
                plot_video_modality(
                    cfg.name,
                    pred=cache["pred"],
                    target=cache["target"],
                    ctx=cache["ctx"],
                    out_path=out_path,
                )
            else:
                plot_ts_modality(
                    cfg.name,
                    cfg=cfg,
                    pred=cache["pred"],
                    target=cache["target"],
                    ctx=cache["ctx"],
                    n_samples=args.n_plot_samples,
                    out_path=out_path,
                    rng=rng,
                )
        except Exception as exc:
            logger.warning(f"Plot for {cfg.name} failed: {exc}")
```

with this:

```python
    # ── Demo-shot trajectory pass (Panel A) ─────────────────────────
    demo_shot: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    if val_files:
        logger.info(f"Demo-shot trajectory: {val_files[0].name}")
        demo_shot = collect_demo_shot_trajectory(
            model=model,
            file_path=val_files[0],
            chunk_duration_s=args.chunk_duration_s,
            warmup_s=args.warmup_s,
            stats=stats,
            diag_names=diag_names,
            act_names=act_names,
            device=device,
            max_chunks=args.demo_shot_max_chunks
                if hasattr(args, "demo_shot_max_chunks") else 200,
        )

    # ── Plots ────────────────────────────────────────────────────────
    for cfg in diagnostics:
        out_path = plots_dir / f"{cfg.name}.png"
        try:
            if cfg.kind == "video":
                vcache = video_first_batch_cache.get(cfg.name)
                if vcache is None:
                    continue
                plot_video_modality(
                    cfg.name,
                    pred=vcache["pred"],
                    target=vcache["target"],
                    ctx=vcache["ctx"],
                    out_path=out_path,
                )
            else:
                rows = per_channel_results.get(cfg.name, [])
                hex_xy = hexbin.get(cfg.name)
                cache = pct_cache.gather(cfg.name)
                shot_data = (
                    demo_shot.get(cfg.name) if demo_shot is not None else None
                )
                plot_ts_4panel(
                    name=cfg.name,
                    cfg=cfg,
                    per_channel_rows=rows,
                    hexbin_xy=hex_xy,
                    cache=cache,
                    demo_shot=shot_data,
                    chunk_duration_s=args.chunk_duration_s,
                    out_path=out_path,
                    rng=rng,
                )
        except Exception as exc:
            logger.warning(f"Plot for {cfg.name} failed: {exc}")
```

That's all three edits. After applying:
- `parse_args` exposes `--hexbin_cap` and `--pct_cache_batches`
- The eval loop instantiates and feeds `HexbinAccumulator` and `PercentileSampleCache` (and the smaller `video_first_batch_cache`)
- The final plot loop runs the demo-shot pass once, then calls `plot_ts_4panel` per TS modality and `plot_video_modality` for video
