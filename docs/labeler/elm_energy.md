# Diamagnetic ELM energy

The ELM editor measures the observed energy loss on a diamagnetic-loop trace.
EFIT WMHD supplies its energy scale. The three diagnostic rows show calibrated
loop energy in J, D-alpha-matched losses in kJ, and loss as a percentage of
pre-ELM energy. Estimates are provisional until the selected loop, its response
and the measurement windows are validated for the campaign.

## Calculation

1. Fit `a*t + b` to at least two non-overlapping quiet baseline windows and
   subtract it from the native loop trace. Fit only the supplied quiet samples;
   detrending the plasma record would also remove real energy evolution.
2. The interval from the end of current ramp-up to the start of ramp-down is
   a useful calibration window. At native EFIT times in that window, interpolate
   the corrected loop within contiguous finite runs. Use the median signed ratio
   `gain = median(WMHD / corrected_loop)` and compute
   `W_loop = gain * corrected_loop`. Negative loop polarity is supported.
   Require at least three overlapping positive WMHD samples, consistent loop
   polarity, and a median relative calibration residual no greater than 20%.
3. Estimate noise and detect peaks in the loop's pre-minus-post energy change
   **inside the calibration interval**, independently of D-alpha. Quiet
   off-plasma samples cannot set the plasma loss threshold. Both pre/post
   windows must lie entirely in this interval. Match against the existing
   filterscope peak picker only when each loop drop and D-alpha peak have
   one admissible counterpart. Crowded or competing matches stay unsized.
   A matched event with complete, isolated measurement windows gets
   `delta_W = mean(W_before) - mean(W_after)` and
   `fraction = delta_W / mean(W_before)`.

Default pre/post windows are -2 to -0.25 ms and +0.25 to +2 ms relative to the
loop-drop time, with at least two samples in each. Matching tolerance and
minimum loop-peak separation are 3 ms. Drop height and prominence must exceed
the larger of five robust noise scales (1.4826 times the MAD of the step trace)
and 0.2% of the typical positive pre-event energy. These are configurable
starting parameters, not thresholds validated against the current campaign.

Original NaNs and time gaps greater than 1.5 native sample steps split the
record. Neither calibration nor loss windows bridge those gaps. Another
D-alpha peak in the measurement support leaves the individual attribution
unresolved. A match cannot assign a size to a different D-alpha event from the
one inside its measurement support. D-alpha peaks without a qualifying unique
loop drop remain unsized, including times outside the calibration interval.
The loss rows use impulses at matched loop-drop times; zero between impulses
inside the interval is a display baseline. All three energy rows are unavailable
outside that interval. Native-record drift correction, calibration and scoped
detection precede page-view slicing.

## Local signal contract

`elm_energy.load` reads a `diamagnetic_loop` group from the features store,
corpus, or raw cache, in that order. It expects:

- `xdata`: finite, strictly increasing native times in seconds;
- `ydata`: one already integrated diamagnetic-flux or energy-proxy trace,
  shaped `(1, times)`, with its native units and polarity;
- string group attrs `units` and `source` identifying the supplied signal;
- JSON string attrs `baseline_windows_ms` and `calibration_window_ms`;
- optional JSON string attr `elm_energy_settings` overriding fields of
  `elm_energy.Settings`.

For example, a particular shot might carry:

```json
{
  "units": "Wb",
  "source": "verified integrated diamagnetic signal for this shot",
  "baseline_windows_ms": [[-200, -50], [10500, 10700]],
  "calibration_window_ms": [1000, 9000],
  "elm_energy_settings": {"match_tolerance_ms": 3.0}
}
```

The example windows are not defaults; choose them from the shot's quiet and
plasma intervals. A raw pickup-coil voltage must be integrated and compensated
before it satisfies this contract. Fitting a line directly to the loop during
the current flat-top would also subtract physical energy evolution. If quiet
baseline windows are unavailable, an EFIT-referenced drift fit would need a
separate, validated procedure; this estimator does not assume constant energy
through the flat-top.

The canonical feature registry now maps `diamagnetic_loop` to PTDATA `DIAMAG3`
at its native cadence. The resolver accepts integrated-flux units (`MVSE`,
`mV s`, `mWb`, or `Wb`), converts milli-units to Wb, and keeps the signal's
polarity. A resolved trace still needs explicit per-shot baseline/calibration
metadata before the editor can measure it. After the user refreshed the FDP
token, PTDATA `DIAMAG3` and EFIT WMHD were read successfully for shots 192238
and 189061. Both native flux records report mWb and 0.05 ms sample spacing;
the sample clock does not establish the physical diagnostic response. Their
configured FS01 records were read from the corpus.

An exploratory calculation used quiet-current windows [-700, -400] and
[8000, 8500] ms. For 192238, calibration over [1400, 4600] ms gave a signed
gain of -1.682e7 J/Wb and a 2.68% median relative residual. Continuing live
validation exposed a defect in the initial detector: its full-record noise
estimate was dominated by quiet samples. Its threshold was 4.41 kJ, while the
calibration interval requires 16.56 kJ under the same settings. The maximum
positive pre-minus-post step there is 11.87 kJ. The revised estimator therefore
sizes **no events** in this trial. The original 384 matched candidates, including
the previously illustrated 6.80 kJ step, do not establish individual ELM sizes.
This does not imply an absence of ELMs.

Twelve quiet-baseline/calibration-window trials gave gains from -1.697e7 to
-1.624e7 J/Wb. In the initial detector, changing the pre/post window
changed matched candidates in 1.8–4.4 s from 374 (short) to 195 (default) to
2 (wide). With interval-scoped noise, all three trial windows remain unsized.
Time-shift controls also have no qualifying drops and cannot establish timing
significance. FS01 and the loop are noisy, and the gain does not track WMHD
outside the chosen regime. For 189061, the trial interval
[1800, 5300] ms failed the loop-polarity consistency check. No campaign
defaults or production calibration metadata were set. Real-shot source access
is verified; compensation, response and measurement-window validation remain
necessary for physical ELM sizing.

WMHD is read from local stores or, with live fetching enabled, from EFIT01
`\efit01::top.results.aeqdsk:wmhd`. D-alpha uses the same FS01/FS02 selection as
the ELM editor. Without a usable loop, baseline, EFIT reference or D-alpha,
the optional energy rows are omitted and the other diagnostics remain usable.

`Analysis.metadata()` uses method `diamagnetic_loop_v2` and records the
`measurement_window_ms` as well as drift coefficients/windows/residual, signed
gain and calibration residual, settings, source/unit provenance, losses and
unmatched D-alpha peaks. Review HDF5 stores retain it in
`params.panel_metadata`. Each loss records both loop and D-alpha times,
pre/post energies, loss in J and fractional loss. These measurements are
separate from manually reviewed individual/group annotations.

ELM review row version 3 replaces earlier caches, including version 2 rows
that used full-record noise or displayed energy outside the calibration regime.

## Diagnostic response

### Identifying the signal

The supplied `.tmp/magnetic_mapping.csv` lists six DSL channels, all at
upper-divertor locations. The installed `imas_composer` machine descriptions
classify every one under `magnetics.b_field_pol_probe` in all eight MHDIN
versions (000001 through 197555). They are local poloidal-field sensors,
which rules out using them as the whole-vessel toroidal-flux loop. This
classification does not establish what the DSL acronym expands to.

The DIII-D OMAS mapping explicitly assigns PTDATA `DIAMAG3` to
`magnetics.diamagnetic_flux.0`, multiplies its data by 1e-3, and converts
milliseconds to seconds. The installed IMAS mapper uses the same pointname
and conversion
([official DIII-D mapping, `ip_bt_dflux_data`](https://github.com/gafusion/omas/blob/master/omas/machine_mappings/d3d.py#L1753-L1788)).
A General Atomics report also shows an archived `DIAMAG3` waveform in MVSE
([GA–A22661, figure 6](https://fusion.gat.com/pubs-ext/MISCONF97/A22661.pdf)).
This establishes a flux signal to investigate, rather than identifying
which of the redundant physical loops feeds each campaign's archived trace.
The mapping does not document the full analog/software compensation chain
quoted in the diagnostic description; calibration against WMHD does not
replace those corrections.

### Limits on ELM sizing

DIII-D's published capabilities list diamagnetic toroidal flux as a kinetic
energy diagnostic with 500 microsecond time resolution
([capabilities document, magnetic diagnostics](https://d3dfusion.org/wp-content/uploads/diii-d_capabilities_document.pdf)).
The OMFIT/TRANSP tutorial documents the `DIAMAG`/`DFX` labels and polarity changes
with toroidal-field direction
([tutorial, sign conventions](https://transp.pppl.gov/tutorials/DIIIDomfit.pdf)).
Those labels do not establish a live fetch expression or the response of a
particular archived signal. The estimator retains native cadence and does not
deconvolve the conducting-wall response or remove compensation noise; a gain
fit alone cannot recover unresolved fast losses.
