# ELM trains: what other fields do about dense, small events

Status: **survey, 2026-10-09; nothing implemented from it yet.** It asks how other fields
detect individual events in dense trains, and which of their methods could fix elm-inst's
weak spot (`elm_inst.md`). On the challenge key, elm-inst finds 22 of the 194 ELMs that
have another within 10 ms.

Five literature searches (cardiology, seismology, spike and pulse deconvolution, finance
and audio with weak labels, fusion) were run by subagents. They opened each source listed
here unless it is marked unverified. Numbers are the papers' own and were not re-derived.

## First, what our trains look like (development data only, no key)

Measured on fs03, out of fold (`round4/elm/inst/inst1`):

- **An isolated ELM is short.** At Smith's 2,316 onsets, the median D-alpha pulse rises
  within one 0.1 ms cell. It peaks 0.2 ms after his onset and decays to 1/e in 1.0 to 1.2
  ms (FS02-FS04). By 10 ms it is at 2 % of its peak.
- **Crowd ELMs are spaced well apart.** In the reviewed crowd spans, the clock's onsets are
  a median 6.1 ms apart (quartiles 4.4 and 8.8 ms). Only 2.8 % of the intervals are under
  3 ms. So the next ELM usually starts after the last has decayed: overlap of tails is
  rare.
- **Crowd ELMs are small, and the missed ones are smaller.** Rise in log10 D-alpha over
  the first 1 ms against the 1 ms before (median, with quartiles):

  | ELMs | count | median rise (decades) | quartiles |
  |---|---|---|---|
  | Smith's isolated ELMs (the onset labels) | 2,316 | 0.94 | 0.79-1.09 |
  | clock onsets in crowds that elm-inst finds | 10,212 | 0.34 | 0.21-0.57 |
  | clock onsets in crowds that elm-inst misses | 13,849 | 0.11 | 0.04-0.23 |

  A missed crowd ELM raises D-alpha by about 30 %, against about 8 times for a labelled
  one. The labels hold no ELM that small.

So the train problem is mainly an **amplitude and label problem, not an overlap one.**
The model learned what an ELM is from large isolated ELMs and was never shown a small one
with a label. Methods for resolving overlapping pulses (deconvolution, peeling) matter
less than the surveys assumed. Methods for teaching a detector small events, and for
lifting weak events between confident ones, matter more.

## What other fields do

| field | method | mechanism | evidence | for ELM trains |
|---|---|---|---|---|
| seismology | **Superimposed events** (Zhu, Mousavi & Beroza, "Seismic signal augmentation..."; [OSTI 1774650](https://www.osti.gov/pages/servlets/purl/1774650)) | sum isolated labelled events at random offsets and amplitude ratios; labels come free | a picker trained on single events "detected only the first large event"; with superposition it picked the second, smaller one (one example, no recall curve). Built from 500 isolated waveforms | our failure, described almost exactly |
| seismology | EQTransformer (Mousavi et al. 2020, [Nat. Commun.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7415159)) | detection box + pick heads; adds a second event scaled 1/k, but only after the first event's coda | augmentations called "essential", no ablation | same two-head layout as elm-inst |
| seismology | Noisy-student self-training (PhaseNet-DAS, Zhu et al. 2023, [Nat. Commun.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10713581/); Xi et al. 2024, [GJI](https://academic.oup.com/gji/article/238/2/1073/7689220)) | the model's picks become labels only if association across stations confirms them; retrain; 2 rounds | 2-5x more events in round 1, +25-50 % in round 2; Xi et al. stop at round 2 because residuals drift in round 3 | clock and model picks kept where filterscopes and density chords agree |
| seismology | Template matching (Ross et al. 2019, [Science](https://igppweb.ucsd.edu/~shearer/mahi/PDF/2019/Ross_Science_2019.pdf)) | cross-correlate catalog events with continuous data | catalog grows 10x; completeness from M1.7 to M0.3 | small ELMs share the shape of large ones? (to check) |
| cardiology | Pan-Tompkins search-back ([TBME 1985](https://people.ece.cornell.edu/land/courses/ece5030/labs/s2014/Pan_Tompkins.pdf)) | if no beat within 166 % of the mean interval, re-search that gap at half the threshold; a peak's slope decides near-misses | 99.3 % of beats on MIT-BIH; fails in bigeminy (alternating sizes) | lift weak peaks in long gaps of a train |
| cardiology | Synthetic or recombined ECG (Kaisti et al., [arXiv 2111.06175](https://arxiv.org/pdf/2111.06175); Jimenez-Perez et al. 2024, [Front. Cardiovasc. Med.](https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2024.1341786/pdf)) | randomise intervals, shape and noise beyond the real range; compose traces from real segments; a count loss | synthetic-only F1 0.968-0.972 against 0.958-0.979 real-trained | the same superposition idea, with shape randomisation |
| cardiology | U-Net R-peak detector, rare-class augmentation (Zahid et al., [TBME 2021](https://arxiv.org/pdf/2101.01666)) | oversample and noise-augment the rare beats | augmentation cut misses by 40 %, raised false detections 9 % | oversample crowd windows |
| audio | DCASE weak + synthetic + unlabelled (Turpault et al. 2019, [DCASE](https://archive.nyu.edu/jspui/bitstream/2451/60771/1/DCASE2019Workshop_Turpault_44.pdf)) | 12,139 synthetic soundscapes mixed from 314 isolated events; clip-level head; mean-teacher on unlabelled data | the standard recipe; drops on unseen recordings | occupancy span = weak label; synthetic trains = strong labels |
| audio | Count-only localisation, LoCo (Schroeter et al., [ICML 2019](https://arxiv.org/pdf/1905.07293)) | loss = -log P(count = observed) with a Poisson-binomial count; localisation emerges without positions | drums F1 94.5 against 93.6-97.5 supervised | a count per crowd span, if reviewers give one |
| audio | DP / HMM beat decoding (Ellis 2007, [JNMR](https://hajim.rochester.edu/ece/sites/zduan/teaching/ece472/reading/Ellis_2007.pdf); Krebs et al., [ISMIR 2015](https://www.ismir2015.uma.es/articles/239_Paper.pdf)) | choose peaks maximising heatmap evidence + an interval prior | peak picking alone "typically insufficient" for beats | decode trains with an interval prior and a free "isolated" state |
| finance | Med9 robust local scale (Khashanah et al. 2025, [JRSS-C](https://cronfa.swan.ac.uk/Record/cronfa69118/Download/69118__34127__c3039a75eeb84180b7ed66d225dbe213.pdf)) | jump test scaled by the median of 9 centred absolute returns, not a trailing variance | Lee-Mykland "failed to detect obvious jumps" in clusters because neighbours inflate the scale | a better clock: local, cluster-tolerant threshold on d(log D-alpha)/dt |
| finance | Hawkes processes | fitted to jumps after detection | no paper found using them to decode detections | low value here |
| pulse pile-up | PADME (Dimitrova, [arXiv 2210.00811](https://arxiv.org/pdf/2210.00811)); OLPF (Wulf et al. 2020, [JAP](https://ntrs.nasa.gov/api/citations/20210012908/downloads/Eckart_Optimal%20filtering_paper.pdf)) | train on synthetic sums of pulses; or jointly fit N shifted templates, lowering the threshold each pass | pulses resolved down to 0.03-0.17 of the decay time; missed pulses are mostly small ones | overlap is rare in our trains; the amplitude finding is the useful part |
| neuroscience | OASIS, CASCADE, Kilosort/YASS ([OASIS](https://pmc.ncbi.nlm.nih.gov/articles/PMC5370160), [CASCADE](https://pmc.ncbi.nlm.nih.gov/articles/PMC7611618), [Kilosort4](https://pmc.ncbi.nlm.nih.gov/articles/PMC11093732)) | non-negative deconvolution; noise-matched augmentation; learn templates from clean events, then recover collided ones by matching pursuit | CASCADE: a model trained at near-zero noise fails on noisier data | augment amplitude *and* noise; learn from clean, recover the rest |
| fusion | ELM-O (O'Shea et al. 2023, [arXiv 2212.08745](https://arxiv.org/html/2212.08745)); BES-ELMnet (Joung et al. 2024, [NF](https://iopscience.iop.org/article/10.1088/1741-4326/ad43fb)) | derivative thresholds across 3 diagnostic families; BES onset prediction | precision 0.995, recall 0.976 on 972 ELMs, but a global percentile threshold drops small ELMs beside large ones and train recall is never measured | the same isolated-ELM bias as ours |
| fusion | DBS CNN (Teo et al. 2024, [RSI](https://arxiv.org/html/2406.01464v2)); JET small-ELM detection (Alhage et al., [EPS 2026 preprint](https://indico.ukaea.uk/event/449/contributions/1703/attachments/642/1089/WPTECP26_41824_submitted.pdf)) | another diagnostic; Laplacian-of-Gaussian plus RNN aimed at small and compound ELMs | DBS found D-alpha spikes the labeller missed; JET 86 % sensitivity, 92 % RNN accuracy | a second diagnostic confirms small ELMs |

**What every field agrees on.** In seismology, cardiology, audio, pulse pile-up and
calcium imaging, the answer to "the labels only hold clean, isolated, large events" is the
same: **synthesise the hard regime from the labelled events**, by superposition at random
spacing and amplitude, randomised shape, and matched noise. Then **self-train on real
unlabelled data, filtered by an independent check.** Seismology names our exact failure:
training on single events teaches the net to suppress smaller neighbours. No field
publishes a recall-against-spacing benchmark for dense trains. SeisBench calls overlap a
blind spot, and the fusion ML papers drop trains from their test sets.

## Recommended plan for elm-inst v2 (ranked)

1. **A labelled development set of trains (the owner's call; it needs people).** Every
   decoding and threshold choice in a train is unscorable without it. The clock is the only
   reference in crowds today, and it finds about 63 % of key ELMs in busy windows.
   - About 300-500 ELMs marked individually in 15-20 reviewed crowd spans would do, chosen
     across regimes and never from the key's shots.
   - With it, the key becomes a true one-look test again.
2. **Synthetic small-ELM trains in training** (the seismology, ECG, DCASE and pile-up
   recipe). In a third of the batches, inject copies of Smith's isolated ELMs into real
   quiet and ELMy background:
   - add them in linear D-alpha and re-log, since our channels are log10 of the cell
     maximum; scale the density drop the same way;
   - draw amplitudes log-uniform down to about 0.05 decades of rise;
   - draw intervals from the clock's crowd intervals (median 6 ms), with some under 3 ms;
   - stretch rise and decay times by a random factor;
   - target the exact injected onsets.

   Report recall against injected amplitude and spacing on held-out synthetic data, plus
   the existing development views (Smith windows, absent alarms).
3. **Two rounds of filtered self-training** (PhaseNet-DAS, Xi et al.). Peaks of the v2
   model in crowd spans become instance targets only where both a filterscope and a
   density chord show the ELM. Retrain, and stop after round 2.
4. **Interval-aware decoding** (Pan-Tompkins search-back, Ellis DP), chosen on the train
   development set of item 1. The heatmap already holds weak peaks for many missed crowd
   ELMs: at threshold 0.2, elm-inst finds 74 % of the clock's close crowd onsets against
   40 % at its fold thresholds. Without dense truth, that trade cannot be scored, which is
   why item 1 comes first.
5. **A Med9-style clock** (finance): a centred, median-based local scale on d(log
   D-alpha)/dt. It is cheap, and it gives better pseudo-onsets for items 3 and 4.

Not recommended now:

- **Deconvolution or peeling** (OASIS, OLPF, Kilosort): overlap is rare in our trains.
- **Hawkes rescoring:** no field uses it to decode.
- **A count loss (LoCo):** it needs per-span counts. It becomes cheap if item 1 records
  counts.

## Caveats

- The surveys were run by subagents. Unverified items are left out above or marked; the
  URLs were opened by them, not re-checked here.
- The amplitude finding uses the clock's onsets as the crowd reference. Some of the
  "missed" ones (10th percentile rise near 0) are likely not ELMs.
- Superposition assumes D-alpha adds linearly. A real ELM soon after another may be
  smaller (pedestal not recovered), which random amplitudes do not model.
