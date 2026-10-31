# False-Positive Calls of Reimplemented Published Analyses for Covert Command-Following on EEG

Analysis code for the article *False-Positive Calls of Reimplemented Published Analyses for Covert Command-Following
on EEG Recorded Without Study Commands* by Alon Gorenshtein, Yosef Adiniaev, Yiftach Barash, Eyal Klang and Oved Daniel.

## Overview

EEG analyses that detect covert command-following (cognitive motor dissociation) in unresponsive patients were
reimplemented from their publications and public code and run on EEG recorded while no study command was given, so
every positive call is false with respect to commands. The code measures:

- how often each analysis calls patients positive without commands, in clinical EEG (a development set and a
  replication set), in prolonged disorders of consciousness and in coma after cardiac arrest, alone and over
  repeated runs;
- the same analyses with trial order randomized and tested by randomization;
- detection of genuine responses on open recordings with real tasks (healthy volunteers, stroke motor imagery, a
  tactile P300 task) and with sham labels;
- the analytical choices that contribute to excess calls;
- detection at thresholds set on null recordings of each design.

## Analyses

| Analysis | Call rule | Module |
|---|---|---|
| Claassen 2019 | Support vector machine, AUC above 0.5 | `code/bench/pipe_columbia.py`, `code/112_columbia_exact.py` |
| Goldfine 2011 | Spectral test with a jackknife variance | `code/bench/pipe_cornell.py` |
| Cruse 2011 | Motor imagery classifier, 99% binomial interval | `code/bench/pipe_cruse11.py` |
| Cruse 2012 | Motor imagery classifier, permutation tests | `code/bench/pipe_cruse12p.py` |
| Edlow 2017 | Motor command classifier, permutation test | `code/bench/pipe_edlow17.py` |
| Guger 2017, mindBEAGLE-type motor imagery | Accuracy above the 95% binomial limit | `code/bench/pipe_mbmi.py` |
| du Bois 2026 | Peak accuracy over a configuration search, or paired t test | `code/bench/pipe_dubois.py` |
| Guger 2018, mindBEAGLE-type vibrotactile P300 | Accuracy above 23% | `code/bench/pipe_mbvt.py` |
| Gibson 2016 | Event-related potentials, cluster permutation | `code/bench/pipe_gibson16.py` |
| Chennu 2013 | Event-related potentials, three cluster tests | `code/bench/pipe_chennu13.py` |
| Sokoliuk 2021 (covert processing) | Speech tracking, surrogate test | `code/bench/pipe_sokoliuk21.py` |

`code/bench/engine.py` places each analysis's own schedule of commands or stimuli on recordings without commands.
`blind_impl/` holds a second implementation of du Bois 2026 and Guger 2018, written from the publications alone.

## Scripts

Scripts are numbered in the order they run; each script's docstring states its inputs, outputs and command line.

| Result | Scripts |
|---|---|
| False calls without commands: clinical EEG and prolonged disorders of consciousness | `113`, `114`, `115` |
| Replication set | `152`, `153` |
| Coma after cardiac arrest (I-CARE) | `122`, `150_icare_stated_rule.py` |
| Each call rule against its stated or implied rate | `145`, `155` |
| White noise and synthetic EEG | `129`, `147` |
| Repeated runs | `151` |
| Healthy-volunteer detection and label-shuffled sessions | `118`, `144` |
| du Bois 2026 decision pathway and the two implementations | `154`, `161`, `170` |
| Goldfine 2011 with a command-level jackknife | `171`, `175`, `177` |
| Genuine tasks and sham labels (stroke, tactile P300) | `163`, `172`, `173`, `176` |
| Detection at thresholds set within each design | `174` |
| Full-length schedules on longer windows | `178`, `179`, `180` |
| Claassen 2019 on six channels | `181` |
| Figures 1, 2 and S1; Figures 3 and 4 | `fig_null_benchmark.py`; `fig_validation.py` |

The other scripts produce the supplementary analyses.

## Repository layout

```
code/                  analysis scripts, shared modules and tests
code/bench/            the reimplemented analyses and the engine that runs them
code/bench/tests/      tests of the analyses and the engine
code/vendor_spindle/   spindle detectors read by 123_spindles.py
blind_impl/            the second implementations of du Bois 2026 and Guger 2018
docs/                  classification of every analysis choice against the publications (stated-rule-*.json),
                       published numbers of assessments (benchmark-k-table.json), simulated effect sizes
                       (benchmark-effect-grid.json) and the case map of the open prolonged-DoC recordings
                       (pdoc-case-map.csv)
output/tables/         the call rules applied unchanged to the replication set (replication_frozen.json) and the
                       settings of the command-level Goldfine 2011 test (cornell_command_frozen.json)
```

## Data

No recordings are included. Each dataset is obtained from its host under its own licence.

| Dataset | Use | Access |
|---|---|---|
| SpikeNet 2.0, Brain Data Science Platform | Clinical EEG without commands | Credentialed access, data use agreement |
| MORGOTH EEG foundation model labels, Brain Data Science Platform | Background of each window | Credentialed access |
| Overnight polysomnography in disorders of consciousness, Mendeley Data | Prolonged disorders of consciousness | CC BY 4.0 |
| I-CARE, version 2.1 | Coma after cardiac arrest | Obtained from PhysioNet (CC BY-NC-SA 4.0); now distributed by the Brain Data Science Platform |
| OpenBMI, GigaDB | Healthy motor imagery | CC0 1.0 |
| EEG Motor Movement/Imagery Database, PhysioNet | Healthy motor imagery | ODC-By 1.0 |
| Four-class auditory BCI (Kojima), Harvard Dataverse | Healthy auditory counting | CC0 1.0 |
| ERP CORE; OpenNeuro ds003061 | Passive oddball | CC BY 4.0; CC0 |
| Acute-stroke hand motor imagery (Liu 2024), figshare | Instructed stroke patients | CC BY 4.0 |
| Sixth-finger and affected-hand motor imagery after stroke (Wang 2026), figshare | Stroke motor imagery with sham labels | CC BY 4.0 |
| Braille tactile P300 (Grigoryan 2020), G-Node | Tactile P300 with sham labels | CC0 1.0 |

`code/bdsp_morgoth.py` also needs the MORGOTH model's code and weights, which are distributed with that model.
Scripts that read the clinical EEG stream it from the host named in `code/bdsp_stream.py`.

## Requirements

Python 3.9 for the main analyses and Python 3.12.0 for scripts 170 to 181 and `code/fig_validation.py`:

```
pip install -r requirements.txt          # Python 3.9
pip install -r requirements-py312.txt    # Python 3.12
```

`145_stated_rule.py`, `147_white_noise_eligible.py` and `150_icare_stated_rule.py` check the classification files in
`docs/` by their sha256:

```
--motor-sha ada8f17b8bb4ea8efd5c3bc5b1142912316e01fde57c127d8eea01e7d9276709
--erp-sha d8803e7ca4c1e78a22c183da5612de39dce3fbc436313ba067195f7f1407ee26
```

## Tests

```
cd code
python -m pytest
```

The tests run on synthetic data. `code/test_columbia_exact.py` checks the Claassen 2019 reimplementation against the
authors' public code ([kwdoyle/CMD-Detection](https://github.com/kwdoyle/CMD-Detection)) and runs as a script:
`python test_columbia_exact.py`.

## Licence

MIT (see `LICENSE`). The datasets keep their own licences.
