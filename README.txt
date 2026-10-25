Code for: False-Positive Calls of Reimplemented Published Analyses for Covert Command-Following on EEG
Recorded Without Study Commands.

code/                  analysis scripts, numbered in the order they run, their modules and their tests (test_*.py)
code/bench/            the reimplemented analyses and the engine that runs them on recordings without study commands
code/bench/tests/      tests of the reimplemented analyses, the engine, the runner and the merge
code/vendor_spindle/   the spindle detectors read by 123_spindles.py, byte-identical (123 checks their sha256)
blind_impl/            the independent implementations of two analyses (Supplementary Data S1, Supplementary Methods 16)
docs/                  stated-rule-motor.json and stated-rule-erp.json: the classification of every analysis
                       choice against the publications (Supplementary Data S1, Supplementary Methods 3); stated-rule-*-first.json: the
                       first classification, whose classes the call-rule test reads; benchmark-k-table.json and
                       benchmark-effect-grid.json: the published numbers of assessments and the simulated effect
                       sizes; pdoc-case-map.csv: case, recording and status of the 40 recordings of the open
                       Mendeley prolonged-disorders-of-consciousness release (CC BY 4.0)
output/tables/replication_frozen.json   the readings fixed before the replication patients were drawn

How this copy differs from the working copy the results came from
- Comments are removed, and some words in names, strings and docstrings are changed, including five script file
  names (117, 131, 132, 162 and 165) and bench/locked_spec.py; the code is otherwise the same.
- Checks tied to the working copy are not applied: the frozen-code hash, commit-state checks and checks against
  a private notes file. Their functions say so in their docstrings. The sha256 check of the classification files
  stays: 145, 147 and 150_icare_stated_rule.py need --motor-sha ada8f17b8bb4ea8efd5c3bc5b1142912316e01fde57c127d8eea01e7d9276709
  --erp-sha d8803e7ca4c1e78a22c183da5612de39dce3fbc436313ba067195f7f1407ee26 (the sha256 of the two docs/ files in this copy).
- output/tables/replication_frozen.json carries the same word changes as the code, so its checksum differs from
  the one recorded for the working copy.
- Tests that read the study's outputs or recordings, or that exercise the checks above, are not included; the
  others run on synthetic data (python3 -m pytest from code/; two test files need h5py).
- The BigP3BCI loader (code/bigp3_als_io.py) reads the dataset folder named by the BIGP3BCI_DATA environment
  variable. Scripts that read the clinical EEG stream it over ssh from the host named in their code.

Data are not included. Supplementary Data S1, Table S1, lists each dataset, its licence and how to obtain it; the clinical
EEG (Brain Data Science Platform) requires credentialed access. Each script's docstring states what it does. The
Python packages and the versions used are in requirements.txt. code/bdsp_morgoth.py also needs the Morgoth EEG
model's own code and weights, which are not included.
