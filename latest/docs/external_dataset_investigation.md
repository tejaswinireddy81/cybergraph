# External held-out dataset: verification status and provenance of the local UNSW file

Status (revision 2): **verification attempted; the external file could NOT be obtained or header-checked by script.**
Nothing was downloaded: no dataset bytes, not even a header. Licensing: treated as **academic-only** (your decision); commercial
use is not assumed anywhere. Storage for the external file when you have it: `latest/data/external/` (gitignored).

## 1. Result for NF-CSE-CIC-IDS2018-v3

**Download status: (C) the real file needs an interactive/authenticated download. Header not obtained.**

| Question | Result |
|---|---|
| Where is it served? | The link on the UQ page, `https://rdm.uq.edu.au/files/4ac221b1-6bd6-42b1-bdf7-03f4fc7efb22`, is a **single-page web app** (same HTML shell, ~1.9 KB, for every path including `/robots.txt`). Its API host is `https://api.rdm.uq.edu.au` (endpoints `/datasets/{id}`, `/datasets/{id}/file-details`, `/datasets/{id}/download`, `/datasets/licenses`). |
| Can it be fetched without logging in? | **No.** Unauthenticated calls to the API return **HTTP 401** `{"code":"unauthorized"}`. The app has a login/session flow and an access-request feature; whether a login alone is enough, or an access request / terms click-through is also needed, is **not verified**. Nothing was bypassed and no login was attempted. |
| File size | **Not verified** (not published on the UQ page or in the paper; API metadata is behind the login). |
| Published checksum | **None found.** |
| DOI / handle / record | Not found. The researchdata.edu.au record pages returned **403** to automated access, so they were not read. |
| Real CSV header | **Not obtained.** |
| Does the project's downloader work? | **No, very likely not.** `scripts/download_dataset.py` points at the same kind of link and would save an HTML page as `NF-UNSW-NB15-v3.csv`. (`scripts/verify_external_dataset.py` now rejects such a file.) |

What you need to do: download the CSV yourself from https://staff.itee.uq.edu.au/marius/NIDS_datasets/ (the NF-CSE-CIC-IDS2018-v3 link;
log in or accept terms if the page asks), save it as `latest/data/external/NF-CSE-CIC-IDS2018-v3.csv`, then run
`python scripts/verify_external_dataset.py data/external/NF-CSE-CIC-IDS2018-v3.csv`. It will report the real header, size, SHA-256, row and
category counts and a compatibility verdict, and it writes the full report to `outputs/`.

## 2. Compatibility (from documentation only; NOT verified against the file)

* Published column list (arXiv:2503.04404v3, Table 1): 53 features = the 43 v2 features + 10 temporal ones. It includes all 47 of our
  model features and `FLOW_START_MILLISECONDS`; extra: `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `MIN_TTL`, `MAX_TTL`, `FLOW_END_MILLISECONDS`.
  This list was extracted from the page by a summarising model, not read verbatim: **likely, but check against the real header**.
* Labels: the paper describes a binary column (0 benign, 1 malicious) plus an attack-type column but does not name them. The **UNSW
  sibling file** (our local one) has `Label` (0/1) and `Attack` (`Benign` or the category): verified from its real header (section 4).
  The same names are *likely* for the CSE-CIC file, **not verified**.
* Exporter: all four v3 sets were made with nProbe from the original pcaps, so exporter semantics are shared among them (and differ from
  our live scapy extractor, see `extractor_investigation_and_proposal.md`).
* Published numbers: 20,115,529 flows; benign 17,514,626; attack 2,600,903 (12.93 %): BruteForce 575,194, Bot 207,703, DoS 302,966,
  DDoS 1,324,350, Infiltration 188,152, Web attacks 2,538. Unverified until the file is checked.
* Other candidates: NF-ToN-IoT-v3 (27.5 M flows, 38.98 % attack, same lab family as UNSW, reported duplicate/label-conflict issues from a
  snippet) and NF-BoT-IoT-v3 (99.7 % attack, unusable for false-positive rate). CSE-CIC-IDS2018-v3 remains the recommendation.

## 3. Licence and terms (academic-only)

* **No dataset licence text was found for v3.** The v3 page has no licence or terms statement (its footer links only the generic UQ terms).
  The v3 paper is CC BY 4.0, which covers the article, not the data.
* An older UQ page for the **v1/v2** datasets (cyber.uq.edu.au/node/824) says: *"The use of the datasets for academic research purpuses is
  granted in perpetuity after citing the above papers. For commerical purposes it should be agreed by the authours."* Applying that to v3
  is an inference, not a stated v3 licence.
* Required citation for v3 (from the UQ page): Luay, Layeghy, Hosseininoorbin, Sarhan, Moustafa, Portmann, *Temporal Analysis of NetFlow
  Datasets for Network Intrusion Detection Systems*, 2025, arXiv:2503.04404. The page also points to per-dataset citations (v1/v2 papers).
* Redistribution and commercial use: **not stated** where checked. The underlying CSE-CIC-IDS2018 data (CIC/AWS) has its own terms, **not checked**.
* Working rule: **academic research only, cite the papers, do not redistribute the files, no commercial use.**

## 4. Provenance of OUR local `NF-UNSW-NB15-v3.csv` (what could be established)

**Corrected figures.** The published NF-UNSW-NB15-v3 has 2,365,424 flows: **2,237,731 benign and 127,693 attack** (paper Tables 2 and 3).
My earlier "127,639 attack / +54" was wrong: the UQ web page has a typo (127,639) and also swaps three category counts (it lists Analysis 2,381,
Backdoor 1,226, Shellcode 4,659); the paper and our file agree with each other.

| | local file | published (paper) |
|---|---|---|
| total flows | 377,279 | 2,365,424 |
| attack flows | **127,693** | **127,693** (identical) |
| attack categories | Exploits 42,748, Fuzzers 33,816, Generic 19,651, Reconnaissance 17,074, DoS 5,980, Backdoor 4,659, Shellcode 2,381, Analysis 1,226, Worms 158 | **identical per category** |
| benign flows | **249,586** | 2,237,731 (local = **11.2 %** of it) |
| attack share | 33.85 % | 5.40 % |

**Evidence collected (read-only)**

| Item | Finding |
|---|---|
| Windows mark-of-the-web (`Zone.Identifier` stream) | `ZoneId=3`, `ReferrerUrl=C:\Users\tejaswini\Downloads\hybrid_nids_complete.zip`: the CSV was extracted from a project archive of that name |
| That archive | **no longer on disk** (checked `Downloads`); its own origin could not be read |
| File timestamps | created (extracted) 2026-11-07 20:18; **modified 2026-08-07 17:55:47**, the time preserved from the archive, i.e. the CSV was produced locally on that date |
| Size / checksum | 91,880,451 bytes; SHA-256 `7b0dde74...5785` (recorded in `datasets/NF-UNSW-NB15-v3.csv.sha256`) |
| Git | listed in `.gitignore` since the initial commit; **never tracked**; no script in the repo builds it |
| Project documents | `CyberGraph_Project_Brief.pdf` (generated 2026-09-06) and `build_temporal_pipeline.py` use **377,279**; `generate_research_report.py` says 377,279 in one place and **2,365,424 records / 49 features** in another (inconsistent) |
| Real header | 55 columns, starting `FLOW_START_MILLISECONDS, FLOW_END_MILLISECONDS, IPV4_SRC_ADDR, L4_SRC_PORT, IPV4_DST_ADDR, L4_DST_PORT, ...`, ending `..., Label, Attack`. `Label` is 0/1, `Attack` is `Benign` or the category; they agree on all 377,279 rows |
| Content | not time-sorted in file order; **200 exact duplicate rows**; 3 capture days (2015-01-22, 01-23, 02-18; 120,434 / 4,440 / 124,712 benign); only **35 distinct benign source IPs** and 29 destination IPs; 90,944 non-finite feature values (the training code replaces them with 0) |
| Sampling method of the benign subset | **cannot be determined** from the content (per-minute benign counts are far burstier than a random thinning of a Poisson process, but the original traffic is bursty too, so this is not conclusive) |
| Published variant? | **none found** with 377,279 rows (no sampled/balanced release in the paper or on the UQ page) |

**Conclusion.** The file is *all* published attack flows (exactly) plus ~11 % of the benign flows. Everything points to a **locally built,
benign-downsampled subset** (modified 2026-08-07, delivered inside `hybrid_nids_complete.zip`), but **where it was built and how the benign
flows were chosen is not established**. Per user instruction, browser history was NOT accessed and will not be accessed; the origin and provenance of the local UNSW dataset are officially documented as **UNVERIFIED**.
Consequences: the class prior is not the published one, precision must be reported at realistic prevalence, and the benign host diversity
is tiny (35 source hosts), which strengthens the shortcut-learning concern.

## 5. What happens next (only after you act)

1. You download the external CSV (section 1) and place it in `latest/data/external/` (gitignored).
2. I run `verify_external_dataset.py`: real header, size, SHA-256, counts, compatibility. If it passes, record the SHA-256 in `datasets/`.
3. The file stays **sealed** for the final evaluation only: never for training, tuning, thresholds or fusion; time-stratified slice, never sampled by label.
4. Evaluation is binary benign-vs-attack plus the dataset's own categories reported separately; every attack family is reported on its own.
5. If the file needs a login/terms click-through you are not comfortable with, or the licence turns out to exclude your use, we pick another set.

## 6. Not verified

File size and real header of NF-CSE-CIC-IDS2018-v3; whether its download needs only a login or also an access request/terms; any
checksum; its DOI/record page and licence fields (403); the label column names in that file; benign protocol mix of any candidate; whether
the sibling datasets share addresses or generator artifacts with UNSW-NB15; where and how the local UNSW subset was built.

Sources: staff.itee.uq.edu.au/marius/NIDS_datasets/ (HTML), cyber.uq.edu.au/node/824 (v1/v2 terms), arxiv.org/abs/2503.04404 and its HTML v3,
rdm.uq.edu.au (web-app shell and its JavaScript, read only for endpoint names), api.rdm.uq.edu.au (HTTP 401); researchdata.edu.au records (403, not read).
