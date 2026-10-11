# Day Ahead Residential Energy Forecasting

Final hackathon project for the E.ON challenge: forecast next-day electricity demand across 410 households and explore uncertainty-aware energy procurement.

**Team:** Sudip Phuyal, Mahsa Azarnia, Manila Bhandari, Rujjal Sada, Vignesh Pechiappan Ayyathurai.

## Explore the results

- [Interactive results dashboard](https://sudipphuyal.github.io/rwth_hackathon/)
- [Final report (PDF)](Final%20Report.pdf) and [editable report](Final%20Report.docx)
- [Final presentation](Presentation/Final_Presentation_Ready_To_Present.pptx) and [presenter notes](Presentation/Final_Presenter_Notes.md)
- [One-page project description](Approach%20A/ONE_PAGE_DESCRIPTION.pdf)
- [Original challenge and dataset description](CHALLENGE.md)

The dashboard includes actual-versus-predicted demand, weather errors, feature comparisons, uncertainty coverage, final forecasts, interactive purchasing scenarios and links to the supporting evidence. It works offline: open `Results_Dashboard/dist/index.html` directly in a browser.

For a local web view, run this from the repository root and open http://127.0.0.1:8765:

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory Results_Dashboard/dist
```

On macOS, you can also double-click `Results_Dashboard/Start_Dashboard.command`.

## Implementation and saved outputs

| Folder | Contents |
| --- | --- |
| `Approach A/src/` | Preprocessing, weather and household forecasting, validation, uncertainty calibration and procurement support |
| `Approach A/tests/` | Tests of data handling, timing, forecasting and procurement calculations |
| `Approach A/outputs/` | Frozen models, selection/calibration locks, reports, figures, metrics, validation/TEST predictions, TEST replay inputs and final February 28 schedules |
| `Approach B/` | Global Ridge regression and fallback implementation, plus the supplied results description |
| `Results_Dashboard/` | Complete static web view, evidence snapshots and repackaging script |
| `Presentation/` | Latest final slides, presenter scripts and source references |

Saved results remain unchanged. Approach A's held-out portfolio WAPE is **14.29%** for ML and **14.47%** for the lag blend. Temperature MAE is **2.50 °C** and measured ML interval P10–P90 coverage is **79.0%**. The February 28 forecast is **13.015 MWh** across 410 households; the separately calibrated daily P80 decision is **14.011 MWh** under the illustrative 4:1 scenario.

The historical procurement improvement of approximately **24%** is a simulated reduction in normalized interval imbalance cost. Approach B's published historical results use different support and timing; a controlled annual A-versus-B comparison is unavailable. Measured February 28 demand and actual market prices were not supplied.

## Review and reproduce

Start with [Approach A's run and review guide](Approach%20A/README.md). Run its commands from the `Approach A` directory, using Python 3.12 and `requirements-reproduction.txt`. Output-only uncertainty and procurement checks use the supplied frozen files. The saved submission audit covers 857 output fingerprints; `PROJECT_MANIFEST.json` inventories the published contribution; run `python3 verify_project.py` from the repository root to check it without installing dependencies.

The fork already includes the organizer dataset at `data/`. To make it available to Approach A on macOS/Linux without duplicating it, run once from the repository root:

```sh
ln -s ../data 'Approach A/data'
```

On systems without symlink support, copy the organizer data into `Approach A/data/`. Approach B uses the repository's `data/15min/` by default; set `METER_DATA_DIR` to use another location. Its whitespace and data-path setup were normalized for normal Python execution; its forecasting logic and saved report are unchanged.

Fresh model reproduction needs a separate copy of the source and original data with newly generated outputs. Existing TEST completion locks protect the frozen experiment. Recomputable training/validation feature caches are excluded; commands that explicitly consume those caches need a fresh materialization. The supplied outputs retain the inputs needed for frozen procurement replay and output verification. No training, retuning or recalibration was performed when preparing this project.

Temporary builds, local environments, duplicate archives and superseded slide decks are excluded. The original dataset and starter files already present in the fork remain available. The complete static dashboard is also published from `main` by the included GitHub Pages workflow.

## Share with the team

Share this repository and the dashboard link above. Team members can clone the final project with:

```sh
git clone --branch main https://github.com/sudipphuyal/rwth_hackathon.git
```
