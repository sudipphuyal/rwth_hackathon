# Local results dashboard

Open `dist/index.html` directly in a browser. The dashboard works offline, with all plotting code and verified project data stored locally.

For a localhost view, double-click `Start_Dashboard.command`. It serves only the dashboard's `dist` folder on your own computer at **http://127.0.0.1:8765**. Keep the Terminal window open while viewing. Press Control-C in that window to stop it.

The dashboard covers the two approaches, preprocessing, chronological testing, actual-versus-predicted graphs, validation feature studies, weather errors, uncertainty coverage, final forecasts and interactive cost-based purchasing scenarios. The Evidence section contains source-file links and full forms.

Important interpretation: Approach B lacks a comparable annual test and common-support prediction export. P80 describes an estimated demand percentile. Under the illustrative 4:1 scenario, 14.011 MWh is a separate net-daily P80, while 14.998 MWh adds 96 interval P80 quantities. The 24.09% reduction describes historical simulated interval penalty. Real tariffs and final-day actual demand are unavailable.

The supplied `data-evidence.json` and `data-procurement.json` retain verified source values. Run `build_snapshot.py` after updating these snapshots to repackage the static view from the repository's saved source files. The view does not refit models or execute purchases.

The hosted dashboard is https://sudipphuyal.github.io/rwth_hackathon/. GitHub Pages deploys the complete `dist/` folder from the `main` branch.
