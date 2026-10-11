# Provisional procurement decision-support demonstration

From the repository root, with the Step 4/5 Python dependencies available:

```sh
python -m src.procurement_support
python -m src.validate_procurement_outputs
```

No fitting or calibration changes occur. Default costs are hypothetical normalized
units per kWh. The exact quantile rule is `under / (under + over)`.
Read `PROCUREMENT_REPORT.md` before interpreting schedules as decisions.

To later supply different cost coefficients, pass a JSON list:

```json
[
  {"Scenario": "A", "Description": "Supplied cost coefficients", "C_under": 2, "C_over": 1}
]
```

```sh
python -m src.procurement_support --scenarios path/to/scenarios.json --cost-units 'supplied units per kWh'
```

All coefficients must be finite, nonnegative and have a positive sum. Scenario
identifiers should be simple letters such as A/B/C/D. This prototype uses constant
marginal coefficients; time-varying tariffs, purchase-price terms, bid constraints,
liquidity and executed orders require a later implementation.

Daily procurement uses the separately frozen daily distribution. Summing interval
quantiles gives a different decision and is reported separately. Do not treat the
two quantities as a reconciled executable schedule. TEST evaluation is restricted
to matched measured demand, since actual full-portfolio ground truth is incomplete.
