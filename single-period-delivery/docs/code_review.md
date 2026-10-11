# Single-period Code Review

## Scope

This review covers:

- `monolithic/`: direct integer-programming model;
- `column_generation/`: master and pricing models;
- `heuristic/`: two-stage heuristic and generated result files.

The monolithic and column-generation models were reviewed statically. They were
not solved because the local machine does not have a valid Gurobi license.

## Heuristic Result Validation

The heuristic output for `dataset_9` was validated independently.

Checks passed:

- every dealer-SKU demand is satisfied exactly;
- no plant ships more than its available inventory;
- vehicle capacity is respected;
- the result and detail files contain the same delivery records;
- all shipment quantities are positive integers.

The corrected heuristic objective now contains:

```text
fixed vehicle cost
+ alpha * vehicle cost for every underloaded used vehicle
```

The minimum-load penalty was previously omitted. It is now read from
`model_config.json`, included in `calculate_cost()`, and written to
`opt_summary.csv` as:

- `fixed_cost`;
- `minimum_load_penalty`;
- `total_cost`;
- `below_min_load`.

## Monolithic Model Review

The monolithic model is consistent with the thesis formulation:

- objective: fixed vehicle cost plus minimum-load penalties;
- vehicle-use definition;
- minimum-load satisfaction variable;
- relationship among vehicle use, minimum-load satisfaction, and penalty;
- vehicle-capacity constraint;
- dealer-demand coverage;
- plant-inventory supply limit;
- integer shipment quantities.

No correctness issue was identified in the static review.

## Column-generation Review

The master and pricing structures are present and internally coherent at the
programming level, but one important modeling risk remains:

- the master pattern columns use vehicle fixed cost as their objective
  coefficient;
- the pricing model has a `p_f` variable for minimum-load violation;
- the pattern column does not carry an equivalent `alpha * vehicle_cost`
  penalty coefficient;
- the pricing objective subtracts `p_f` directly rather than the thesis's
  minimum-load penalty.

Therefore, the current column-generation implementation may not be exactly
equivalent to the monolithic objective when a used vehicle is below its minimum
load.

This is a model-equivalence risk, not only an implementation detail. It should
be resolved and tested against the monolithic model on small instances once a
valid Gurobi environment is available.

## Current Conclusion

- Heuristic results are internally feasible and the objective is corrected.
- The monolithic implementation matches the reviewed thesis formulation.
- The column-generation model still requires an objective-equivalence fix and
  numerical validation.
