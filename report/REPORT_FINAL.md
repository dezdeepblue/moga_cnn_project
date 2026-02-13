# A+ Final Report Text (Auto-filled)

## Key outcomes
- Best proxy CIFAR-10 validation accuracy: **0.5600**
- Smallest model (params): **29,738** (acc=0.2114)
- Lowest FLOPs model: **29,721,536** (acc=0.2114)

## Stability (Q6)
- HV (Acc-Params) mean±std: **0.5163 ± nan**
- HV (Acc-FLOPs) mean±std: **0.4870 ± nan**
- IGD mean±std: **0.0000 ± nan**

## Transfer / Generalization (Q4)
Dataset: **CIFAR100**
- Mean target accuracy across selected models: **0.0643**
- Mean retrain std across models: **0.0000**

## Representative Pareto models
| rep_tag    |    acc |   params |    flops | activation   | optimizer   |          lr |
|:-----------|-------:|---------:|---------:|:-------------|:------------|------------:|
| min_params | 0.2114 |    29738 | 29721536 | gelu         | sgd         | 0.000174316 |
| min_flops  | 0.2114 |    29738 | 29721536 | gelu         | sgd         | 0.000174316 |
| max_acc    | 0.56   |   289034 | 79766464 | leaky_relu   | adam        | 0.0071632   |
| knee       | 0.56   |   289034 | 79766464 | leaky_relu   | adam        | 0.0071632   |

