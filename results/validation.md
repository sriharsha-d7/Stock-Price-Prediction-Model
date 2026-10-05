# Harness validation on synthetic series (n = 20000 days, 15% test)

## random_walk  (returns are iid: NO model can beat naive)

|                           |   price_MAPE_% |   within_2.5%_of_actual_% |   directional_accuracy_% |   return_IC_spearman | skill vs naive [95% CI]    |
|:--------------------------|---------------:|--------------------------:|-------------------------:|---------------------:|:---------------------------|
| naive (price unchanged)   |           1.42 |                     84.41 |                   nan    |               nan    | n/a                        |
| drift (mean train return) |           1.42 |                     84.41 |                    49.71 |               nan    | -0.0002 [-0.0015, +0.0010] |
| ridge                     |           1.42 |                     84.41 |                    50.12 |                -0.01 | -0.0005 [-0.0020, +0.0011] |

## ar1  (returns follow AR(1), phi = 0.15: a lag model has small, real skill)

|                           |   price_MAPE_% |   within_2.5%_of_actual_% |   directional_accuracy_% |   return_IC_spearman | skill vs naive [95% CI]    |
|:--------------------------|---------------:|--------------------------:|-------------------------:|---------------------:|:---------------------------|
| naive (price unchanged)   |           1.43 |                     83.56 |                   nan    |               nan    | n/a                        |
| drift (mean train return) |           1.43 |                     83.43 |                    50.22 |               nan    | -0.0003 [-0.0020, +0.0013] |
| ridge                     |           1.42 |                     84.30 |                    54.29 |                 0.13 | +0.0192 [+0.0083, +0.0298] |
