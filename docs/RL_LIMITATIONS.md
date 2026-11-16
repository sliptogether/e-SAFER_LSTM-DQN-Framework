# Interpretation of the DQN demonstration

The primary environment is `SurrogateExerciseEnv` in `src/sweat_rl/rl/env.py`.
It replays a fixed record of sensor-derived states. Its configurable action
effects are illustrative offsets, not physiological response estimates.

## Observations and reward

With the main configuration, the Q-network observes 30 standardized values:
six values for each of `Na`, `K`, `Ca`, `TEMP`, and `FR`. The environment retains
`HR_current` as an additional internal reward variable. Excluding HR from the
Q-network does not make the entire workflow independent of HR: HR enters the
reward, and the LSTM search can select historical HR as context.

For action `a`, the environment adds the configured offsets to current HR,
temperature, and FR in standardized units. It then calculates:

```text
reward = HR_range_reward
         - future_risk_weight * prediction_risk
         - action_change_penalty * abs(action - previous_action)

prediction_risk = [sum(abs(z_endpoint_change))
                   + uncertainty_weight * sum(abs(z_endpoint_SD))]
                  / number_of_endpoint_change_features
```

`HR_range_reward` is +1 inside the record-derived HR quantile range and -1
outside it. Those quantiles are demonstration settings, not medical thresholds.
The standard environment does not change the forecast endpoint or uncertainty
features when applying an action. Its prediction-risk term is therefore
independent of the action at a given recorded state. Action-dependent reward
comes from the HR offset and action-switching penalty.

After this calculation the environment advances to the **next unchanged recorded
state**. Action effects do not propagate into later sensor trajectories. The
previous action affects the switching penalty but is not part of the current
Q-network observation. This is a simplified replay environment, not a validated
physiological transition model or complete state representation.

## Evaluation boundary

- Day 4 is held out from LSTM fitting and model selection.
- The DQN is trained and reported on the same Day 4 state record. State means,
  standard deviations, and HR quantiles also use that entire record. There is no
  independent held-out policy evaluation or online-only normalization here.
- A rising training reward shows fit to this reward specification. It does not
  establish improved electrolyte balance, exercise safety, personalization,
  causal benefit, or clinical effectiveness.
- `Policy_score` is the largest softmax-transformed Q value. It is not a calibrated
  probability of a correct or safe recommendation.
- Ensemble SD describes variation among seed-specific forecasts; it is not a
  calibrated confidence interval.
- With a five-minute horizon, origins 51–55 lack a fully observed subsequent
  five-minute outcome. The decision CSV identifies these rows explicitly.
- Synthetic data and the smoke configuration validate software operation only.
  Their metrics must not be presented as experimental performance.

## Optional protocol-informed surrogate

`scripts/11_protocol_informed_ablation.py` uses a separate environment whose
response contrasts are estimated from a fixed exercise/rest schedule and the
same Day 4 data. It changes additional projected state features and is an
exploratory alternative. It is not run by the default pipeline and does not
provide independent validation of the ordinary fixed-record environment.
Keep its results separate from the primary DQN demonstration.

## Appropriate description

> LSTM forecasts are converted into sensor-derived state features for a DQN
> demonstration of exercise-intensity recommendations in a fixed-record
> surrogate environment.

Claims about actual intervention effects require action-response data and
independent prospective validation beyond this release.
