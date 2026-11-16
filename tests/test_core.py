import numpy as np
import torch
from sweat_rl.data import causal_denoise
from sweat_rl.metrics import regression_metrics, slope, trend_label
from sweat_rl.rolling import aggregate_rolling_predictions
from sweat_rl.rl.dqn import QNetwork, infer_policy
from sweat_rl.rl.ablation import make_feature_sets
from sweat_rl.rl.env import SurrogateExerciseEnv
from sweat_rl.rl.protocol_env import ProtocolInformedSurrogateEnv, protocol_phase
from sweat_rl.training.lstm_pipeline import (
    blend_with_persistence, local_trend_delta, loss_components
)

def test_identity_metrics():
    y=np.array([1.,2.,3.])
    m=regression_metrics(y,y)
    assert abs(m["r2"]-1)<1e-12
    assert m["rmse"]==0

def test_slope():
    assert abs(slope([1,2,3,4])-1)<1e-8

def test_trend():
    assert trend_label(2,1)==1
    assert trend_label(-2,1)==-1
    assert trend_label(.2,1)==0

def test_composite_lstm_loss():
    pt=torch.tensor([[1.,3.]])
    yt=torch.tensor([[0.,1.]])
    ps=torch.tensor([[2.,0.]])
    ys=torch.tensor([[0.,0.]])
    trajectory,summary,total=loss_components(pt,yt,ps,ys,3.0)
    assert abs(float(trajectory)-2.5)<1e-8
    assert abs(float(summary)-2.0)<1e-8
    assert abs(float(total)-8.5)<1e-8

def test_local_trend_skip():
    mat=np.arange(20,dtype=float)[:,None]
    cfg={
        "data":{
            "minute_start":1,"input_length_min":5,
            "forecast_horizon_min":3
        }
    }
    delta=local_trend_delta(mat,5,0,cfg)
    assert np.allclose(delta,[1.,2.,3.])

def test_causal_holt_filter_does_not_use_future_samples():
    base=np.array([1.,2.,4.,3.,6.,5.])
    changed=base.copy()
    changed[4:]=[1000.,-1000.]
    spec={"method":"holt","alpha":0.2,"beta":0.1}
    a=causal_denoise(base,spec)
    b=causal_denoise(changed,spec)
    assert np.allclose(a[:4],b[:4])

def test_causal_denoise_validation():
    with np.testing.assert_raises(ValueError):
        causal_denoise([1.,2.],{"method":"ema","alpha":0.0})
    with np.testing.assert_raises(ValueError):
        causal_denoise([1.,2.],{
            "method":"median_ema","alpha":0.5,"median_window":4
        })

def test_persistence_blend():
    predicted=np.array([2.,4.,6.])
    assert np.allclose(
        blend_with_persistence(predicted,2.0,0.5),[2.,3.,4.]
    )

def test_rolling_forecast_aggregation():
    rows=[
        {"Target":"Na","Origin":10,"Horizon":2,"Predicted_minute":12,
         "Predicted":4.0,"Ensemble_SD":1.0,"Actual":5.0},
        {"Target":"Na","Origin":11,"Horizon":1,"Predicted_minute":12,
         "Predicted":6.0,"Ensemble_SD":1.0,"Actual":5.0}
    ]
    out=aggregate_rolling_predictions(rows,10,{"Na":3.0})
    anchor=out[out.Minute==10].iloc[0]
    aggregate=out[out.Minute==12].iloc[0]
    assert np.isnan(anchor.Predicted)
    assert aggregate.Predicted==5.0
    assert abs(aggregate.Consensus_SD-np.sqrt(2.0))<1e-12
    assert aggregate.Num_forecasts==2

def test_dqn_policy_inference_scores_sum_to_one():
    torch.manual_seed(1)
    q=QNetwork(4,3)
    actions,q_values,scores=infer_policy(q,np.zeros((2,4),dtype=float))
    assert actions.shape==(2,)
    assert q_values.shape==(2,3)
    assert scores.shape==(2,3)
    assert np.allclose(scores.sum(1),1.0)


def test_ablation_observation_subset_keeps_full_state_reward():
    columns = [
        "HR_current", "Na_current", "Na_pred_end_delta5", "Na_endpoint_sd"
    ]
    states = np.array(
        [[80.0, 40.0, 1.0, 0.2], [90.0, 41.0, 2.0, 0.4]], dtype=float
    )
    cfg = {
        "rl_demo": {
            "action_effect": {
                "reduce": {"HR": -0.15},
                "maintain": {"HR": 0.0},
                "increase": {"HR": 0.15},
            },
            "reward": {
                "future_risk_weight": 1.0,
                "uncertainty_weight": 0.25,
                "action_change_penalty": 0.1,
                "hr_quantiles": [0.25, 0.75],
            },
        }
    }
    full = SurrogateExerciseEnv(states, columns, cfg)
    subset = SurrogateExerciseEnv(
        states, columns, cfg, observation_columns=["HR_current", "Na_current"]
    )
    full_state, _ = full.reset(seed=1)
    subset_state, _ = subset.reset(seed=1)
    assert full_state.shape == (4,)
    assert subset_state.shape == (2,)
    _, full_reward, *_ = full.step(1)
    _, subset_reward, *_ = subset.step(1)
    assert abs(full_reward - subset_reward) < 1e-12


def test_confirmed_day4_protocol_boundaries():
    minutes = np.array([0, 14, 15, 19, 20, 39, 40, 44, 45, 54])
    assert np.array_equal(
        protocol_phase(minutes),
        np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1]),
    )


def test_protocol_action_changes_full_state_risk_without_changing_measurements():
    columns = [
        "HR_current", "Na_current", "Na_pred_end_delta5",
        "Na_pred_mean_delta5", "Na_pred_slope5", "Na_endpoint_sd",
    ]
    states = np.array(
        [
            [80.0, 40.0, 1.0, 0.6, 0.1, 0.2],
            [90.0, 41.0, 2.0, 1.2, 0.2, 0.4],
        ],
        dtype=float,
    )
    cfg = {
        "rl_demo": {
            "action_effect": {
                "reduce": {}, "maintain": {}, "increase": {}
            },
            "reward": {
                "future_risk_weight": 1.0,
                "uncertainty_weight": 0.25,
                "action_change_penalty": 0.1,
                "hr_quantiles": [0.25, 0.75],
            },
        },
        "protocol_surrogate": {"response_decay": 0.5},
    }
    response = {column: 0.0 for column in columns}
    response["Na_pred_end_delta5"] = -0.5
    maintain = ProtocolInformedSurrogateEnv(
        states, columns, cfg, minutes=[16, 17], response=response,
        features=["Na"],
    )
    increase = ProtocolInformedSurrogateEnv(
        states, columns, cfg, minutes=[16, 17], response=response,
        features=["Na"],
    )
    maintain.reset(seed=1)
    increase.reset(seed=1)
    maintain.step(1)
    increase.step(2)
    assert not np.allclose(maintain.last_full_state, increase.last_full_state)
    assert np.array_equal(maintain.base, increase.base)


def test_sensor_ablation_removes_complete_sensor_family_and_excludes_hr():
    columns = [
        "HR_current",
        "Na_current", "Na_pred_end_delta5", "Na_endpoint_sd",
        "K_current", "K_pred_end_delta5", "K_endpoint_sd",
    ]
    variants = dict(make_feature_sets(columns))
    assert set(("No Na", "No K")).issubset(variants)
    assert all("HR_current" not in observed for observed in variants.values())
    assert len(variants["Full"]) == len(columns) - 1
    assert not any(column.startswith("Na_") for column in variants["No Na"])
    assert not any(column.startswith("K_") for column in variants["No K"])
    assert any(column.startswith("K_") for column in variants["No Na"])
