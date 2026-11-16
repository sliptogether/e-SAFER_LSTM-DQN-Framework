import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Use a non-interactive backend for local/headless runs and keep Matplotlib's
# writable cache inside the project instead of the user's profile directory.
_project_root = Path(__file__).resolve().parents[3]
_mpl_cache = _project_root / "outputs" / ".matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_PUB_STYLE = {
    "font.family":"sans-serif",
    "font.sans-serif":["Arial","Helvetica","DejaVu Sans","sans-serif"],
    "font.size":7,
    "axes.spines.right":False,
    "axes.spines.top":False,
    "axes.linewidth":0.8,
    "legend.frameon":False,
    "pdf.fonttype":42,
    "svg.fonttype":"none"
}


def _save_figure_bundle(fig, out_base):
    out_base=Path(out_base)
    fig.savefig(out_base.with_suffix(".png"),dpi=400,bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"),bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"),bbox_inches="tight")
    fig.savefig(
        out_base.with_suffix(".tiff"),dpi=600,bbox_inches="tight",
        pil_kwargs={"compression":"tiff_lzw"}
    )


def prediction_grid(
    csv_path,out_base,targets=("Na","K","Ca","FR"),origin=20
):
    """Four-panel forecast trajectories for the requested sweat variables."""
    df=pd.read_csv(csv_path)
    available=set(df.Target.astype(str))
    missing=[target for target in targets if target not in available]
    if missing:
        raise ValueError(f"Prediction targets not found: {missing}")

    display={
        "Na":r"Na$^+$","K":r"K$^+$","Ca":r"Ca$^{2+}$","FR":"FR"
    }
    colors={"predicted":"#2563A6","measured":"#222222"}
    with matplotlib.rc_context(_PUB_STYLE):
        fig,axes=plt.subplots(2,2,figsize=(7.2,4.8),constrained_layout=False)
        axes=axes.ravel()
        for panel,(ax,target) in enumerate(zip(axes,targets)):
            s=df[(df.Target==target)&(df.Origin==origin)].sort_values("Horizon")
            if s.empty:
                raise ValueError(
                    f"No prediction rows for target={target}, origin={origin}"
                )
            horizon=s.Horizon.to_numpy(float)
            predicted=s.Predicted.to_numpy(float)
            sd=s.Ensemble_SD.to_numpy(float)
            ax.fill_between(
                horizon,predicted-sd,predicted+sd,
                color=colors["predicted"],alpha=.17,linewidth=0,
                label="Ensemble SD"
            )
            ax.plot(
                horizon,predicted,color=colors["predicted"],marker="o",
                markersize=3.1,linewidth=1.5,label="Predicted"
            )
            valid=s.Actual.notna().to_numpy()
            if valid.any():
                ax.plot(
                    horizon[valid],s.Actual.to_numpy(float)[valid],
                    color=colors["measured"],marker="s",markersize=2.8,
                    linewidth=1.25,label="Measured"
                )
            ax.set_title(display.get(target,target),fontweight="bold",pad=4)
            ax.set_xlabel("Forecast horizon (min)")
            ax.set_ylabel("Processed signal value")
            ax.set_xlim(horizon.min()-.25,horizon.max()+.25)
            if len(horizon) <= 5:
                ax.set_xticks(horizon)
            else:
                ax.set_xticks(horizon[::2])
            ax.grid(axis="y",color="#D9D9D9",linewidth=.5,alpha=.7)
            ax.text(
                -.16,1.04,chr(ord("a")+panel),transform=ax.transAxes,
                fontweight="bold",fontsize=8,va="bottom"
            )

        handles,labels=axes[0].get_legend_handles_labels()
        order=[labels.index(x) for x in ["Predicted","Measured","Ensemble SD"]]
        fig.legend(
            [handles[i] for i in order],[labels[i] for i in order],
            loc="upper center",bbox_to_anchor=(.5,.995),ncol=3
        )
        max_horizon=int(df[(df.Target==targets[0])&(df.Origin==origin)].Horizon.max())
        fig.suptitle(
            f"Day 4 ion and flow forecasts from minute {origin} "
            f"(future {max_horizon} min)",fontweight="bold",y=1.035
        )
        fig.tight_layout(rect=(0,0,1,.94),h_pad=1.2,w_pad=1.2)
        _save_figure_bundle(fig,out_base)
        plt.close(fig)


def _rolling_axis(ax,subset,target,display,show_ylabel=True):
    subset=subset.sort_values("Minute")
    minute=subset.Minute.to_numpy(float)
    actual=subset.Actual.to_numpy(float)
    valid=subset.Predicted.notna().to_numpy()
    predicted=subset.Predicted.to_numpy(float)
    sd=subset.Consensus_SD.to_numpy(float)
    ax.plot(
        minute,actual,color="#222222",linewidth=1.35,
        marker="s",markersize=2.3,markevery=5,label="Measured"
    )
    ax.fill_between(
        minute[valid],predicted[valid]-sd[valid],predicted[valid]+sd[valid],
        color="#2563A6",alpha=.17,linewidth=0,label="Rolling uncertainty"
    )
    ax.plot(
        minute[valid],predicted[valid],color="#2563A6",linewidth=1.55,
        marker="o",markersize=2.4,markevery=5,label="Rolling prediction"
    )
    ax.axvline(10,color="#777777",linestyle="--",linewidth=.8)
    ax.set_xlim(10,55)
    ax.set_xticks([10,15,20,25,30,35,40,45,50,55])
    ax.set_title(display.get(target,target),fontweight="bold",pad=4)
    ax.set_xlabel("Time (min)")
    if show_ylabel:
        ax.set_ylabel("Processed signal value")
    ax.grid(axis="y",color="#D9D9D9",linewidth=.5,alpha=.7)


def rolling_prediction_figures(
    csv_path,out_dir,targets=("Na","K","Ca","FR"),horizon=5
):
    """Export full minute 10-55 rolling trajectories as grid and singles."""
    df=pd.read_csv(csv_path)
    out_dir=Path(out_dir)
    display={
        "Na":r"Na$^+$","K":r"K$^+$","Ca":r"Ca$^{2+}$","FR":"FR"
    }
    with matplotlib.rc_context(_PUB_STYLE):
        fig,axes=plt.subplots(2,2,figsize=(7.2,4.8))
        axes=axes.ravel()
        for panel,(ax,target) in enumerate(zip(axes,targets)):
            subset=df[df.Target==target]
            if subset.empty:
                raise ValueError(f"Rolling target not found: {target}")
            _rolling_axis(ax,subset,target,display)
            ax.text(
                -.16,1.04,chr(ord("a")+panel),transform=ax.transAxes,
                fontweight="bold",fontsize=8,va="bottom"
            )
        handles,labels=axes[0].get_legend_handles_labels()
        order=[
            labels.index("Rolling prediction"),labels.index("Measured"),
            labels.index("Rolling uncertainty")
        ]
        fig.legend(
            [handles[i] for i in order],[labels[i] for i in order],
            loc="upper center",bbox_to_anchor=(.5,.995),ncol=3
        )
        fig.suptitle(
            f"Day 4 continuous rolling {horizon}-min forecasts "
            "(minutes 10–55)",fontweight="bold",y=1.035
        )
        fig.tight_layout(rect=(0,0,1,.94),h_pad=1.2,w_pad=1.2)
        _save_figure_bundle(
            fig,out_dir/f"Fig_rolling_{horizon}min_all_ions_10to55"
        )
        plt.close(fig)

        for target in targets:
            subset=df[df.Target==target]
            fig,ax=plt.subplots(figsize=(4.6,3.1))
            _rolling_axis(ax,subset,target,display)
            handles,labels=ax.get_legend_handles_labels()
            order=[
                labels.index("Rolling prediction"),labels.index("Measured"),
                labels.index("Rolling uncertainty")
            ]
            ax.legend(
                [handles[i] for i in order],[labels[i] for i in order],
                loc="best",fontsize=6.5
            )
            fig.tight_layout()
            _save_figure_bundle(
                fig,out_dir/f"Fig_rolling_{horizon}min_{target}_10to55"
            )
            plt.close(fig)

def prediction_plot(csv_path, out_path, target="Ca", origin=20):
    df = pd.read_csv(csv_path)
    s = df[(df.Target==target)&(df.Origin==origin)]
    plt.figure(figsize=(6,4))
    plt.plot(s.Predicted_minute, s.Predicted, marker="o", label="Predicted")
    if s.Actual.notna().any():
        plt.plot(s.Predicted_minute, s.Actual, marker="o", label="Measured")
    x=s.Predicted_minute.to_numpy()
    p=s.Predicted.to_numpy()
    sd=s.Ensemble_SD.to_numpy()
    plt.fill_between(x,p-sd,p+sd,alpha=.2,label="Ensemble SD")
    plt.xlabel("Time (min)")
    plt.ylabel(target)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path,dpi=300)
    plt.close()

def performance_heatmap(json_path, out_path):
    with open(json_path,"r",encoding="utf-8") as f:
        m=json.load(f)
    targets=list(m)
    rows=["R²","MAE","RMSE","Direction acc."]
    a=np.array([
        [m[t]["r2"] for t in targets],
        [m[t]["mae"] for t in targets],
        [m[t]["rmse"] for t in targets],
        [m[t]["direction_accuracy"] for t in targets]
    ],float)
    plt.figure(figsize=(7,4))
    im=plt.imshow(a,aspect="auto")
    plt.colorbar(im)
    plt.xticks(range(len(targets)),targets)
    plt.yticks(range(len(rows)),rows)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            plt.text(j,i,f"{a[i,j]:.2f}",ha="center",va="center")
    plt.tight_layout()
    plt.savefig(out_path,dpi=300)
    plt.close()

def rl_state_plot(csv_path, out_path, origin=30):
    df=pd.read_csv(csv_path)
    r=df[df.Origin_min==origin].iloc[0]
    targets=["Na","K","Ca","TEMP","FR"]
    columns={
        t:next(c for c in df.columns if c.startswith(f"{t}_pred_end_delta"))
        for t in targets
    }
    y=[r[columns[t]] for t in targets]
    plt.figure(figsize=(7,4))
    plt.bar(targets,y)
    plt.axhline(0,linewidth=1)
    plt.ylabel("Predicted endpoint change")
    plt.title(f"Prediction-derived RL state at {origin} min")
    plt.tight_layout()
    plt.savefig(out_path,dpi=300)
    plt.close()

def reward_plot(csv_path,out_path):
    df=pd.read_csv(csv_path)
    plt.figure(figsize=(6,4))
    plt.plot(df.Episode,df.Reward)
    if len(df)>=20:
        plt.plot(df.Episode,df.Reward.rolling(20).mean(),label="20-episode mean")
        plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(out_path,dpi=300)
    plt.close()

def lstm_loss_plot(history_dir, out_base):
    files=sorted(Path(history_dir).glob("*/seed_*_loss.csv"))
    if not files:
        raise FileNotFoundError(f"No LSTM loss histories found in {history_dir}")
    df=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    selection=df[df.phase=="epoch_selection"].copy()
    targets=[t for t in ["Na","K","Ca","TEMP","FR"] if t in set(selection.target)]

    with matplotlib.rc_context(_PUB_STYLE):
        fig,axes=plt.subplots(2,3,figsize=(7.2,4.6),sharex=False)
        axes=axes.ravel()
        for ax,target in zip(axes,targets):
            d=selection[selection.target==target]
            for _,seed_data in d.groupby("seed"):
                ax.plot(
                    seed_data.epoch,seed_data.train_total_loss,
                    color="#4C78A8",alpha=.14,linewidth=.7
                )
                ax.plot(
                    seed_data.epoch,seed_data.validation_total_loss,
                    color="#E28E2C",alpha=.18,linewidth=.7
                )
            train_mean=d.groupby("epoch").train_total_loss.mean()
            val_mean=d.groupby("epoch").validation_total_loss.mean()
            ax.plot(
                train_mean.index,train_mean.values,color="#2F5D8A",
                linewidth=1.8,label="Training"
            )
            ax.plot(
                val_mean.index,val_mean.values,color="#D97706",
                linewidth=1.8,label="Validation"
            )
            selected=float(d.groupby("seed").selected_epoch.first().median())
            ax.axvline(
                selected,color="#555555",linestyle="--",linewidth=.9,
                label="Median selected epoch"
            )
            ax.set_title(target,fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Normalized composite MSE")
            ax.grid(axis="y",color="#D9D9D9",linewidth=.5,alpha=.7)
            if (d.train_total_loss>0).all() and (d.validation_total_loss>0).all():
                ax.set_yscale("log")

        for ax in axes[len(targets):]:
            ax.axis("off")
        handles,labels=axes[0].get_legend_handles_labels()
        axes[-1].legend(handles,labels,loc="center",fontsize=7)
        axes[-1].text(
            .5,.22,"Epoch selection: Day 1–2 training, Day 3 validation\n"
            "Final models: refit on Day 1–3; Day 4 remains independent test",
            ha="center",va="center",fontsize=6.5,transform=axes[-1].transAxes
        )
        fig.suptitle("LSTM epoch-selection loss",fontweight="bold",y=.995)
        fig.tight_layout(rect=(0,0,.99,.97))
        out_base=Path(out_base)
        _save_figure_bundle(fig,out_base)
        plt.close(fig)
