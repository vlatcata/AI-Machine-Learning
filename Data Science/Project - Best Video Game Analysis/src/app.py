import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import gradio as gr

# ---------- CONFIG ----------
DATA_PATH = Path(__file__).parent / "../" / "data" / "best_video_games.csv"

DEFAULT_WEIGHTS = {
    "owners_count": 0.15,
    "retention_score": 0.30,
    "peak_concurrent_users": 0.20,
    "recommendations": 0.15,
    "review_score": 0.10,
    "playtime_score": 0.15,
}

REQUIRED_BASE_COLS = [
    "name",
    "owners_count",
    "retention_score",
    "peak_concurrent_users",
    "recommendations",
    "positive",
    "negative",
    "average_playtime_forever",
    "median_playtime_forever",
]

# ---------- LOAD ONCE ----------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"CSV not found at {DATA_PATH}. Adjust DATA_PATH accordingly.")
_games_raw = pd.read_csv(DATA_PATH)

# ---------- CORE FUNCTION ----------
def compute_topN(
    w_owners=DEFAULT_WEIGHTS["owners_count"],
    w_ret=DEFAULT_WEIGHTS["retention_score"],
    w_peak=DEFAULT_WEIGHTS["peak_concurrent_users"],
    w_recs=DEFAULT_WEIGHTS["recommendations"],
    w_rev=DEFAULT_WEIGHTS["review_score"],
    w_play=DEFAULT_WEIGHTS["playtime_score"],
    topn=10,
):
    games = _games_raw.copy()

    # sanity check columns
    missing = [c for c in REQUIRED_BASE_COLS if c not in games.columns]
    if missing:
        err = pd.DataFrame({"error":[f"Missing columns: {', '.join(missing)}"]})
        return err, px.bar(title="Missing columns")

    # engineered features
    alpha = 1.5
    denom = (games["positive"] + games["negative"] + 1)  # avoid /0
    games["review_score"] = ((games["positive"] * games["negative"]) - alpha) / denom
    games["playtime_score"] = (games["average_playtime_forever"] + games["median_playtime_forever"]) / 2

    cols = [
        "owners_count",
        "retention_score",
        "peak_concurrent_users",
        "recommendations",
        "review_score",
        "playtime_score",
    ]
    # fill NaNs before scaling
    for c in cols:
        if games[c].isna().any():
            games[c] = games[c].fillna(games[c].median())

    scaler = MinMaxScaler()
    games[cols] = scaler.fit_transform(games[cols])

    # weights (normalize to sum=1 for safety)
    w = np.array([w_owners, w_ret, w_peak, w_recs, w_rev, w_play], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)

    games["GameScore_live"] = (
        w[0]*games["owners_count"] +
        w[1]*games["retention_score"] +
        w[2]*games["peak_concurrent_users"] +
        w[3]*games["recommendations"] +
        w[4]*games["review_score"] +
        w[5]*games["playtime_score"]
    )

    topn = int(max(1, topn))
    top = games.nlargest(topn, "GameScore_live").copy()

    view_cols = [
        "name","GameScore_live",
        "owners_count","retention_score","peak_concurrent_users",
        "recommendations","review_score","playtime_score"
    ]
    present = [c for c in view_cols if c in top.columns]
    table = top[present].rename(columns={"GameScore_live":"GameScore"})
    table["GameScore"] = table["GameScore"].round(3)
    table = table.reset_index(drop=True)

    fig = px.bar(
        table.sort_values("GameScore"),
        x="GameScore", y="name",
        orientation="h",
        hover_data={"GameScore":":.3f"},
        title=f"Top {topn} Games by GameScore"
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending", automargin=True),
                      margin=dict(l=220, r=40, t=60, b=40),
                      height=max(420, 24*len(table)+200))

    return table, fig

# ---------- GRADIO UI ----------
demo = gr.Interface(
    fn=compute_topN,
    inputs=[
        gr.Slider(0, 1, value=DEFAULT_WEIGHTS["owners_count"], step=0.01, label="owners_count weight"),
        gr.Slider(0, 1, value=DEFAULT_WEIGHTS["retention_score"], step=0.01, label="retention_score weight"),
        gr.Slider(0, 1, value=DEFAULT_WEIGHTS["peak_concurrent_users"], step=0.01, label="peak_concurrent_users weight"),
        gr.Slider(0, 1, value=DEFAULT_WEIGHTS["recommendations"], step=0.01, label="recommendations weight"),
        gr.Slider(0, 1, value=DEFAULT_WEIGHTS["review_score"], step=0.01, label="review_score weight"),
        gr.Slider(0, 1, value=DEFAULT_WEIGHTS["playtime_score"], step=0.01, label="playtime_score weight"),
        gr.Slider(5, 50, value=10, step=1, label="Top N"),
    ],
    outputs=[
        gr.Dataframe(label="Top N (recomputed)"),
        gr.Plot(label="Bar chart"),
    ],
    title="Best Video Game Explorer (Gradio, no CSV input)",
)

if __name__ == "__main__":
    demo.launch()
