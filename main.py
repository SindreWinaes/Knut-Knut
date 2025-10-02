import json
from collections import defaultdict
import pandas as pd
import plotly.express as px
from datetime import datetime, time

from train import (
train_baseline, predict_baseline, train_quadratic, predict_quadratic, train_sinus, predict_sinus,
train_sinus2_fn, predict_sinus2, save_models, find_best_route
)


def to_minutes(hhmm):
    h, m = map(int, hhmm.split(":"))
    return h * 60  + m

def to_clock_time(m):
    hh = m // 60
    mm = m % 60
    return f"{hh:02d}:{mm:02d}"



traffic_data = "traffic.jsonl"
paths = defaultdict(list)

with open(traffic_data, "r") as f:

    for line in f:

        record = json.loads(line)

        road = record.get("road")
        departure = record.get("depature")
        arrival = record.get("arrival")

        departure = to_minutes(departure)
        arrival = to_minutes(arrival)

        travel_time = arrival - departure

        paths[road].append({
            "depature" : departure,
            "arrival" : arrival,
            "travel_time" : travel_time
        })

A_C_D = paths["A->C->D"]
A_C_E = paths["A->C->E"]
B_C_D = paths["B->C->D"]
B_C_E = paths["B->C->E"]



A_C_D_df = pd.DataFrame(A_C_D)
A_C_E_df = pd.DataFrame(A_C_E)
B_C_D_df = pd.DataFrame(B_C_D)
B_C_E_df = pd.DataFrame(B_C_E)


A_C_D_df["dep_dt"] = pd.to_datetime(A_C_D_df["depature"], unit="m", origin="2000-01-01")
A_C_E_df["dep_dt"] = pd.to_datetime(A_C_E_df["depature"], unit="m", origin="2000-01-01")
B_C_D_df["dep_dt"] = pd.to_datetime(B_C_D_df["depature"], unit="m", origin="2000-01-01")
B_C_E_df["dep_dt"] = pd.to_datetime(B_C_E_df["depature"], unit="m", origin="2000-01-01")



models = {
    "A->C->D": train_quadratic(A_C_D_df),
    "A->C->E": train_baseline(A_C_E_df),
    "B->C->D": train_sinus2_fn(B_C_D_df),
    "B->C->E": train_sinus(B_C_E_df)
}

save_models(models, filename="trained_model.pkl")

print("\nModel losses:")
for road, model in models.items():
    print(f"{road}: loss={model['loss']:.3f}")

def predict_for(road, t_min):

    m = models[road]
    if m["type"] == "baseline":
        return predict_baseline(m, t_min)
    if m["type"] == "quadratic":
        return predict_quadratic(m, t_min)
    if m["type"] == "sinus":
        return predict_sinus(m, t_min)
    if m["type"] == "sinus2":
        return predict_sinus2(m, t_min)
    raise ValueError(f"Unoown model type: {m['type']}")


def best_path_at(hhmm):
    t = to_minutes(hhmm)
    preds = {road: predict_for(road, t) for road in models}
    best = min(preds, key=preds.get)
    return best, preds

best, preds = best_path_at("08:30")
print("Predicted times @ 08:30:", {k: round(v,1) for k,v in preds.items()})
print("Best path:", best)

print("\nBest path predictions (every 30 min from 07:00–17:00):")
for hh in range(7, 17):
    for mm in [0, 30]:
        best, est = find_best_route(hh, mm, models)  # <- est is a float
        print(f"{hh:02d}:{mm:02d} -> Best: {best} | est={est:.1f} min")




'''
xmin = pd.to_datetime(390, unit="m", origin="2000-01-01")   # 06:30
xmax = pd.to_datetime(960, unit="m", origin="2000-01-01")

fig = px.scatter(
    A_C_D_df,
    x="dep_dt",
    y="travel_time",
    title="A->C->D"
)

fig.update_xaxes(range=[xmin, xmax],tickformat="%H:%M",dtick=30*60*1000,  title="Departure (HH:MM)")
fig.update_yaxes(title="Travel time (minutes)")

fig.show()
'''