import pandas as pd

zonos = {
    "Pitch": 0.60,
    "Timbre": 0.37,
    "Voice quality": 0.26,
    "Pauses": 0.23,
    "Energy": 0.20,
    "Tempo": 0.17
}

chatterbox = pd.read_csv("results/ablation_accuracy.csv")
chatterbox_dict = dict(zip(chatterbox["Channel"], chatterbox["Accuracy"]))

comparison = pd.DataFrame({
    "Channel": list(zonos.keys()),
    "Zonos_accuracy": list(zonos.values()),
    "Chatterbox_accuracy": [chatterbox_dict.get(k, None) for k in zonos.keys()]
})
comparison["Rank_change"] = (
    comparison["Zonos_accuracy"].rank(ascending=False).astype(int).astype(str)
    + " → "
    + comparison["Chatterbox_accuracy"].rank(ascending=False).astype(int).astype(str)
)

comparison.to_csv("results/comparison_table.csv", index=False)
print(comparison.to_string(index=False))

# Largest physical mover
heatmap_df = pd.read_csv("results/heatmap_values.csv", index_col=0)
largest_mover = heatmap_df.mean(axis=1).idxmax()
print(f"\nChatterbox largest physical mover: {largest_mover}")
print(f"Zonos largest physical mover: Timbre")
