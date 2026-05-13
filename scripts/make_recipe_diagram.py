"""Generate a clean recipe-pipeline diagram for the README."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

fig, ax = plt.subplots(figsize=(13, 6.4))
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.4)
ax.axis("off")

stages = [
    {"n": "1", "title": "PROBLEM\nGENERATION",
     "body": "Base model emits Python\nfunction + 3 asserts.\nKeep only those where\nthe canonical passes.",
     "color": "#264653", "text": "white"},
    {"n": "2", "title": "DIVERSE\nSAMPLING",
     "body": "Resample 4–8 attempts\nat T=0.7–0.8. Run\neach against the\nasserts.",
     "color": "#2A9D8F", "text": "white"},
    {"n": "3", "title": "AT-EDGE\nMINING",
     "body": "If some pass and\nsome fail → (broken,\nfixed) pair. Skip if\nall-pass or all-fail.",
     "color": "#E9C46A", "text": "#222"},
    {"n": "4", "title": "LoRA\nTRAIN",
     "body": "Rank 16–32, q/k/v/o\nprojections. 2 epochs,\nlr=1e-4. No human\ndata, no RL.",
     "color": "#F4A261", "text": "white"},
    {"n": "5", "title": "EVALUATE",
     "body": "HumanEval / HE+ /\nMBPP / GSM8K. Verify\nthe lift before\ndeclaring a win.",
     "color": "#E76F51", "text": "white"},
]

box_w, box_h = 2.15, 3.55
gap = 0.30
total_w = len(stages) * box_w + (len(stages) - 1) * gap
x0 = (13 - total_w) / 2
y0 = 1.55  # raise the row so loop arc has more space below

for i, s in enumerate(stages):
    x = x0 + i * (box_w + gap)
    shadow = FancyBboxPatch(
        (x + 0.05, y0 - 0.05), box_w, box_h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=0, facecolor="#00000020")
    ax.add_patch(shadow)
    card = FancyBboxPatch(
        (x, y0), box_w, box_h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.2, edgecolor="#222", facecolor=s["color"])
    ax.add_patch(card)

    circ_x, circ_y = x + 0.32, y0 + box_h - 0.32
    ax.add_patch(plt.Circle((circ_x, circ_y), 0.24,
                            facecolor="white", edgecolor=s["color"],
                            linewidth=2, zorder=4))
    ax.text(circ_x, circ_y, s["n"], ha="center", va="center",
            fontsize=13, fontweight="bold", color=s["color"], zorder=5)

    ax.text(x + box_w/2, y0 + box_h - 0.85, s["title"],
            ha="center", va="center", fontsize=12.5, fontweight="bold",
            color=s["text"])
    ax.text(x + box_w/2, y0 + 1.15, s["body"],
            ha="center", va="center", fontsize=9.5, color=s["text"],
            linespacing=1.35)

    if i < len(stages) - 1:
        ax_x = x + box_w + 0.02
        ax_xe = x + box_w + gap - 0.02
        ay = y0 + box_h / 2
        ax.add_patch(FancyArrowPatch(
            (ax_x, ay), (ax_xe, ay),
            arrowstyle="-|>", mutation_scale=18,
            color="#333", linewidth=2.0))

# Title
ax.text(6.5, 6.05, "The TinyForge-Zero recipe",
        ha="center", va="center", fontsize=17, fontweight="bold", color="#1a1a1a")
ax.text(6.5, 5.55,
        "No human-written training data — the base model bootstraps from its own divergent attempts.",
        ha="center", va="center", fontsize=10.5, color="#555", style="italic")

# Loop-back arrow well below the cards
sx = x0 + 4 * (box_w + gap) + box_w / 2
ex = x0 + 2 * (box_w + gap) + box_w / 2
ax.annotate("",
    xy=(ex, y0 - 0.05), xytext=(sx, y0 - 0.05),
    arrowprops=dict(arrowstyle="-|>",
                    connectionstyle="arc3,rad=0.30",
                    color="#999", linewidth=1.3,
                    linestyle=(0, (4, 3))))
ax.text((sx + ex) / 2, y0 - 0.95, "iterate: mine more pairs, retrain",
        ha="center", va="center", fontsize=9.0, color="#888", style="italic")

# Footer
ax.text(6.5, 0.30,
        "Control: replacing self-mined pairs with mechanically-corrupted external pairs yields +0 — the signal is in the content, not the format.",
        ha="center", va="center", fontsize=9.0, color="#666", style="italic")

plt.tight_layout()
plt.savefig("/Users/usman/tinyforge-zero/docs/recipe_diagram.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig("/Users/usman/tinyforge-zero/docs/recipe_diagram.pdf",
            bbox_inches="tight", facecolor="white")
print("Diagram regenerated.")
