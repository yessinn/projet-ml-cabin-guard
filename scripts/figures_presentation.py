"""Figures for the CabinGuard AI deck. All numbers come from reproduction.json."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
from matplotlib import font_manager

for f in font_manager.findSystemFonts():
    if "Inter" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Inter", "font.size": 15, "axes.edgecolor": "#d8d3c8", "figure.facecolor": "#f4f3ee", "axes.facecolor": "#f4f3ee", "savefig.facecolor": "#f4f3ee",
                     "axes.labelcolor": "#1B2A41", "xtick.color": "#4A5568", "ytick.color": "#4A5568"})

NAVY, TEAL, CORAL, GRAY, LIGHT, INK = "#191817", "#5a554e", "#c96442", "#8a7668", "#e6e1d6", "#191817"
OUT = Path(sys.argv[2])
OUT.mkdir(exist_ok=True)
R = json.load(open(sys.argv[1]))
KNN, SVM = R["modeles"]["KNN"], R["modeles"]["SVM"]


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, facecolor="#f4f3ee", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def box(ax, x, y, w, h, title, sub="", fc=LIGHT, tc=NAVY, fs=15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fc, ec="none"))
    if sub:
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=fs, weight="bold", color=tc)
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=fs - 3, color=tc)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, weight="bold", color=tc)


def arrow(ax, x1, y1, x2, y2, c=GRAY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, color=c, lw=2))


# 1. Pipeline
fig, ax = plt.subplots(figsize=(16, 4.2)); ax.set_xlim(0, 16); ax.set_ylim(0, 4.2); ax.axis("off")
steps = [("Photo", "habitacle"), ("Gris + EXIF", "orientation"), ("128 × 128", "proportions"),
         ("HOG", "8 100 valeurs"), ("Variance", "+ standardisation"), ("ANOVA", "k = 256"),
         ("KNN / SVM", "GridSearchCV"), ("Évaluation", "CV = 4 · F1")]
w, gap = 1.62, 0.37
for i, (t, s) in enumerate(steps):
    x = 0.2 + i * (w + gap)
    fc = NAVY if i in (3, 6) else (TEAL if i == 5 else LIGHT)
    tc = "white" if fc != LIGHT else NAVY
    box(ax, x, 1.2, w, 1.8, t, s, fc=fc, tc=tc, fs=14)
    if i < len(steps) - 1:
        arrow(ax, x + w + 0.03, 2.1, x + w + gap - 0.03, 2.1)
for (a, b, lab) in [(0, 2, "Prétraitement"), (3, 5, "Caractéristiques"), (6, 7, "Modélisation")]:
    x1, x2 = 0.2 + a * (w + gap), 0.2 + b * (w + gap) + w
    ax.plot([x1, x2], [0.85, 0.85], color=GRAY, lw=1.5)
    ax.text((x1 + x2) / 2, 0.45, lab, ha="center", fontsize=13, color=GRAY)
save(fig, "01_pipeline.png")

# 2. HOG explanation (synthetic cabin-like image)
yy, xx = np.mgrid[0:128, 0:128]
img = 0.35 + 0.15 * (yy / 128)
img[np.hypot(xx - 45, yy - 70) < 34] = 0.15; img[np.hypot(xx - 45, yy - 70) < 26] = 0.45
img[30:75, 85:108] = 0.08; img[34:71, 88:105] = 0.75
from scipy.ndimage import gaussian_filter
img = gaussian_filter(img, 1.2)
gy, gx = np.gradient(img); mag = np.hypot(gx, gy); ang = np.mod(np.arctan2(gy, gx), np.pi)
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.15]})
axs[0].imshow(img, cmap="gray", vmin=0, vmax=1); axs[0].set_title("1 · Image 128 × 128", color=NAVY, weight="bold")
axs[1].imshow(img, cmap="gray", vmin=0, vmax=1, alpha=0.35)
for cy in range(0, 128, 8):
    for cx in range(0, 128, 8):
        m = mag[cy:cy + 8, cx:cx + 8]
        if m.sum() < 0.25: continue
        a = np.average(ang[cy:cy + 8, cx:cx + 8], weights=m + 1e-9) + np.pi / 2
        L = min(3.6, 3.6 * m.sum() / 0.6)
        axs[1].plot([cx + 4 - L * np.cos(a), cx + 4 + L * np.cos(a)], [cy + 4 - L * np.sin(a), cy + 4 + L * np.sin(a)], color=CORAL, lw=1.6)
for k in range(0, 129, 8):
    axs[1].axhline(k - .5, color="#D5DBE1", lw=0.4); axs[1].axvline(k - .5, color="#D5DBE1", lw=0.4)
axs[1].set_title("2 · Contours par cellule 8 × 8", color=NAVY, weight="bold")
for a in axs[:2]: a.set_xticks([]); a.set_yticks([])
cell = (slice(40, 48), slice(16, 24))
h = np.bincount(np.minimum((ang[cell] * 9 / np.pi).astype(int), 8).ravel(), weights=mag[cell].ravel(), minlength=9)
axs[1].add_patch(Rectangle((15.5, 39.5), 8, 8, fill=False, ec=TEAL, lw=2.5))
axs[2].bar(range(9), h / h.max(), color=[TEAL] * 9, width=0.75)
axs[2].set_xticks(range(9), [f"{d}°" for d in range(0, 180, 20)], fontsize=12)
axs[2].set_yticks([]); axs[2].spines[["top", "right", "left"]].set_visible(False)
axs[2].set_title("3 · 9 orientations (1 cellule)", color=NAVY, weight="bold")
fig.text(0.5, -0.03, "16 × 16 cellules → blocs 2 × 2 normalisés → 15 × 15 blocs × 36 = 8 100 valeurs par image",
         ha="center", fontsize=15, color=NAVY)
save(fig, "02_hog.png")

# 3. Feature selection funnel
fig, ax = plt.subplots(figsize=(16, 4.4)); ax.set_xlim(0, 16); ax.set_ylim(0, 4.4); ax.axis("off")
stages = [("8 100", "coefficients HOG", 3.6, LIGHT, NAVY), ("VarianceThreshold", "retire les constantes", 3.0, LIGHT, NAVY),
          ("StandardScaler", "moyenne 0 · écart-type 1", 2.4, LIGHT, NAVY), ("SelectKBest", "k = 64, 256 ou toutes", 1.8, NAVY, "white"),
          ("256", "variables", 1.4, TEAL, "white")]
for i, (t, s, hgt, fc, tc) in enumerate(stages):
    x = 0.2 + i * 3.2
    box(ax, x, 2.2 - hgt / 2, 2.8, hgt, t, s, fc=fc, tc=tc, fs=15 if len(t) > 6 else 24)
    if i < 4: arrow(ax, x + 2.83, 2.2, x + 3.17, 2.2)
ax.text(8, 0.05, "Ajusté à chaque pli sur les seules images d'apprentissage (pas de fuite de données)", ha="center", fontsize=13, color=GRAY, style="italic")
save(fig, "03_selection.png")

# 4. KNN
rng = np.random.default_rng(3)
a0 = rng.normal([2.6, 3.2], 0.75, (14, 2)); a1 = rng.normal([5.2, 5.0], 0.75, (12, 2))
q = np.array([4.0, 4.1])
fig, ax = plt.subplots(figsize=(8, 6.4))
ax.scatter(*a0.T, s=110, color=TEAL, label="0 · pas de téléphone", zorder=3, edgecolor="white")
ax.scatter(*a1.T, s=110, color=CORAL, label="1 · téléphone visible", zorder=3, edgecolor="white")
allp = np.vstack([a0, a1]); lab = np.r_[np.zeros(14), np.ones(12)]
d = np.hypot(*(allp - q).T); idx = np.argsort(d)[:5]; r = d[idx].max() + 0.12
for i in idx: ax.plot([q[0], allp[i, 0]], [q[1], allp[i, 1]], color=GRAY, lw=1.2, ls="--", zorder=2)
ax.add_patch(Circle(q, r, fill=False, ec=NAVY, lw=2, ls=(0, (4, 3))))
ax.scatter(*q, s=260, marker="D", color="white", edgecolor=NAVY, lw=2.5, zorder=4, label="image à classer")
n1 = int(lab[idx].sum())
ax.set_title(f"k = 5 voisins : {n1} « téléphone » contre {5 - n1} → classe 1", fontsize=15, weight="bold", color=NAVY)
ax.set_xlabel("caractéristique HOG 1"); ax.set_ylabel("caractéristique HOG 2"); ax.set_xticks([]); ax.set_yticks([])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=12); ax.set_aspect("equal"); ax.spines[["top", "right"]].set_visible(False)
save(fig, "04_knn.png")

# 5. SVM
b0 = rng.normal([2.4, 2.6], 0.6, (14, 2)); b1 = rng.normal([5.4, 5.4], 0.6, (12, 2))
from sklearn.svm import SVC
X = np.vstack([b0, b1]); y = np.r_[np.zeros(14), np.ones(12)]
clf = SVC(kernel="linear", C=1).fit(X, y); wv, b = clf.coef_[0], clf.intercept_[0]
fig, ax = plt.subplots(figsize=(8, 6.4))
xs = np.linspace(0.5, 7.5, 10)
for off, st in [(0, "-"), (1, "--"), (-1, "--")]:
    ax.plot(xs, -(wv[0] * xs + b - off) / wv[1], color=NAVY, lw=2.5 if off == 0 else 1.5, ls=st)
ax.scatter(*b0.T, s=110, color=TEAL, edgecolor="white", zorder=3, label="0 · pas de téléphone")
ax.scatter(*b1.T, s=110, color=CORAL, edgecolor="white", zorder=3, label="1 · téléphone visible")
sv = clf.support_vectors_
ax.scatter(*sv.T, s=260, facecolor="none", edgecolor=NAVY, lw=2, zorder=4, label="vecteurs de support")
ax.set_xlim(0.5, 7.5); ax.set_ylim(0.5, 7.5)
ax.set_title("Trait plein : frontière · pointillés : marge", fontsize=15, weight="bold", color=NAVY)
ax.set_xlabel("caractéristique HOG 1"); ax.set_ylabel("caractéristique HOG 2"); ax.set_xticks([]); ax.set_yticks([])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=12); ax.set_aspect("equal"); ax.spines[["top", "right"]].set_visible(False)
save(fig, "05_svm.png")

# 6. Cross-validation
groups = [("Bus · barres jaunes", 5, 2), ("Bus · montants rouges", 17, 16), ("Bus · vitres bleues", 2, 1), ("Voiture · volant beige", 7, 7)]
fig, ax = plt.subplots(figsize=(16, 5.6)); ax.set_xlim(0, 16); ax.set_ylim(0, 5.6); ax.axis("off")
for j, (g, n0, n1) in enumerate(groups):
    ax.text(2.3 + j * 2.6 + 1.25, 5.2, g, ha="center", fontsize=12.5, color=NAVY, weight="bold")
    ax.text(2.3 + j * 2.6 + 1.25, 4.85, f"{n0} sans · {n1} avec", ha="center", fontsize=11.5, color=GRAY)
for i in range(4):
    yb = 3.6 - i * 1.0
    ax.text(0.2, yb + 0.35, f"Itération {i + 1}", fontsize=14, color=NAVY, weight="bold", va="center")
    for j in range(4):
        val = i == j
        box(ax, 2.3 + j * 2.6, yb, 2.5, 0.7, "Validation" if val else "Apprentissage", fc=CORAL if val else LIGHT,
            tc="white" if val else GRAY, fs=13)
    f1k, f1s = KNN["f1_macro_par_pli"][i], SVM["f1_macro_par_pli"][i]
    ax.text(15.9, yb + 0.35, f"KNN {f1k:.2f} · SVM {f1s:.2f}".replace(".", ","), ha="right", fontsize=11.5, color=NAVY, va="center")
ax.text(14.2, 4.85, "F1 macro", ha="center", fontsize=12.5, color=NAVY, weight="bold")
ax.text(8, -0.05, "GridSearchCV teste chaque réglage sur ces 4 itérations · critère : F1 macro moyen", ha="center", fontsize=13, color=GRAY, style="italic")
save(fig, "06_cv.png")

# 7. Confusion matrices
fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, (name, m, col) in zip(axs, [("KNN (k = 5)", KNN, TEAL), ("SVM linéaire (C = 0,1)", SVM, CORAL)]):
    cm = np.array(m["validation_interne"]["matrice_confusion"])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("c", ["#fbfaf6", col])
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max() * 1.15)
    for (r_, c_), v in np.ndenumerate(cm):
        ax.text(c_, r_, str(v), ha="center", va="center", fontsize=34, weight="bold", color="#fbfaf6" if v > cm.max()*0.6 else NAVY)
    ax.set_xticks([0, 1], ["0 · absent", "1 · présent"]); ax.set_yticks([0, 1], ["0 · absent", "1 · présent"])
    ax.set_xlabel("Classe prédite"); ax.set_ylabel("Classe réelle")
    ax.set_title(name, fontsize=18, weight="bold", color=NAVY, pad=12)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
save(fig, "07_confusion.png")

# 8. F1 per fold
fig, ax = plt.subplots(figsize=(12, 5.6))
x = np.arange(4); wd = 0.36
k, s = KNN["f1_macro_par_pli"], SVM["f1_macro_par_pli"]
b1_ = ax.bar(x - wd / 2, k, wd, color=TEAL, label=f"KNN · moyenne {KNN['f1_macro_moyen']:.3f}".replace(".", ","))
b2_ = ax.bar(x + wd / 2, s, wd, color=CORAL, label=f"SVM · moyenne {SVM['f1_macro_moyen']:.3f}".replace(".", ","))
ax.axhline(KNN["f1_macro_moyen"], color=TEAL, ls="--", lw=1.5); ax.axhline(SVM["f1_macro_moyen"], color=CORAL, ls="--", lw=1.5)
for bars in (b1_, b2_):
    for bb in bars:
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.02, f"{bb.get_height():.2f}".replace(".", ","), ha="center", fontsize=13, color=NAVY)
sizes = [7, 33, 3, 14]
ax.set_xticks(x, [f"Pli {i + 1}\n({n} images)" for i, n in enumerate(sizes)])
ax.set_ylim(0, 1.12); ax.set_ylabel("F1 macro"); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left", fontsize=13)
save(fig, "08_f1_plis.png")
print("ok", sorted(p.name for p in OUT.iterdir()))
