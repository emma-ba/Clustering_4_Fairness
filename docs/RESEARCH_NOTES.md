# Research notes and open threads

Moved out of `c4fairness/cli.py`, where they sat as comments in the middle of the
argparse block. Verbatim except where marked; original wording (including the French
notes) kept so nothing is lost in translation.

`docs/RESEARCH.md` is deleted in the working tree but still present in `HEAD`. If it
comes back, these belong there and this file should go away.

## Method

- **Euclidean vs Gower, side by side.** For the same k, show cluster proportions, error
  separation (chi-square / Kruskal-Wallis), and sensitive-feature distribution per
  cluster under both distances. The point is to establish whether Gower adds anything
  over plain Euclidean.
- **When is which better on mixed data?** Run the experiment sweep across the three
  options on the datasets already in `Data/`, and check whether the answer is
  consistent or depends on the balance between categorical and numerical features.
- **Iterative clustering.** Untested.
- **Finding k.** Confirm the k-search behaves as intended.
- **K-centroid variant**, including a fair-centroid version.
- **Ranking / recommender systems.** These need precision and recall as error measures,
  for clustering that considers several error forms at once. *Original: "On peut faire un
  clustering qui considere +ieurs formes d'erreur. Pour pb de ranking, on a P & Recall —
  pour + tard."*
- **NDCG.** Only one datapoint is needed, same arrangement as regression. *Original: "On
  a besoin juste d'un datapoint pour le ndcg. Meme system que pour regression."*

## Publication

- Look into journals that accept research artifacts, or a demo track at a conference.
- Open-science journals, one-vs-all.
- ACM artifact badge.
- Documentation.

## Done since these were written

- **Web app.** *"site web ou on peut uploader le dataset, confirmer les colonnes à
  utiliser, sensitives. Penser un peu aux tests qu'on peut appliquer."* Shipped as
  `c4fairness/webapp.py` (`c4fairness-web`).
