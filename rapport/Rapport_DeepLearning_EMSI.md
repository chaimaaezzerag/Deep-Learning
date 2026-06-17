# Rapport scientifique — Projet de fin de module Deep Learning

**Établissement :** EMSI Casablanca — Cycle MSI
**Module :** Deep Learning
**Type de travail :** Projet individuel (équivalent examen)
**Année universitaire :** 2025–2026

> Ce document est le **rapport de résultats** du projet. Le fichier `Sujet_Projet_DeepLearning_EMSI.md`, placé dans le même dossier, correspond à l'énoncé original (cahier des charges) et ne doit pas être confondu avec le présent rapport.

---

## 1. Introduction

Ce projet couvre trois grandes familles d'architectures de réseaux de neurones — perceptron multicouche (MLP), réseau de neurones convolutif (CNN) et réseaux récurrents (RNN/LSTM/GRU) assemblés en un système séquence-à-séquence (Seq2Seq) — appliquées respectivement à des données tabulaires, des images et du texte. L'objectif n'est pas seulement d'obtenir des modèles fonctionnels, mais de comprendre **pourquoi** une architecture donnée convient à un type de données donné, à travers une implémentation manuelle de certains blocs de base, une comparaison numérique avec les couches PyTorch équivalentes, et une analyse critique systématique des résultats.

## 2. Objectifs

- Manipuler `nn.Module`, les paramètres, le calcul de gradient et la sauvegarde/chargement de modèles dans les trois contextes (tabulaire, image, séquence).
- Comparer plusieurs choix de conception (stratégies d'initialisation, hyperparamètres convolutifs, type de cellule récurrente) à l'aide de tableaux et de courbes.
- Implémenter manuellement certaines opérations clés (convolution, pooling) et vérifier leur cohérence numérique avec l'implémentation PyTorch correspondante.
- Évaluer chaque modèle avec des métriques adaptées (accuracy, précision, rappel, F1, matrice de confusion pour la classification ; perplexité et BLEU pour la génération de séquences).
- Relier les trois parties dans une discussion transversale sur l'adéquation entre architecture et structure des données.

## 3. Méthodologie générale

L'ensemble du projet est implémenté en Python avec PyTorch, sous la forme de trois notebooks Jupyter indépendants (`part1_mlp.ipynb`, `part2_cnn.ipynb`, `part3_rnn.ipynb`), chacun correspondant à une partie du cahier des charges. Pour chaque partie, la même démarche est suivie : préparation des données et séparation train/validation/test, implémentation du ou des modèles, entraînement avec suivi de la perte, évaluation quantitative, puis analyse critique et question de synthèse spécifique à la partie. Les modèles entraînés sont sauvegardés au format `.pth` dans le dossier `saved_models/`. L'exécution a été réalisée sur un environnement Python local (CPU) disposant de PyTorch 2.x, pandas, scikit-learn et matplotlib.

## 4. Partie I — MLP sur données tabulaires

### 4.1 Données

Le jeu de données utilisé est `winequality-red.csv` (1599 lignes, 12 colonnes : 11 variables physico-chimiques et une note de qualité). Il a été séparé en 959 exemples d'entraînement, 320 de validation et 320 de test.

### 4.2 Implémentation

Deux versions équivalentes du même MLP ont été implémentées : une via `nn.Sequential` et une via une classe personnalisée héritant de `nn.Module`, afin d'illustrer les deux approches de construction d'un réseau et de comparer leurs paramètres via `named_parameters()` et `state_dict()`.

### 4.3 Stratégies d'initialisation

Trois stratégies d'initialisation des poids ont été comparées : initialisation gaussienne, initialisation constante, et initialisation de Xavier. La boucle d'entraînement comparative (nouvellement ajoutée, voir notebook section « Comparaison expérimentale des trois stratégies d'initialisation ») entraîne le même modèle avec chacune des trois stratégies et produit un tableau récapitulatif (perte finale d'entraînement et de validation, accuracy, précision, rappel, F1) ainsi que les courbes de perte correspondantes. **Cette cellule n'a pas pu être exécutée dans l'environnement utilisé pour la rédaction de ce rapport** (absence de PyTorch) ; le tableau et les courbes seront produits automatiquement à la prochaine exécution locale du notebook. D'un point de vue théorique, on attend de l'initialisation constante une absence de brisure de symétrie (tous les neurones d'une même couche reçoivent un gradient identique et évoluent de façon identique), ce qui devrait se traduire par une perte qui stagne nettement plus haut que pour les deux autres stratégies ; l'initialisation de Xavier, conçue pour préserver la variance des activations à travers les couches, devrait converger au moins aussi rapidement que l'initialisation gaussienne simple.

### 4.4 Résultats (modèle entraîné, initialisation Xavier)

| Modèle | Accuracy (val) | Précision (val) | Rappel (val) | F1 (val) |
|---|---|---|---|---|
| MLP (`nn.Sequential`) | 0,7156 | 0,7564 | 0,6901 | 0,7217 |
| MLP (classe personnalisée) | 0,7188 | 0,7368 | 0,7368 | 0,7368 |

Ces résultats correspondent à l'exécution réelle déjà réalisée dans le notebook original (initialisation de Xavier). Les deux implémentations donnent des résultats très proches, ce qui est attendu puisqu'il s'agit du même modèle exprimé de deux façons différentes.

### 4.5 Interprétation

Les performances obtenues (accuracy autour de 0,72) sont cohérentes avec la difficulté intrinsèque de la tâche : prédire une note de qualité de vin à partir de mesures physico-chimiques est un problème bruité, où la variable cible est elle-même une moyenne d'évaluations humaines subjectives. L'écart minime entre les deux implémentations du MLP confirme la cohérence de l'API PyTorch entre les deux styles de construction de modèle.

## 5. Partie II — CNN sur données image

### 5.1 Données

Le jeu de données utilisé est Fashion-MNIST (images en niveaux de gris 28×28, 10 classes : T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot), téléchargé automatiquement via `torchvision`.

### 5.2 Implémentations manuelles vs PyTorch

Une convolution 2D, un max-pooling et un average-pooling ont été implémentés manuellement (boucles explicites sur les positions spatiales) puis comparés numériquement à `nn.Conv2d`, `nn.MaxPool2d` et `nn.AvgPool2d` sur le même exemple d'image. L'écart maximal observé entre la convolution manuelle et `nn.Conv2d` est de l'ordre de **9,5 × 10⁻⁷**, ce qui confirme que l'implémentation manuelle est numériquement équivalente à l'implémentation PyTorch (l'écart résiduel s'explique par la précision flottante). La comparaison équivalente pour le max-pooling et l'average-pooling a été ajoutée au notebook (cellule « Comparaison numérique : pooling manuel vs couches PyTorch ») mais n'a pas pu être exécutée dans l'environnement de rédaction ; on attend, par construction des deux implémentations, un écart du même ordre de grandeur (erreur de précision flottante uniquement).

### 5.3 Modèles et résultats

Un MLP de référence (image aplatie en vecteur) et un CNN inspiré de LeNet ont été entraînés et évalués sur l'ensemble de test complet :

| Modèle | Accuracy (test) |
|---|---|
| MLP (baseline) | 0,8678 |
| CNN | 0,9028 |

Le CNN améliore l'accuracy de test de plus de 3 points par rapport au MLP, ce qui illustre directement l'intérêt d'exploiter la structure spatiale de l'image plutôt que de la traiter comme un vecteur de pixels indépendants.

### 5.4 Étude expérimentale des choix architecturaux

Une étude d'ablation a été ajoutée (notebook, section « Étude expérimentale de l'influence des choix architecturaux ») : à partir d'une configuration de référence, cinq variantes sont testées en ne changeant qu'un facteur à la fois — absence de padding, stride = 2 dans les convolutions, average pooling au lieu de max pooling, davantage de filtres, ajout d'une convolution 1×1 — sur un sous-échantillon (5000 exemples d'entraînement, 2000 de test, 3 époques) afin de garder l'étude rapide. **Cette étude n'a pas pu être exécutée dans l'environnement de rédaction** ; le tableau et l'histogramme comparatif correspondants seront produits à la prochaine exécution locale du notebook. Sur le plan théorique, on attend que l'absence de padding réduise légèrement la taille effective des cartes de caractéristiques (formule $(n+2p-k)/s+1$) et puisse pénaliser l'information portée par les bords de l'image, que le stride=2 joue un rôle de sous-échantillonnage proche du pooling mais sans son invariance locale, que l'average pooling lisse l'information par rapport au max pooling, et que l'ajout de filtres ou d'une convolution 1×1 augmente la capacité du modèle au prix d'un coût de calcul plus élevé.

### 5.5 Comparaison détaillée MLP vs CNN

Une comparaison plus complète (accuracy, précision/rappel/F1 macro, nombre de paramètres) a été ajoutée au notebook (section « Comparaison détaillée MLP vs CNN ») afin de compléter le simple écart d'accuracy déjà mesuré ci-dessus par une vision multi-métrique. Cette cellule recalcule notamment l'accuracy déjà obtenue (0,8678 / 0,9028) avec les métriques macro-moyennées additionnelles ; les valeurs précises de ces métriques complémentaires seront disponibles après ré-exécution du notebook.

## 6. Partie III — RNN / LSTM / GRU et Seq2Seq sur données textuelles

### 6.1 Données

Le corpus utilisé est un petit corpus parallèle français-anglais (`fra-eng.txt`) comportant **40 phrases**. Après construction du vocabulaire, la taille du vocabulaire source est de 141 tokens et celle du vocabulaire cible de 134 tokens. Le corpus a été séparé en 32 phrases d'entraînement et 8 phrases de test.

### 6.2 Modèles et entraînement

Trois variantes d'un même système encodeur-décodeur ont été entraînées sur 8 époques, en ne changeant que le type de cellule récurrente (RNN simple, LSTM, GRU) :

| Architecture | Perte train (époque 0 → 6) | Perte validation (époque 0 → 6) |
|---|---|---|
| RNN simple | 4,896 → 3,960 | 4,772 → 4,302 |
| LSTM | 4,884 → 4,237 | 4,854 → 4,405 |
| GRU | 4,917 → 4,123 | 4,851 → 4,314 |

Les trois architectures convergent de façon cohérente sur ce petit corpus, le GRU atteignant la perte d'entraînement la plus basse après 6 époques parmi les trois.

### 6.3 Illustration de l'effet du gradient clipping

Le gradient clipping (`torch.nn.utils.clip_grad_norm_`, seuil = 1,0) est déjà utilisé par défaut dans la boucle d'entraînement principale. Pour en illustrer concrètement l'effet, une expérience dédiée a été ajoutée (notebook, section « Illustration expérimentale de l'effet du gradient clipping ») : un RNN simple est entraîné sur 30 pas avec un taux d'apprentissage volontairement élevé (0,05), avec et sans clipping actif, en enregistrant à chaque pas la norme du gradient avant clipping. **Cette expérience n'a pas pu être exécutée dans l'environnement de rédaction** ; les courbes correspondantes (norme du gradient et perte, avec et sans clipping) seront produites à la prochaine exécution locale. On attend, conformément à la théorie de la rétropropagation à travers le temps (BPTT), que la norme du gradient dépasse occasionnellement le seuil de clipping en l'absence de celui-ci, et que le clipping se traduise par une trajectoire de perte plus régulière.

### 6.4 Décodage et BLEU (exemples)

Le décodage glouton et le beam search (largeur 3) ont été testés sur les 5 premières phrases de l'ensemble de test. À titre d'exemple, pour la phrase source « ce film est très intéressant. » (référence : « this movie is very interesting . »), le RNN simple obtient un BLEU de 0,245 (glouton) et 0,455 (beam search), tandis que le GRU obtient un BLEU de 0,655 (glouton) sur la même phrase. Ces exemples confirment que le beam search égale ou dépasse fréquemment le décodage glouton, et que les performances de traduction varient sensiblement d'une phrase à l'autre sur ce corpus très réduit.

### 6.5 Tableau comparatif (perplexité et BLEU moyen sur l'ensemble de test complet)

Un tableau comparatif a été ajouté au notebook (section « Tableau comparatif RNN / LSTM / GRU — perplexité et BLEU sur l'ensemble de test complet ») afin de dépasser la simple illustration sur 5 phrases : il calcule, pour chaque architecture, la perte de validation, la perplexité associée ($\exp(\text{perte})$), le BLEU moyen (glouton et beam search) sur l'intégralité de l'ensemble de test, ainsi que le nombre de paramètres. **Ce tableau n'a pas pu être calculé dans l'environnement de rédaction** ; il sera disponible après ré-exécution du notebook. Le meilleur modèle (perplexité la plus faible) est automatiquement sauvegardé dans `saved_models/part3_seq2seq_best.pth` par la cellule suivante.

### 6.6 Limites spécifiques à cette partie

Avec seulement 40 phrases au total (32 pour l'entraînement), le corpus est largement insuffisant pour qu'un modèle Seq2Seq apprenne une traduction généralisable : les traductions produites (visibles dans les exemples du notebook, par exemple « the is is » ou « the the the ») reflètent davantage une mémorisation partielle de motifs fréquents qu'une compréhension linguistique. Les résultats de cette partie doivent donc être interprétés comme une **démonstration du fonctionnement et du comportement attendu des architectures** plutôt que comme un système de traduction performant.

## 7. Discussion transversale

> *Comment le deep learning adapte-t-il ses architectures à la structure des données – tabulaire, image et séquentielle – et pourquoi un même paradigme d'apprentissage supervisé doit-il être décliné différemment selon la géométrie, la dépendance locale, la temporalité et la représentation des données ?*

Les trois parties de ce projet illustrent un même principe directeur : le choix d'une architecture de réseau de neurones n'est jamais arbitraire, il découle directement de la structure géométrique et statistique des données à traiter. Un même paradigme — l'apprentissage supervisé par descente de gradient et rétropropagation — se décline en trois familles d'architectures différentes parce que les données tabulaires, les images et les séquences textuelles ne partagent ni la même géométrie, ni le même type de dépendance locale, ni la même relation à la temporalité ou à la représentation.

```
Données tabulaires  →  vecteurs indépendants      →  MLP   (connectivité totale)
Données image        →  grille spatiale 2D         →  CNN   (connectivité locale + partage de poids)
Données séquentielles →  ordre temporel, longueur variable →  RNN/LSTM/GRU + Seq2Seq (état caché récurrent)
```

**Données tabulaires et MLP.** Le jeu de données *winequality-red* est constitué de lignes indépendantes, décrites par des variables physico-chimiques sans relation d'ordre ni de proximité particulière entre elles. Il n'existe donc aucune structure spatiale ou séquentielle à exploiter : un perceptron pleinement connecté est le choix le plus naturel. Les deux versions du MLP obtiennent des performances très proches (accuracy de validation autour de 0,715–0,719), ce qui confirme qu'aucune structure cachée n'est laissée inexploitée par cette architecture pour ce type de données.

**Données image et CNN.** Une image possède une structure spatiale forte : les pixels voisins sont corrélés et un même motif visuel peut apparaître à différents endroits. Le CNN exploite cette structure par la connectivité locale et le partage de paramètres, ce qui lui confère une forme d'invariance à la translation que ne possède pas un MLP. Les résultats expérimentaux le confirment directement : le CNN atteint 0,9028 d'accuracy de test contre 0,8678 pour le MLP sur la même tâche — un écart qui s'explique par la perte de toute notion de voisinage entre pixels dès l'aplatissement de l'image en vecteur dans le MLP.

**Données séquentielles et RNN/Seq2Seq.** Le texte introduit une troisième contrainte, absente des deux cas précédents : la temporalité et la dépendance à longue portée dans l'ordre des éléments. Les architectures récurrentes y répondent en maintenant un état caché mis à jour séquentiellement, qui résume le contexte passé. Le passage du RNN simple vers LSTM/GRU, motivé par le problème de vanishing/exploding gradient lié à la rétropropagation à travers le temps, illustre que des raffinements supplémentaires (portes de mémoire, gradient clipping) sont nécessaires pour stabiliser l'apprentissage sur des séquences. Le passage à un schéma encodeur-décodeur répond enfin à un besoin de représentation différent : générer une séquence cible de longueur variable plutôt que produire une simple classe.

**Synthèse.** La performance d'un modèle ne dépend pas seulement de son nombre de paramètres, mais de l'adéquation entre les biais structurels de l'architecture (connectivité totale, connectivité locale avec partage de poids, ou récurrence avec mémoire) et la géométrie intrinsèque des données (vecteurs indépendants, grilles spatiales, séquences ordonnées). Il n'existe pas d'architecture universelle : le choix doit être guidé par la nature même des données à modéliser.

## 8. Limites générales du projet

- Les expériences nouvellement ajoutées (comparaison des stratégies d'initialisation pour la Partie I, étude architecturale et comparaison détaillée pour la Partie II, illustration du gradient clipping et tableau de perplexité/BLEU pour la Partie III) ont été intégrées au code des notebooks mais **n'ont pas pu être exécutées** dans l'environnement utilisé pour rédiger ce rapport (absence de PyTorch et d'accès réseau). Elles doivent être exécutées une fois sur la machine de développement habituelle avant la remise finale, afin que les tableaux, courbes et le modèle `part3_seq2seq_best.pth` soient effectivement générés.
- Le corpus de traduction (40 phrases) est trop restreint pour permettre une généralisation réelle des modèles Seq2Seq ; les résultats de la Partie III doivent être lus comme illustratifs.
- Les métriques de la Partie I (accuracy ≈ 0,72) restent modestes, ce qui reflète la difficulté intrinsèque de prédire une note de qualité subjective à partir de mesures physico-chimiques plutôt qu'une limite de l'implémentation.

## 9. Conclusion

Ce projet a permis de mettre en œuvre, de comparer et d'analyser trois familles d'architectures de deep learning sur trois types de données distincts. Les résultats obtenus — un MLP performant mais limité par la nature peu structurée des données tabulaires, un CNN surpassant nettement un MLP sur des données image grâce à l'exploitation de la structure spatiale, et des architectures récurrentes capables d'apprendre (même sur un corpus minuscule) une tâche de traduction séquence-à-séquence — confirment empiriquement le principe directeur de la discussion transversale : l'architecture doit être choisie en fonction de la structure intrinsèque des données, et non l'inverse.

---

*Rapport rédigé à partir des résultats réels obtenus lors de l'exécution des notebooks fournis dans `notebooks/`, complété par la description méthodologique des expériences ajoutées dans le cadre de la consolidation du projet (voir section « Limites générales du projet » pour le détail des éléments restant à exécuter localement).*
