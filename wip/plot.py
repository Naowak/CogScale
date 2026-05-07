import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re

# 1. PARSING DES DONNÉES
# J'ai copié votre tableau markdown directement ici.
markdown_table = """
| Tâches                                    | DYNAMICAL_TRANSFORMER   | ESN             | GRU             | LSTM            | MAMBA           | TRANSFORMER_DECODER   | TRANSFORMER_ENCDEC   | XLSTM           |
|:------------------------------------------|:------------------------|:----------------|:----------------|:----------------|:----------------|:----------------------|:---------------------|:----------------|
| adding_problem - SM - 1k                  | 0.54 ± 0.28             | 0.47 ± 0.09     | **0.08 ± 0.05** | 0.50 ± 0.31     | 0.63 ± 0.21     | 0.56 ± 0.24           | 0.69 ± 0.06          | 0.32 ± 0.37     |
| adding_problem - SM - 10k                 | 0.12 ± 0.20             | **0.02 ± 0.03** | 0.13 ± 0.13     | 0.36 ± 0.33     | 0.63 ± 0.20     | 0.15 ± 0.22           | 0.42 ± 0.27          | 0.07 ± 0.06     |
| adding_problem - SM - 100k                | 0.05 ± 0.06             | nan             | 0.16 ± 0.10     | 0.55 ± 0.29     | 0.61 ± 0.23     | **0.04 ± 0.03** | 0.22 ± 0.32          | 0.06 ± 0.07     |
| adding_problem - MD - 1k                  | 0.11 ± 0.27             | 0.85 ± 0.02     | **0.01 ± 0.01** | 0.02 ± 0.01     | 0.88 ± 0.01     | 0.70 ± 0.29           | 0.88 ± 0.01          | 0.44 ± 0.45     |
| adding_problem - MD - 10k                 | 0.10 ± 0.28             | 0.71 ± 0.02     | 0.01 ± 0.00     | 0.01 ± 0.01     | 0.23 ± 0.37     | 0.01 ± 0.01           | 0.54 ± 0.44          | **0.00 ± 0.00** |
| adding_problem - MD - 100k                | **0.00 ± 0.00** | nan             | 0.01 ± 0.00     | 0.02 ± 0.01     | 0.23 ± 0.38     | 0.01 ± 0.01           | 0.20 ± 0.36          | **0.00 ± 0.00** |
| associative_recall - SM - 1k              | 0.93 ± 0.02             | **0.91 ± 0.03** | 0.92 ± 0.02     | 0.92 ± 0.03     | 0.92 ± 0.02     | **0.91 ± 0.03** | 0.94 ± 0.02          | 0.94 ± 0.03     |
| associative_recall - SM - 10k             | 0.92 ± 0.02             | 0.92 ± 0.03     | **0.91 ± 0.03** | 0.92 ± 0.02     | 0.93 ± 0.04     | 0.92 ± 0.03           | 0.93 ± 0.03          | 0.92 ± 0.03     |
| associative_recall - SM - 100k            | **0.90 ± 0.02** | nan             | 0.92 ± 0.02     | 0.92 ± 0.02     | 0.92 ± 0.02     | 0.93 ± 0.02           | 0.92 ± 0.02          | 0.92 ± 0.02     |
| associative_recall - MD - 1k              | 0.92 ± 0.01             | **0.88 ± 0.02** | 0.93 ± 0.01     | 0.93 ± 0.00     | 0.92 ± 0.01     | 0.93 ± 0.01           | 0.93 ± 0.01          | 0.92 ± 0.01     |
| associative_recall - MD - 10k             | 0.92 ± 0.01             | nan             | 0.93 ± 0.01     | 0.93 ± 0.01     | 0.92 ± 0.01     | **0.90 ± 0.03** | 0.92 ± 0.01          | 0.92 ± 0.01     |
| associative_recall - MD - 100k            | 0.91 ± 0.01             | nan             | 0.92 ± 0.01     | 0.93 ± 0.01     | 0.92 ± 0.01     | **0.89 ± 0.03** | 0.92 ± 0.02          | 0.91 ± 0.01     |
| bracket_matching - SM - 1k                | 0.39 ± 0.16             | **0.25 ± 0.11** | 0.30 ± 0.09     | 0.33 ± 0.06     | 0.39 ± 0.05     | 0.45 ± 0.08           | 0.48 ± 0.06          | 0.38 ± 0.06     |
| bracket_matching - SM - 10k               | 0.42 ± 0.07             | **0.20 ± 0.06** | 0.35 ± 0.03     | 0.36 ± 0.06     | 0.38 ± 0.04     | 0.46 ± 0.05           | 0.46 ± 0.06          | 0.28 ± 0.10     |
| bracket_matching - SM - 100k              | **0.26 ± 0.16** | nan             | 0.39 ± 0.11     | 0.38 ± 0.05     | 0.38 ± 0.04     | 0.45 ± 0.11           | 0.43 ± 0.07          | 0.32 ± 0.10     |
| bracket_matching - MD - 1k                | 0.09 ± 0.12             | 0.20 ± 0.02     | **0.06 ± 0.09** | 0.08 ± 0.11     | 0.27 ± 0.11     | 0.42 ± 0.10           | 0.47 ± 0.05          | 0.17 ± 0.13     |
| bracket_matching - MD - 10k               | 0.04 ± 0.02             | 0.16 ± 0.02     | 0.14 ± 0.13     | 0.07 ± 0.11     | 0.19 ± 0.11     | 0.23 ± 0.14           | 0.37 ± 0.14          | **0.03 ± 0.01** |
| bracket_matching - MD - 100k              | 0.07 ± 0.09             | nan             | 0.11 ± 0.11     | 0.20 ± 0.16     | 0.19 ± 0.13     | 0.11 ± 0.07           | 0.14 ± 0.14          | **0.02 ± 0.01** |
| simple_copy - SM - 1k                     | 0.40 ± 0.11             | 0.47 ± 0.01     | 0.64 ± 0.06     | 0.67 ± 0.01     | 0.65 ± 0.04     | **0.26 ± 0.33** | 0.65 ± 0.03          | 0.53 ± 0.05     |
| simple_copy - SM - 10k                    | 0.15 ± 0.18             | **0.00 ± 0.00** | 0.62 ± 0.06     | 0.66 ± 0.02     | 0.61 ± 0.05     | **0.00 ± 0.00** | 0.12 ± 0.26          | 0.49 ± 0.03     |
| simple_copy - SM - 100k                   | **0.00 ± 0.00** | nan             | 0.63 ± 0.05     | 0.65 ± 0.03     | 0.50 ± 0.05     | **0.00 ± 0.00** | **0.00 ± 0.00** | 0.46 ± 0.02     |
"""
# Note: j'ai réduit la liste des tâches dans cette string pour simplifier l'exemple, 
# mais le code fonctionne pour l'ensemble complet.

# Nettoyage et transformation en DataFrame
lines = [line.strip().strip('|') for line in markdown_table.strip().split('\n') if line.strip() and not line.startswith('|-')]
header = [col.strip() for col in lines[0].split('|')]
data = []

for line in lines[1:]:
    row = [col.strip() for col in line.split('|')]
    # Extraction des infos de la colonne Tâche (Ex: "adding_problem - SM - 1k")
    task_parts = row[0].split(' - ')
    task_name = task_parts[0].strip()
    size = task_parts[1].strip()
    seq_len = task_parts[2].strip()
    
    # Remplacement des '**' et extraction de la moyenne (ignorer l'écart-type pour les plots basiques)
    for i in range(1, len(row)):
        val_str = row[i].replace('**', '').strip()
        if val_str == 'nan':
            mean_val = np.nan
        else:
            mean_val = float(val_str.split('±')[0].strip())
        
        data.append({
            'Task': task_name,
            'Size': size,
            'Seq_Len': seq_len,
            'Model': header[i],
            'Mean': mean_val
        })

df = pd.DataFrame(data)

# Ordre correct pour les longueurs de séquence (catégoriel ordonné)
df['Seq_Len'] = pd.Categorical(df['Seq_Len'], categories=['1k', '10k', '100k'], ordered=True)

# Configuration globale pour papier scientifique
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
# Palette de couleurs distinctes (utile pour 8 modèles)
palette = sns.color_palette("tab10", n_colors=8)


# ==========================================
# GRAPHIQUE 1 : Évolution selon la longueur (Faceted Line Plot)
# ==========================================
# On crée une grille : Lignes = Complexité (SM/MD), Colonnes = Tâches
# (Filtre sur quelques tâches pour que ça soit lisible dans le snippet)
tasks_to_plot = ['adding_problem', 'associative_recall', 'bracket_matching', 'simple_copy']
df_filtered = df[df['Task'].isin(tasks_to_plot)]

g = sns.relplot(
    data=df_filtered, 
    x='Seq_Len', y='Mean', hue='Model', 
    col='Task', row='Size', 
    kind='line', marker='o', linewidth=2.5, markersize=8,
    palette=palette, height=3, aspect=1.2,
    facet_kws={'sharey': False} # Laisse l'axe Y libre car les tâches ont des échelles différentes
)

g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.set_axis_labels("Sequence Length", "Error/Metric Mean")
# Amélioration de la légende
sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 0), ncol=4, title=None, frameon=True)
plt.subplots_adjust(bottom=0.15) # Place pour la légende
g.fig.suptitle("Model Performance Scaling across Context Lengths", y=1.05, fontsize=16, fontweight='bold')
plt.savefig("scaling_plot.pdf", bbox_inches='tight') # Idéal pour LaTeX
plt.show()


# ==========================================
# GRAPHIQUE 2 : Carte de Chaleur (Heatmap) des performances globales
# ==========================================
# Création d'une matrice : Index = "Tâche (Taille/Len)", Colonnes = Modèles
df['Config'] = df['Task'] + " (" + df['Size'] + " - " + df['Seq_Len'].astype(str) + ")"
pivot_df = df.pivot(index='Config', columns='Model', values='Mean')

plt.figure(figsize=(12, 10))
# On utilise une colormap où les valeurs basses (meilleures) sont plus claires/bleues,
# et les valeurs hautes (mauvaises) sont rouges. (Ajustez si la métrique veut dire l'inverse !)
cmap = sns.diverging_palette(250, 10, as_cmap=True)

sns.heatmap(
    pivot_df, 
    cmap=cmap, 
    annot=True, # Affiche les valeurs dans les cases
    fmt=".2f",  # 2 décimales
    cbar_kws={'label': 'Mean Metric (Lower is usually better)'},
    linewidths=.5,
    na_color='lightgrey' # Pour les 'nan' de ESN
)
plt.title("Global Performance Heatmap Across All Configurations", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Architectures", fontsize=14)
plt.ylabel("Tasks & Constraints", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("heatmap_performance.pdf", bbox_inches='tight')
plt.show()


# ==========================================
# GRAPHIQUE 3 : Rang Moyen (Average Rank)
# ==========================================
# Calcul du rang pour chaque configuration (1 = le meilleur, 8 = le pire)
df['Rank'] = df.groupby('Config')['Mean'].rank(method='min', na_option='bottom')
avg_rank = df.groupby('Model')['Rank'].mean().sort_values()

plt.figure(figsize=(10, 6))
bars = sns.barplot(x=avg_rank.index, y=avg_rank.values, palette="viridis")
plt.title("Average Rank Across All Tasks (Lower is Better)", fontsize=16, fontweight='bold')
plt.ylabel("Average Rank", fontsize=14)
plt.xlabel("Model Architecture", fontsize=14)
plt.xticks(rotation=30, ha='right')

# Ajouter le chiffre au-dessus des barres
for p in bars.patches:
    bars.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')

plt.tight_layout()
plt.savefig("average_rank.pdf", bbox_inches='tight')
plt.show()