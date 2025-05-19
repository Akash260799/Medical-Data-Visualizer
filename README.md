# Medical-Data-Visualizer

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import data
df = pd.read_csv('medical_examination.csv')

# 2. Add overweight column
# BMI = weight (kg) / (height (m))^2
bmi = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (bmi > 25).astype(int)

# 3. Normalize data: cholesterol and gluc (0 = good, 1 = bad)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

def draw_cat_plot():
    # 4. Create DataFrame for cat plot
    df_cat = pd.melt(df,
                     id_vars=['cardio'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # 5. Group and reformat data
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']) \
                   .size() \
                   .reset_index(name='total')

    # 6. Draw the catplot
    fig = sns.catplot(
        x='variable', y='total', hue='value', col='cardio',
        data=df_cat, kind='bar'
    ).fig

    return fig

def draw_heat_map():
    # 7. Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 8. Calculate correlation matrix
    corr = df_heat.corr()

    # 9. Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 10. Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 11. Draw heatmap
    sns.heatmap(
        corr,
        annot=True,
        fmt='.1f',
        mask=mask,
        square=True,
        linewidths=.5,
        cbar_kws={'shrink': .5},
        ax=ax
    )

    return fig
