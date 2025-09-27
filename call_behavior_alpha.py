import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import defaultdict
import matplotlib.colors as mcolors
import colorsys
import numpy as np
import io
import base64
from matplotlib.patches import Patch

matplotlib.use('Agg')  # ✅ server-friendly backend

def generate_behavior_heatmap():
    # Load and preprocess
    df = pd.read_csv('call_entry_raw.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['callType'] = df['callType'].str.strip().str.lower()
    df['number'] = df['number'].str.strip().str.strip("'").str.strip('"')
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.to_period('M').apply(lambda r: r.start_time)

    incoming = df[df['callType'] == 'incoming'].groupby('date').size()
    outgoing = df[df['callType'] == 'outgoing'].groupby('date').size()
    all_dates = pd.date_range(start=min(incoming.index.min(), outgoing.index.min()),
                              end=max(incoming.index.max(), outgoing.index.max()))
    incoming = incoming.reindex(all_dates.date, fill_value=0)
    outgoing = outgoing.reindex(all_dates.date, fill_value=0)

    # Behavior clustering
    top_numbers = df['number'].value_counts().head(20).index.tolist()
    df_top = df[df['number'].isin(top_numbers)]
    monthly_stats = df_top.groupby(['number', 'month', 'callType']).size().unstack(fill_value=0).reset_index()
    agg_features = monthly_stats.groupby('number').agg({
        'incoming': ['sum', 'mean', 'std', 'count'],
        'outgoing': ['sum', 'mean', 'std', 'count']
    })
    agg_features.columns = ['_'.join(col).strip() for col in agg_features.columns.values]
    agg_features.fillna(0, inplace=True)

    scaled = StandardScaler().fit_transform(agg_features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    agg_features['cluster'] = kmeans.fit_predict(scaled)

    pca_result = PCA(n_components=2).fit_transform(scaled)
    agg_features[['pca1', 'pca2']] = pca_result

    cluster_map = {0: 'Frequent', 1: 'Seasonal', 2: 'One-Way'}
    agg_features['behavior_type'] = agg_features['cluster'].map(cluster_map)

    # Monthly evolution
    behavior_evolution = defaultdict(dict)
    all_months = sorted(df['month'].unique())

    for month in all_months:
        df_month = df[df['month'] == month]
        df_month_top = df_month[df_month['number'].isin(top_numbers)]
        if df_month_top.empty:
            continue

        monthly_stats = df_month_top.groupby(['number', 'callType']).size().unstack(fill_value=0)
        for col in ['incoming', 'outgoing']:
            if col not in monthly_stats:
                monthly_stats[col] = 0
        monthly_stats = monthly_stats[['incoming', 'outgoing']]

        scaled_month = StandardScaler().fit_transform(monthly_stats)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_month)

        centers = pd.DataFrame(kmeans.cluster_centers_, columns=['incoming', 'outgoing'])
        sorted_centers = centers.sum(axis=1).sort_values(ascending=False)
        behavior_map = {idx: ['Frequent', 'Seasonal', 'One-Way'][i] for i, idx in enumerate(sorted_centers.index)}

        for number, label in zip(monthly_stats.index, labels):
            behavior_evolution[number][month] = behavior_map[label]

    evolution_df = pd.DataFrame(behavior_evolution).T
    evolution_df = evolution_df[sorted(evolution_df.columns)]
    evolution_df.index = evolution_df.index.astype(str)
    evolution_df.reset_index(inplace=True)
    evolution_df.rename(columns={'index': 'number'}, inplace=True)

    # Initiator shading
    initiator_map = defaultdict(dict)
    for month in all_months:
        df_month = df[df['month'] == month]
        df_month_top = df_month[df_month['number'].isin(top_numbers)]
        if df_month_top.empty:
            continue
        counts = df_month_top.groupby(['number', 'callType']).size().unstack(fill_value=0)
        counts = counts.reindex(columns=['incoming', 'outgoing'], fill_value=0)
        counts['ratio'] = counts['outgoing'] / (counts['incoming'] + counts['outgoing']).replace(0, 1)
        for number in counts.index:
            initiator_map[number][month] = counts.at[number, 'ratio']

    base_colors = {'Frequent': "#2a9d8f", 'Seasonal': "#f4a261", 'One-Way': "#d3d3d3"}
    def adjust_lightness(color_hex, factor):
        rgb = mcolors.to_rgb(color_hex)
        h, l, s = colorsys.rgb_to_hls(*rgb)
        l = max(0, min(1, l + factor))
        return colorsys.hls_to_rgb(h, l, s)

    evo_df = evolution_df.set_index('number')
    color_matrix = []
    for idx, row in evo_df.iterrows():
        row_colors = []
        for col in row.index:
            behavior = row[col]
            base_color = base_colors.get(behavior, "#ffffff")
            ratio = initiator_map.get(idx, {}).get(col, 0.5)
            lightness_factor = (ratio - 0.5) * -0.5
            row_colors.append(adjust_lightness(base_color, lightness_factor))
        color_matrix.append(row_colors)

    # Plot
    fig, ax = plt.subplots(figsize=(len(evo_df.columns) * 0.6, len(evo_df) * 0.4))
    color_array = np.array(color_matrix).reshape(len(color_matrix), len(color_matrix[0]), 3)
    ax.imshow(color_array, aspect='auto')
    # Custom legend box

    legend_elements = [
        Patch(facecolor="#2a9d8f", label='Frequent'),
        Patch(facecolor="#f4a261", label='Seasonal'),
        Patch(facecolor="#d3d3d3", label='One-Way'),
        Patch(facecolor='black', label='Darker = More Outgoing'),
        Patch(facecolor='white', label='Lighter = More Incoming')
    ]

    ax.legend(handles=legend_elements,
              loc='upper left',
              bbox_to_anchor=(1.05, 1.0),
              borderaxespad=0.,
              title='Behavior Legend')

    ax.set_xticks(np.arange(len(evo_df.columns)))
    month_names = [pd.to_datetime(m).strftime('%b %Y') for m in evo_df.columns]
    ax.set_xticklabels(month_names, rotation=45, ha='right')
    # ax.set_xticklabels(evo_df.columns, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(evo_df)))
    # ax.set_yticklabels(evo_df.index)
    masked_labels = ['****' + number[-4:] for number in evo_df.index]
    ax.set_yticklabels(masked_labels)
    ax.set_title("Behavioral Pattern Heatmap (Shaded by Call Initiator)", fontsize=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("Contact Number")
    plt.tight_layout()
    # Return as base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()
    return image_base64
