import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.filters import sobel
from skimage.morphology import closing, disk
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def plot_rgb_w_water(eopatch, idx, title=None):
    ratio = np.abs(eopatch.bbox.max_x - eopatch.bbox.min_x) / np.abs(eopatch.bbox.max_y - eopatch.bbox.min_y)
    fig, ax = plt.subplots(figsize=(ratio * 10, 10))

    ax.imshow(np.clip(2.5 * eopatch.data["BANDS"][idx][:, :, [2, 1, 0]], 0, 1))

    observed = closing(eopatch.mask["WATER_MASK"][idx, :, :, 0], disk(1))
    nominal = sobel(eopatch.mask_timeless["NOMINAL_WATER"][:, :, 0])
    observed = sobel(observed)

    nominal = np.ma.masked_where(~nominal.astype(bool), nominal)
    observed = np.ma.masked_where(~observed.astype(bool), observed)

    ax.imshow(nominal, cmap=plt.cm.Reds)
    ax.imshow(observed, cmap=plt.cm.Blues)

    if title:
        ax.set_title(title)

    ax.axis("off")
    plt.tight_layout()

    return fig, ax


def get_water_level_data(eopatch, max_coverage=1.0):
    dates = np.asarray(eopatch.timestamps)
    valid_data = (eopatch.scalar["COVERAGE"][:, 0] < max_coverage) & (
                eopatch.scalar["WATER_LEVEL"][:, 0] >= (np.median(eopatch.scalar["WATER_LEVEL"]) * 0.96))
    valid_indices = np.where(valid_data)[0]

    return dates, valid_data, valid_indices


def train_water_level_model(dates, water_levels):
    X = np.array([
        [np.sin(2 * np.pi * d.timetuple().tm_yday / 365), np.cos(2 * np.pi * d.timetuple().tm_yday / 365),
         np.sin(2 * np.pi * d.month / 12), np.cos(2 * np.pi * d.month / 12), d.year]
        for d in dates
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, water_levels, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler


def predict_future_water_levels(model, scaler, last_date, periods=24):
    future_dates = pd.date_range(start=last_date, periods=periods, freq='ME')

    X_future = np.array([
        [np.sin(2 * np.pi * d.timetuple().tm_yday / 365), np.cos(2 * np.pi * d.timetuple().tm_yday / 365),
         np.sin(2 * np.pi * d.month / 12), np.cos(2 * np.pi * d.month / 12), d.year]
        for d in future_dates
    ])

    X_future_scaled = scaler.transform(X_future)
    predictions = np.clip(model.predict(X_future_scaled), 0, 1)

    return future_dates, predictions


def plot_water_levels(eopatch, max_coverage=1.0, predict_months=24):
    dates, valid_data, valid_indices = get_water_level_data(eopatch, max_coverage)
    fig, ax = plt.subplots(figsize=(20, 7))

    hist_dates = dates[valid_data]
    water_levels = eopatch.scalar["WATER_LEVEL"][valid_data, 0]

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    if len(hist_dates) > 0:
        min_date = min(hist_dates)
        max_date = max(hist_dates)
        if predict_months > 0 and len(hist_dates) > 10:
            last_pred_date = min_date + pd.DateOffset(months=predict_months)
            if last_pred_date > max_date:
                max_date = last_pred_date

        for year in range(min_date.year, max_date.year + 1):
            ax.add_patch(Rectangle((mdates.date2num(pd.Timestamp(f"{year}-06-01")), 0),
                                   mdates.date2num(pd.Timestamp(f"{year}-09-01")) - mdates.date2num(
                                       pd.Timestamp(f"{year}-06-01")),
                                   1.1, color='yellow', alpha=0.1, zorder=0))

            ax.add_patch(Rectangle((mdates.date2num(pd.Timestamp(f"{year}-12-01")), 0),
                                   mdates.date2num(pd.Timestamp(f"{year + 1}-03-01")) - mdates.date2num(
                                       pd.Timestamp(f"{year}-12-01")),
                                   1.1, color='lightblue', alpha=0.1, zorder=0))

    ax.plot(hist_dates, water_levels, "bo-", alpha=0.7, label="Poziom wody (historyczne)", zorder=5)
    ax.plot(hist_dates, eopatch.scalar["COVERAGE"][valid_data, 0], "--", color="gray", alpha=0.5, label="Zachmurzenie",
            zorder=2)

    if len(hist_dates) > 2:
        model = LinearRegression()
        model.fit(mdates.date2num(hist_dates).reshape(-1, 1), water_levels)
        ax.plot(hist_dates, model.predict(mdates.date2num(hist_dates).reshape(-1, 1)), '-', color='darkblue',
                linewidth=1.5, alpha=0.7, label=f"Trend (zmiana {model.coef_[0] * 365:.4f}/rok)", zorder=4)

    if len(water_levels) > 0:
        max_idx = np.argmax(water_levels)
        min_idx = np.argmin(water_levels)

        ax.plot(hist_dates[max_idx], water_levels[max_idx], 'ro', markersize=8, zorder=10)
        ax.annotate(f'Max: {water_levels[max_idx]:.2f}', (hist_dates[max_idx], water_levels[max_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8), zorder=10)

        ax.plot(hist_dates[min_idx], water_levels[min_idx], 'go', markersize=8, zorder=10)
        ax.annotate(f'Min: {water_levels[min_idx]:.2f}', (hist_dates[min_idx], water_levels[min_idx]),
                    xytext=(10, -15), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8), zorder=10)

    mean_level = np.mean(water_levels)
    stats_text = (f"Statystyki historyczne:\nŚrednia: {mean_level:.3f}\n"
                  f"Mediana: {np.median(water_levels):.3f}\n"
                  f"Odch. std: {np.std(water_levels):.3f}\n"
                  f"Liczba pomiarów: {len(water_levels)}")
    ax.text(0.02, 0.15, stats_text, transform=ax.transAxes, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if len(hist_dates) > 10 and predict_months > 0:
        print("Trenowanie modelu do predykcji poziomów wody...")

        model, scaler = train_water_level_model(hist_dates, water_levels)
        future_dates, predictions = predict_future_water_levels(model, scaler, hist_dates[-1], periods=predict_months)

        bootstrap_predictions = []
        for _ in range(100):
            bootstrap_indices = np.random.choice(np.arange(len(hist_dates)), size=len(hist_dates), replace=True)
            boot_model, boot_scaler = train_water_level_model(np.array(hist_dates)[bootstrap_indices],
                                                              water_levels[bootstrap_indices])
            _, boot_pred = predict_future_water_levels(boot_model, boot_scaler, hist_dates[-1], periods=predict_months)
            bootstrap_predictions.append(boot_pred)

        ax.fill_between(future_dates, np.percentile(bootstrap_predictions, 5, axis=0),
                        np.percentile(bootstrap_predictions, 95, axis=0),
                        color='red', alpha=0.2, label='90% przedział ufności')

        ax.plot(future_dates, predictions, "--", color="red", linewidth=2, alpha=0.8, label="Predykcja AI", zorder=6)
        ax.axvline(x=hist_dates[-1], color='red', linestyle='-', alpha=0.3)
        ax.text(hist_dates[-1], 1.05, "Początek predykcji", ha='center', color='red')

        pred_stats_text = (f"Statystyki predykcji:\nŚrednia: {np.mean(predictions):.3f}\n"
                           f"Mediana: {np.median(predictions):.3f}\n"
                           f"Odch. std: {np.std(predictions):.3f}")
        ax.text(0.98, 0.15, pred_stats_text, transform=ax.transAxes, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))

    print(f"Prognoza poziomów wody na kolejne {predict_months} miesiące (do {future_dates[-1].strftime('%Y-%m-%d')}).")

    ax.axhline(y=mean_level, color='darkgrey', linestyle='--', alpha=0.5)
    ax.text(0.01, mean_level + 0.02, f'Średni poziom: {mean_level:.2f}', transform=ax.get_yaxis_transform(),
            color='darkgrey')

    ax.set_ylim(0.0, 1.1)
    ax.set_xlabel("Rok", fontsize=12)
    ax.set_ylabel("Poziom wody (ułamek całkowitego pokrycia) / Zachmurzenie (%)", fontsize=12)
    ax.set_title("Poziom wody w jeziorze - dane historyczne i predykcja AI", fontsize=14)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    season_elements = [
        Rectangle((0, 0), 1, 1, color='yellow', alpha=0.2, label='Lato (Cze-Sie)'),
        Rectangle((0, 0), 1, 1, color='lightblue', alpha=0.2, label='Zima (Gru-Lut)')
    ]

    fig.legend(handles=handles + season_elements, loc='lower center',
               bbox_to_anchor=(0.5, 0.02), fontsize=9, ncol=len(handles + season_elements))

    current_date = pd.Timestamp.now()
    if min(hist_dates) <= current_date <= max(future_dates if 'future_dates' in locals() else hist_dates):
        ax.axvline(x=current_date, color='black', linestyle='-.', alpha=0.5)
        ax.text(current_date, 0.02, 'Dziś', ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.subplots_adjust(bottom=0.15)

    return ax, valid_data, valid_indices


def plot_ndvi(eopatch, idx, title=None, compare_idx=None):
    red = eopatch.data["BANDS"][idx, :, :, 2]
    nir = eopatch.data["BANDS"][idx, :, :, 3]
    ndvi = np.zeros_like(red)
    valid_mask = (nir + red) > 0
    ndvi[valid_mask] = (nir[valid_mask] - red[valid_mask]) / (nir[valid_mask] + red[valid_mask])

    ratio = np.abs(eopatch.bbox.max_x - eopatch.bbox.min_x) / np.abs(eopatch.bbox.max_y - eopatch.bbox.min_y)
    ndvi_cmap = ListedColormap(['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850'])

    if compare_idx is None:
        fig, ax = plt.subplots(figsize=(ratio * 10, 10))
        im = ax.imshow(ndvi, cmap=ndvi_cmap, vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax).set_label('NDVI (indeks -1 do 1)')
        ax.set_title(title if title else f"NDVI - {eopatch.timestamps[idx]}")
        ax.axis("off")
        return fig, ax
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ratio * 20, 10))

        red_compare = eopatch.data["BANDS"][compare_idx, :, :, 2]
        nir_compare = eopatch.data["BANDS"][compare_idx, :, :, 3]
        ndvi_compare = np.zeros_like(red_compare)
        valid_mask_compare = (nir_compare + red_compare) > 0
        ndvi_compare[valid_mask_compare] = (nir_compare[valid_mask_compare] - red_compare[valid_mask_compare]) / (
                    nir_compare[valid_mask_compare] + red_compare[valid_mask_compare])

        im1 = ax1.imshow(ndvi, cmap=ndvi_cmap, vmin=-1, vmax=1)
        ax1.set_title(f"NDVI - {eopatch.timestamps[idx]}")
        ax1.axis("off")

        im2 = ax2.imshow(ndvi_compare, cmap=ndvi_cmap, vmin=-1, vmax=1)
        ax2.set_title(f"NDVI - {eopatch.timestamps[compare_idx]}")
        ax2.axis("off")

        cax = make_axes_locatable(ax2).append_axes("right", size="3%", pad=0.2)
        cbar = fig.colorbar(im1, cax=cax).set_label('NDVI (indeks -1 do 1)', fontsize=12)

        if title:
            fig.suptitle(title, fontsize=16, y=0.98)

        plt.subplots_adjust(wspace=0.05, top=0.9)
        return fig, (ax1, ax2)


def plot_seasonal_water_levels(eopatch, max_coverage=1.0):
    dates, valid_data, _ = get_water_level_data(eopatch, max_coverage)
    valid_dates = dates[valid_data]
    water_levels = eopatch.scalar["WATER_LEVEL"][valid_data, 0]

    df = pd.DataFrame({
        'date': valid_dates,
        'water_level': water_levels,
        'month': [d.month for d in valid_dates],
        'season': [('Zima' if m in [12, 1, 2] else
                    'Wiosna' if m in [3, 4, 5] else
                    'Lato' if m in [6, 7, 8] else 'Jesień')
                   for m in [d.month for d in valid_dates]]
    })

    season_order = ['Zima', 'Wiosna', 'Lato', 'Jesień']
    season_colors = {'Zima': 'lightblue', 'Wiosna': 'springgreen',
                     'Lato': 'gold', 'Jesień': 'orange'}
    month_colors = ['lightblue'] * 3 + ['springgreen'] * 3 + ['gold'] * 3 + ['orange'] * 3
    month_colors = [month_colors[m - 1] for m in range(1, 13)]

    season_stats = df.groupby('season')['water_level'].agg(['mean', 'count']).reset_index()
    season_stats = pd.merge(
        pd.DataFrame({'season': season_order}),
        season_stats,
        on='season',
        how='left'
    ).fillna({'mean': 0, 'count': 0})
    season_stats['count'] = season_stats['count'].astype(int)

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.20, top=0.90)

    monthly_avg = df.groupby('month')['water_level'].agg(['mean', 'std', 'count']).reset_index()
    all_months = pd.DataFrame({'month': range(1, 13)})
    monthly_avg = pd.merge(all_months, monthly_avg, on='month', how='left').fillna({'mean': 0, 'std': 0, 'count': 0})
    monthly_avg['count'] = monthly_avg['count'].astype(int)

    month_names = ['Sty', 'Lut', 'Mar', 'Kwi', 'Maj', 'Cze', 'Lip', 'Sie', 'Wrz', 'Paź', 'Lis', 'Gru']
    monthly_avg['month_name'] = [month_names[m - 1] for m in monthly_avg['month']]

    ax.bar(monthly_avg['month'], monthly_avg['mean'], yerr=monthly_avg['std'],
           alpha=0.7, capsize=5, color=month_colors)

    for i, row in monthly_avg.iterrows():
        ax.text(row['month'], 0.01, f'n={row["count"]}',
                ha='center', va='bottom', fontsize=9,
                color='dimgray', rotation=90)

    ax.set_xticks(monthly_avg['month'])
    ax.set_xticklabels(monthly_avg['month_name'])
    ax.set_xlabel('Miesiąc', fontsize=14)
    ax.set_ylabel('Poziom wody (ułamek pokrycia)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    if len(df) > 0:
        overall_avg = df['water_level'].mean()
        ax.axhline(y=overall_avg, color='red', linestyle='--', alpha=0.7)
        ax.text(11.5, overall_avg + 0.005, f'Śr. roczna: {overall_avg:.3f}',
                ha='right', color='red', bbox=dict(facecolor='white', alpha=0.7))

        season_text = []
        for season, stats in season_stats.iterrows():
            if stats['count'] > 0:
                season_text.append(f"{stats['season']}: {stats['mean']:.3f} (n={stats['count']})")

        ax.text(0.98, 0.97, "\n".join(season_text),
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                fontsize=10)

        non_zero_values = monthly_avg[monthly_avg['mean'] > 0]
        if len(non_zero_values) > 0:
            max_idx = non_zero_values['mean'].idxmax()
            min_idx = non_zero_values['mean'].idxmin()

            max_month = monthly_avg.iloc[max_idx]
            min_month = monthly_avg.iloc[min_idx]

            ax.annotate(f'Max: {max_month["mean"]:.3f}',
                        xy=(max_month["month"], max_month["mean"]),
                        xytext=(0, 25), textcoords="offset points",
                        ha='center', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='green'))

            if max_idx != min_idx:
                ax.annotate(f'Min: {min_month["mean"]:.3f}',
                            xy=(min_month["month"], min_month["mean"]),
                            xytext=(0, -25), textcoords="offset points",
                            ha='center', va='top',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='red'))

            max_val = non_zero_values['mean'].max()
            min_val = non_zero_values['mean'].min()
            padding = (max_val - min_val) * 0.3 if max_val > min_val else 0.05
            ax.set_ylim(max(0, min_val - padding), min(1, max_val + padding))

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in season_colors.values()]
    fig.legend(handles, season_colors.keys(), loc='lower center',
               bbox_to_anchor=(0.5, 0.05), ncol=4, fontsize=12)

    fig.suptitle('Analiza sezonowa poziomów wody w jeziorze', fontsize=18, y=0.95)

    return fig, ax


def plot_water_mask_comparison(eopatch, idx1, idx2, titles=None):
    mask1 = eopatch.mask["WATER_MASK"][idx1, :, :, 0]
    mask2 = eopatch.mask["WATER_MASK"][idx2, :, :, 0]

    diff_mask = np.zeros_like(mask1, dtype=np.uint8)
    diff_mask[np.logical_and(~mask1, mask2)] = 1
    diff_mask[np.logical_and(mask1, ~mask2)] = 2
    diff_mask[np.logical_and(mask1, mask2)] = 3

    diff_cmap = ListedColormap(['#f2f2f2', '#21a9e1', '#e10c0c', '#0c5de1'])

    if titles is None:
        title1 = f"Maska wodna - {eopatch.timestamps[idx1]}"
        title2 = f"Maska wodna - {eopatch.timestamps[idx2]}"
        title_diff = f"Porównanie masek wodnych\n{eopatch.timestamps[idx1]} vs {eopatch.timestamps[idx2]}"
    else:
        title1, title2, title_diff = titles

    ratio = np.abs(eopatch.bbox.max_x - eopatch.bbox.min_x) / np.abs(eopatch.bbox.max_y - eopatch.bbox.min_y)
    fig = plt.figure(figsize=(ratio * 20, 8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.2, 0.1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    cbar_ax = fig.add_subplot(gs[0, 3])

    ax1.imshow(mask1, cmap='Blues')
    ax1.set_title(title1, fontsize=12)
    ax1.axis('off')

    ax2.imshow(mask2, cmap='Blues')
    ax2.set_title(title2, fontsize=12)
    ax2.axis('off')

    im = ax3.imshow(diff_mask, cmap=diff_cmap, vmin=0, vmax=3)
    ax3.set_title(title_diff, fontsize=14)
    ax3.axis('off')

    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0.375, 1.125, 1.875, 2.625])
    cbar.set_ticklabels(['Brak wody', 'Nowa woda', 'Zanikła woda', 'Stała woda'])

    return fig, [ax1, ax2, ax3]


def plot_water_level_histogram(eopatch, max_coverage=1.0, bins=20):
    _, valid_data, _ = get_water_level_data(eopatch, max_coverage)
    water_levels = eopatch.scalar["WATER_LEVEL"][valid_data, 0]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(water_levels, bins=bins, kde=True, color='steelblue', ax=ax)

    mean_level = np.mean(water_levels)
    median_level = np.median(water_levels)
    std_level = np.std(water_levels)

    ax.axvline(mean_level, color='red', linestyle='--', linewidth=2, label=f'Średnia: {mean_level:.3f}')
    ax.axvline(median_level, color='green', linestyle='-.', linewidth=2, label=f'Mediana: {median_level:.3f}')

    handles, labels = ax.get_legend_handles_labels()
    handles.extend([
        Patch(color='steelblue', alpha=0.5, label=f'Histogram (n={len(water_levels)})'),
        Line2D([0], [0], color='steelblue', lw=2, label=f'KDE (odch. std={std_level:.3f})')
    ])

    ax.legend(handles=handles)
    ax.set_xlabel('Poziom wody (ułamek całkowitego pokrycia)')
    ax.set_ylabel('Częstotliwość (liczba obserwacji)')
    ax.set_title('Rozkład poziomów wody w jeziorze')
    ax.grid(True, alpha=0.3)

    ax.text(0.02, 0.98,
            f"Statystyki:\nLiczba pomiarów: {len(water_levels)}\nMin: {min(water_levels):.3f}\n"
            f"Max: {max(water_levels):.3f}\nRozstęp: {max(water_levels) - min(water_levels):.3f}",
            transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return fig, ax