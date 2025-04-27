import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
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

    cloud_filter = eopatch.scalar["COVERAGE"][:, 0] < max_coverage
    median_level = np.median(eopatch.scalar["WATER_LEVEL"])
    outlier_filter = eopatch.scalar["WATER_LEVEL"][:, 0] >= (median_level * 0.96)

    valid_data = np.logical_and(cloud_filter, outlier_filter)
    valid_indices = np.where(valid_data)[0]

    return dates, valid_data, valid_indices


def train_water_level_model(dates, water_levels):
    X = np.array([[
        np.sin(2 * np.pi * d.timetuple().tm_yday / 365),
        np.cos(2 * np.pi * d.timetuple().tm_yday / 365),
        np.sin(2 * np.pi * d.month / 12),
        np.cos(2 * np.pi * d.month / 12),
        d.year
    ] for d in dates])

    X_train, X_test, y_train, y_test = train_test_split(X, water_levels, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler


def predict_future_water_levels(model, scaler, last_date, periods=24):
    future_dates = pd.date_range(start=last_date, periods=periods, freq='ME')

    X_future = np.array([[
        np.sin(2 * np.pi * d.timetuple().tm_yday / 365),
        np.cos(2 * np.pi * d.timetuple().tm_yday / 365),
        np.sin(2 * np.pi * d.month / 12),
        np.cos(2 * np.pi * d.month / 12),
        d.year
    ] for d in future_dates])

    X_future_scaled = scaler.transform(X_future)
    predictions = np.clip(model.predict(X_future_scaled), 0, 1)
    
    return future_dates, predictions


def plot_water_levels(eopatch, max_coverage=1.0, predict_months=24):
    dates, valid_data, valid_indices = get_water_level_data(eopatch, max_coverage)

    fig, ax = plt.subplots(figsize=(20, 7))

    hist_dates = dates[valid_data]
    water_levels = eopatch.scalar["WATER_LEVEL"][valid_data, 0]

    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    years_fmt = mdates.DateFormatter('%Y')
    months_fmt = mdates.DateFormatter('%b')
    
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)

    if len(hist_dates) > 0:
        min_date = min(hist_dates)
        max_date = max(hist_dates)
        if predict_months > 0 and len(hist_dates) > 10:
            last_pred_date = min_date + pd.DateOffset(months=predict_months)
            if last_pred_date > max_date:
                max_date = last_pred_date

        for year in range(min_date.year, max_date.year + 1):
            summer_start = pd.Timestamp(f"{year}-06-01")
            summer_end = pd.Timestamp(f"{year}-09-01")
            if summer_start < max_date and summer_end > min_date:
                ax.add_patch(Rectangle((mdates.date2num(summer_start), 0), 
                                      mdates.date2num(summer_end) - mdates.date2num(summer_start),
                                      1.1, color='yellow', alpha=0.1, zorder=0))

            winter_start = pd.Timestamp(f"{year}-12-01")
            winter_end = pd.Timestamp(f"{year+1}-03-01")
            if winter_start < max_date and winter_end > min_date:
                ax.add_patch(Rectangle((mdates.date2num(winter_start), 0), 
                                      mdates.date2num(winter_end) - mdates.date2num(winter_start),
                                      1.1, color='lightblue', alpha=0.1, zorder=0))

    hist_line = ax.plot(
        hist_dates,
        water_levels,
        "bo-",
        alpha=0.7,
        label="Poziom wody (historyczne)",
        zorder=5
    )

    cloud_line = ax.plot(
        hist_dates,
        eopatch.scalar["COVERAGE"][valid_data, 0],
        "--",
        color="gray",
        alpha=0.5,
        label="Zachmurzenie",
        zorder=2
    )

    if len(hist_dates) > 2:
        date_nums = mdates.date2num(hist_dates)
        X = date_nums.reshape(-1, 1)

        model = LinearRegression()
        model.fit(X, water_levels)

        trend_line = ax.plot(
            hist_dates,
            model.predict(X),
            '-',
            color='darkblue',
            linewidth=1.5,
            alpha=0.7,
            label=f"Trend (zmiana {model.coef_[0]*365:.4f}/rok)",
            zorder=4
        )

    if len(water_levels) > 0:
        max_idx = np.argmax(water_levels)
        min_idx = np.argmin(water_levels)

        ax.plot(hist_dates[max_idx], water_levels[max_idx], 'ro', markersize=8, zorder=10)
        ax.annotate(f'Max: {water_levels[max_idx]:.2f}',
                   (hist_dates[max_idx], water_levels[max_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
                   zorder=10)

        ax.plot(hist_dates[min_idx], water_levels[min_idx], 'go', markersize=8, zorder=10)
        ax.annotate(f'Min: {water_levels[min_idx]:.2f}',
                   (hist_dates[min_idx], water_levels[min_idx]),
                   xytext=(10, -15), textcoords='offset points',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8),
                   zorder=10)

    mean_level = np.mean(water_levels)
    median_level = np.median(water_levels)
    std_level = np.std(water_levels)

    stats_text = (f"Statystyki historyczne:\n"
                 f"Średnia: {mean_level:.3f}\n"
                 f"Mediana: {median_level:.3f}\n"
                 f"Odch. std: {std_level:.3f}\n"
                 f"Liczba pomiarów: {len(water_levels)}")

    ax.text(0.02, 0.15, stats_text,
            transform=ax.transAxes,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if len(hist_dates) > 10 and predict_months > 0:
        print("Trenowanie modelu do predykcji poziomów wody...")
        model, scaler = train_water_level_model(hist_dates, water_levels)
        future_dates, predictions = predict_future_water_levels(
            model, scaler, hist_dates[-1], periods=predict_months
        )

        n_bootstrap = 100
        bootstrap_predictions = []

        indices = np.arange(len(hist_dates))
        for _ in range(n_bootstrap):
            bootstrap_indices = np.random.choice(indices, size=len(indices), replace=True)
            bootstrap_dates = np.array(hist_dates)[bootstrap_indices]
            bootstrap_levels = water_levels[bootstrap_indices]

            boot_model, boot_scaler = train_water_level_model(bootstrap_dates, bootstrap_levels)
            _, boot_pred = predict_future_water_levels(
                boot_model, boot_scaler, hist_dates[-1], periods=predict_months
            )
            bootstrap_predictions.append(boot_pred)

        bootstrap_predictions = np.array(bootstrap_predictions)
        lower_ci = np.percentile(bootstrap_predictions, 5, axis=0)
        upper_ci = np.percentile(bootstrap_predictions, 95, axis=0)

        ax.fill_between(
            future_dates,
            lower_ci,
            upper_ci,
            color='red',
            alpha=0.2,
            label='90% przedział ufności'
        )

        pred_line = ax.plot(
            future_dates,
            predictions,
            "--",
            color="red",
            linewidth=2,
            alpha=0.8,
            label="Predykcja AI",
            zorder=6
        )

        ax.axvline(x=hist_dates[-1], color='red', linestyle='-', alpha=0.3)
        ax.text(hist_dates[-1], 1.05, "Początek predykcji", ha='center', color='red')

        pred_mean = np.mean(predictions)
        pred_median = np.median(predictions)
        pred_std = np.std(predictions)

        pred_stats_text = (f"Statystyki predykcji:\n"
                          f"Średnia: {pred_mean:.3f}\n"
                          f"Mediana: {pred_median:.3f}\n"
                          f"Odch. std: {pred_std:.3f}")

        ax.text(0.98, 0.15, pred_stats_text,
                transform=ax.transAxes,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))

        print(f"Prognoza poziomów wody na kolejne {predict_months} miesiące (do {future_dates[-1].strftime('%Y-%m-%d')}).")

    ax.axhline(y=mean_level, color='darkgrey', linestyle='--', alpha=0.5)
    ax.text(0.01, mean_level+0.02, f'Średni poziom: {mean_level:.2f}',
            transform=ax.get_yaxis_transform(), color='darkgrey')

    ax.set_ylim(0.0, 1.1)
    ax.set_xlabel("Data", fontsize=12)
    ax.set_ylabel("Poziom wody / Zachmurzenie", fontsize=12)
    ax.set_title("Poziom wody w jeziorze Świętajno - dane historyczne i predykcja AI", fontsize=14)

    ax.grid(True, which='major', axis='both', alpha=0.3)
    ax.grid(True, which='minor', axis='x', alpha=0.15)

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    handles, labels = ax.get_legend_handles_labels()
    season_elements = [
        Rectangle((0, 0), 1, 1, color='yellow', alpha=0.2, label='Lato (Cze-Sie)'),
        Rectangle((0, 0), 1, 1, color='lightblue', alpha=0.2, label='Zima (Gru-Lut)')
    ]

    fig.legend(handles=handles + season_elements, 
              loc='lower center',
              bbox_to_anchor=(0.5, 0.02),
              fontsize=9,
              ncol=len(handles + season_elements))
    
    current_date = pd.Timestamp.now()
    if min(hist_dates) <= current_date <= max(future_dates if 'future_dates' in locals() else hist_dates):
        ax.axvline(x=current_date, color='black', linestyle='-.', alpha=0.5)
        ax.text(current_date, 0.02, 'Dziś', ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
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
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('NDVI')

        ax.set_title(title if title else f"NDVI - {eopatch.timestamps[idx]}")
        ax.axis("off")
        plt.tight_layout()

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

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="3%", pad=0.2)
        cbar = fig.colorbar(im1, cax=cax)
        cbar.set_label('NDVI', fontsize=12)

        if title:
            fig.suptitle(title, fontsize=16, y=0.98)

        plt.subplots_adjust(wspace=0.05, top=0.9)

        return fig, (ax1, ax2)


def plot_water_coverage_heatmap(eopatch, max_coverage=1.0):
    dates, valid_data, _ = get_water_level_data(eopatch, max_coverage)

    valid_dates = dates[valid_data]

    water_masks = eopatch.mask["WATER_MASK"][valid_data, :, :, 0]
    total_pixels = water_masks.shape[1] * water_masks.shape[2]
    water_percentages = np.sum(water_masks, axis=(1, 2)) / total_pixels * 100

    df = pd.DataFrame({
        'date': valid_dates,
        'water_pct': water_percentages,
        'water_level': eopatch.scalar["WATER_LEVEL"][valid_data, 0]
    })

    max_idx = df['water_pct'].idxmax()
    min_idx = df['water_pct'].idxmin()
    max_pct = df.loc[max_idx, 'water_pct']
    min_pct = df.loc[min_idx, 'water_pct']
    max_date = df.loc[max_idx, 'date']
    min_date = df.loc[min_idx, 'date']

    fig, ax_level = plt.subplots(figsize=(18, 7))

    date_nums = mdates.date2num(df['date'])
    ax_level.plot(
        date_nums,
        df['water_level'],
        'b-',
        alpha=0.8,
        linewidth=2.5,
        label='Poziom wody'
    )

    ax_level.set_ylabel('Poziom wody', fontsize=12, color='blue')
    ax_level.tick_params(axis='y', labelcolor='blue')
    ax_level.set_ylim(0, 1)
    ax_level.grid(True, axis='y', alpha=0.3, linestyle='-')

    ax_coverage = ax_level.twinx()

    water_colors = [(0.9, 0.9, 0.9), (0.6, 0.8, 1), (0, 0.4, 0.8)]
    water_cmap = LinearSegmentedColormap.from_list('WaterCoverage', water_colors)

    scaled_sizes = 50 + (df['water_pct'] / df['water_pct'].max() * 150)

    scatter = ax_coverage.scatter(
        date_nums,
        df['water_pct'],
        c=df['water_pct'],
        cmap=water_cmap,
        s=scaled_sizes,
        alpha=0.7,
        edgecolor='darkblue',
        linewidth=0.5,
        label='Pokrywa wodna'
    )

    ax_coverage.set_yticks([])
    ax_coverage.set_yticklabels([])
    ax_coverage.set_ylim(df['water_pct'].min() * 0.95, df['water_pct'].max() * 1.05)

    years = sorted(df['date'].dt.year.unique())
    for year in years[1:]:
        year_start = pd.Timestamp(f"{year}-01-01")
        ax_level.axvline(mdates.date2num(year_start), color='gray', linestyle='--', alpha=0.3)

    date_format = mdates.DateFormatter('%Y-%m')
    ax_level.xaxis.set_major_formatter(date_format)
    ax_level.xaxis.set_major_locator(mdates.YearLocator())
    ax_level.xaxis.set_minor_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    ax_level.set_title('Zmiana poziomu i pokrycia wodnego w czasie', fontsize=14)
    ax_level.set_xlabel('Data', fontsize=12)

    max_x = mdates.date2num(max_date)
    min_x = mdates.date2num(min_date)

    ax_coverage.annotate(
        f'Max: {max_pct:.1f}%',
        xy=(max_x, max_pct),
        xytext=(20, 15),
        textcoords="offset points",
        ha='center',
        va='bottom',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkblue", alpha=0.8),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='darkblue')
    )

    ax_coverage.annotate(
        f'Min: {min_pct:.1f}%',
        xy=(min_x, min_pct),
        xytext=(-20, -15),
        textcoords="offset points",
        ha='center',
        va='top',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkblue", alpha=0.8),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='darkblue')
    )

    cbar = fig.colorbar(scatter, ax=ax_coverage, pad=0.01, fraction=0.025)
    cbar.set_label('Pokrywa wodna (%)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    legend_elements = [
        Line2D([0], [0], color='blue', lw=2.5, label='Poziom wody'),
        Line2D([0], [0], marker='o', color='white', label='Pokrywa wodna',
               markerfacecolor=water_colors[2], markersize=10)
    ]
    ax_level.legend(handles=legend_elements, loc='upper left')

    ax_level.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    return fig, ax_level


def plot_water_mask_comparison(eopatch, idx1, idx2, titles=None):
    mask1 = eopatch.mask["WATER_MASK"][idx1, :, :, 0]
    mask2 = eopatch.mask["WATER_MASK"][idx2, :, :, 0]

    diff_mask = np.zeros_like(mask1, dtype=np.uint8)
    diff_mask[np.logical_and(~mask1, mask2)] = 1
    diff_mask[np.logical_and(mask1, ~mask2)] = 2
    diff_mask[np.logical_and(mask1, mask2)] = 3

    colors = ['#f2f2f2', '#21a9e1', '#e10c0c', '#0c5de1']
    diff_cmap = ListedColormap(colors)

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

    plt.subplots_adjust(wspace=0.02)

    return fig, [ax1, ax2, ax3]


def plot_water_level_histogram(eopatch, max_coverage=1.0, bins=20):
    _, valid_data, _ = get_water_level_data(eopatch, max_coverage)
    water_levels = eopatch.scalar["WATER_LEVEL"][valid_data, 0]

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.histplot(water_levels, bins=bins, kde=True, color='steelblue', ax=ax)

    mean_level = np.mean(water_levels)
    median_level = np.median(water_levels)

    ax.axvline(mean_level, color='red', linestyle='--', linewidth=2, label=f'Średnia: {mean_level:.3f}')
    ax.axvline(median_level, color='green', linestyle='-.', linewidth=2, label=f'Mediana: {median_level:.3f}')

    ax.set_xlabel('Poziom wody')
    ax.set_ylabel('Częstotliwość')
    ax.set_title('Rozkład poziomów wody')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig, ax