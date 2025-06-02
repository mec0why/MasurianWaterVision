import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from .constants import SEASON_COLORS, MONTH_NAMES
from ..modeling.forecast import train_water_level_model, predict_future_water_levels


def get_water_level_data(eopatch, max_coverage=1.0, median_threshold_multiplier=0.98):
    dates = np.asarray(eopatch.timestamps)
    water_level_data = eopatch.scalar["WATER_LEVEL"][:, 0]
    median_water_level = np.median(water_level_data) if len(water_level_data) > 0 else 0

    valid_data_mask = (eopatch.scalar["COVERAGE"][:, 0] < max_coverage) & \
                      (water_level_data >= (median_water_level * median_threshold_multiplier))
    valid_indices = np.where(valid_data_mask)[0]

    return dates, valid_data_mask, valid_indices


def plot_water_levels(eopatch, max_coverage=1.0, predict_months=24,
                      median_threshold_multiplier=0.98):
    dates, valid_data_mask, _ = get_water_level_data(eopatch, max_coverage,
                                                     median_threshold_multiplier)
    fig, ax = plt.subplots(figsize=(20, 7))

    hist_dates = dates[valid_data_mask]
    water_levels = eopatch.scalar["WATER_LEVEL"][valid_data_mask, 0]

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    ax.plot(hist_dates, water_levels, "bo-", alpha=0.7, label="Poziom wody (historyczne)", zorder=5)
    ax.plot(hist_dates, eopatch.scalar["COVERAGE"][valid_data_mask, 0], "--", color="gray", alpha=0.5,
            label="Zachmurzenie", zorder=2)

    mean_level = np.mean(water_levels) if len(water_levels) > 0 else 0
    median_level = np.median(water_levels) if len(water_levels) > 0 else 0
    std_dev_level = np.std(water_levels) if len(water_levels) > 0 else 0
    num_measurements = len(water_levels)

    trend_text = "Trend roczny: Brak danych"
    if num_measurements > 2:
        model_lr = LinearRegression()
        dates_num = mdates.date2num(hist_dates).reshape(-1, 1)
        model_lr.fit(dates_num, water_levels)
        trend_per_year = model_lr.coef_[0] * 365
        trend_text = f"Trend roczny: {trend_per_year:+.4f}"

    stats_text = (f"Statystyki historyczne:\n"
                  f"  Średnia: {mean_level:.3f}\n"
                  f"  Mediana: {median_level:.3f}\n"
                  f"  Odch. std: {std_dev_level:.3f}\n"
                  f"  {trend_text}\n"
                  f"  Liczba pomiarów: {num_measurements}")
    ax.text(0.02, 0.05, stats_text, transform=ax.transAxes, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

    if len(water_levels) > 0:
        max_idx_level, min_idx_level = np.argmax(water_levels), np.argmin(water_levels)
        ax.plot(hist_dates[max_idx_level], water_levels[max_idx_level], 'ro', markersize=8, zorder=10)
        ax.annotate(f'Max: {water_levels[max_idx_level]:.2f}', (hist_dates[max_idx_level], water_levels[max_idx_level]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8), zorder=10)

        ax.plot(hist_dates[min_idx_level], water_levels[min_idx_level], 'go', markersize=8, zorder=10)
        ax.annotate(f'Min: {water_levels[min_idx_level]:.2f}', (hist_dates[min_idx_level], water_levels[min_idx_level]),
                    xytext=(10, -15), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8), zorder=10)

    future_dates_pred = []
    if len(hist_dates) > 10 and predict_months > 0:
        print("Trenowanie modelu do predykcji z walidacją...")
        model_rf, scaler_rf = train_water_level_model(hist_dates, water_levels)
        future_dates_pred, predictions_rf = predict_future_water_levels(model_rf, scaler_rf, hist_dates[-1],
                                                                        periods=predict_months)

        bootstrap_predictions = []
        for _ in range(100):
            bootstrap_indices = np.random.choice(np.arange(len(hist_dates)), size=len(hist_dates), replace=True)
            boot_model, boot_scaler = train_water_level_model(np.array(hist_dates)[bootstrap_indices],
                                                              water_levels[bootstrap_indices])
            _, boot_pred = predict_future_water_levels(boot_model, boot_scaler, hist_dates[-1], periods=predict_months)
            bootstrap_predictions.append(boot_pred)

        ax.fill_between(future_dates_pred, np.percentile(bootstrap_predictions, 5, axis=0),
                        np.percentile(bootstrap_predictions, 95, axis=0),
                        color='red', alpha=0.2, label='90% przedział ufności', zorder=3)
        ax.plot(future_dates_pred, predictions_rf, "--", color="red", linewidth=2, alpha=0.8, label="Predykcja AI",
                zorder=6)
        ax.axvline(x=hist_dates[-1], color='dimgray', linestyle='-.', linewidth=1.5, alpha=0.9, zorder=4)
        ax.text(hist_dates[-1] + pd.Timedelta(days=28), 1.05, "Początek predykcji", ha='left', va='bottom', color='red', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='red', alpha=0.8))

        pred_stats_text = (f"Statystyki predykcji ({predict_months} mies.):\n"
                           f"  Średnia: {np.mean(predictions_rf):.3f}\n"
                           f"  Mediana: {np.median(predictions_rf):.3f}\n"
                           f"  Odch. std: {np.std(predictions_rf):.3f}")
        ax.text(0.98, 0.05, pred_stats_text, transform=ax.transAxes, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8), fontsize=9)
        print(
            f"\nPrognoza poziomów wody do {future_dates_pred[-1].strftime('%Y-%m-%d')}.")

    ax.set_ylim(0.0, 1.1)
    ax.set_xlabel("Rok", fontsize=12)
    ax.set_ylabel("Poziom wody (ułamek całkowitego pokrycia) / Zachmurzenie (%)", fontsize=12)
    ax.set_title("Poziom wody w jeziorze - dane historyczne i predykcja AI", fontsize=14)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.02), fontsize=9, ncol=len(handles))

    current_date_marker = pd.Timestamp.now()
    all_dates_for_marker = hist_dates.tolist() + (future_dates_pred.tolist() if len(future_dates_pred) > 0 else [])
    if len(all_dates_for_marker) > 0 and min(all_dates_for_marker) <= current_date_marker <= max(all_dates_for_marker):
        ax.text(current_date_marker - pd.Timedelta(days=42), 0.02, 'Dziś', ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.subplots_adjust(bottom=0.15, top=0.9)

    return ax, valid_data_mask, _


def plot_seasonal_water_levels(eopatch, max_coverage=1.0,
                               median_threshold_multiplier=0.98):
    dates, valid_data_mask, _ = get_water_level_data(eopatch, max_coverage,
                                                     median_threshold_multiplier)
    valid_dates = dates[valid_data_mask]
    water_levels = eopatch.scalar["WATER_LEVEL"][valid_data_mask, 0]

    df = pd.DataFrame({
        'date': valid_dates, 'water_level': water_levels,
        'month': [d.month for d in valid_dates],
        'season': [('Zima' if m in [12, 1, 2] else 'Wiosna' if m in [3, 4, 5] else
        'Lato' if m in [6, 7, 8] else 'Jesień') for m in [d.month for d in valid_dates]]
    })

    season_order = ['Zima', 'Wiosna', 'Lato', 'Jesień']
    month_plot_colors = [SEASON_COLORS[(
        'Zima' if m in [12, 1, 2] else 'Wiosna' if m in [3, 4, 5] else 'Lato' if m in [6, 7, 8] else 'Jesień')] for m in
                         range(1, 13)]

    season_stats = df.groupby('season')['water_level'].agg(['mean', 'count']).reindex(season_order).reset_index()
    season_stats = season_stats.fillna({'mean': 0, 'count': 0})
    season_stats['count'] = season_stats['count'].astype(int)

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.20, top=0.90)

    monthly_avg = df.groupby('month')['water_level'].agg(['mean', 'std', 'count']).reindex(range(1, 13)).reset_index()
    monthly_avg = monthly_avg.fillna({'mean': 0, 'std': 0, 'count': 0})
    monthly_avg['count'] = monthly_avg['count'].astype(int)
    monthly_avg['month_name'] = [MONTH_NAMES[m - 1] for m in monthly_avg['month']]

    ax.bar(monthly_avg['month'], monthly_avg['mean'], yerr=monthly_avg['std'],
           alpha=0.7, capsize=5, color=month_plot_colors)

    for i, row in monthly_avg.iterrows():
        if row['count'] > 0:
            ax.text(row['month'], 0.01, f'n={row["count"]}', ha='center', va='bottom', fontsize=9, color='dimgray',
                    rotation=90, transform=ax.get_xaxis_transform())

    ax.set_xticks(monthly_avg['month'])
    ax.set_xticklabels(monthly_avg['month_name'])
    ax.set_xlabel('Miesiąc', fontsize=14)
    ax.set_ylabel('Poziom wody (ułamek pokrycia)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    if len(df) > 0:
        overall_avg = df['water_level'].mean()
        ax.axhline(y=overall_avg, color='red', linestyle='--', alpha=0.7)
        ax.text(11.5, overall_avg + 0.005, f'Śr. roczna: {overall_avg:.3f}', ha='right', color='red',
                bbox=dict(facecolor='white', alpha=0.7))

        season_text = [f"{s_stat['season']}: {s_stat['mean']:.3f} (n={s_stat['count']})" for _, s_stat in
                       season_stats.iterrows() if s_stat['count'] > 0]
        ax.text(0.98, 0.97, "\n".join(season_text), transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8), fontsize=10)

        non_zero_values = monthly_avg[monthly_avg['mean'] > 0]
        if len(non_zero_values) > 0:
            max_month_stat = non_zero_values.loc[non_zero_values['mean'].idxmax()]
            min_month_stat = non_zero_values.loc[non_zero_values['mean'].idxmin()]

            ax.annotate(f'Max: {max_month_stat["mean"]:.3f}', xy=(max_month_stat["month"], max_month_stat["mean"]),
                        xytext=(0, 25), textcoords="offset points", ha='center', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='green'))

            if max_month_stat["month"] != min_month_stat["month"]:
                ax.annotate(f'Min: {min_month_stat["mean"]:.3f}', xy=(min_month_stat["month"], min_month_stat["mean"]),
                            xytext=(0, -25), textcoords="offset points", ha='center', va='top',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='red'))

            max_val, min_val = non_zero_values['mean'].max(), non_zero_values['mean'].min()
            padding = (max_val - min_val) * 0.3 if max_val > min_val else 0.05
            ax.set_ylim(max(0, min_val - padding), min(1, max_val + padding))

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for season, color in SEASON_COLORS.items() if
               season in season_order]
    fig.legend(handles, [s for s in season_order if s in SEASON_COLORS], loc='lower center', bbox_to_anchor=(0.5, 0.05),
               ncol=4, fontsize=12)
    fig.suptitle('Analiza sezonowa poziomów wody w jeziorze', fontsize=18, y=0.95)

    return fig, ax


def plot_water_level_histogram(eopatch, max_coverage=1.0, bins=20,
                               median_threshold_multiplier=0.98):
    _, valid_data_mask, _ = get_water_level_data(eopatch, max_coverage,
                                                 median_threshold_multiplier)
    water_levels = eopatch.scalar["WATER_LEVEL"][valid_data_mask, 0]

    fig, ax = plt.subplots(figsize=(12, 8))

    if len(water_levels) == 0:
        ax.text(0.5, 0.5, "Brak danych do wyświetlenia histogramu.", horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        return fig, ax

    sns.histplot(water_levels, bins=bins, kde=True, color='steelblue', ax=ax)
    mean_level, median_level, std_level = np.mean(water_levels), np.median(water_levels), np.std(water_levels)

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

    stats_box_text = (f"Statystyki:\nLiczba pomiarów: {len(water_levels)}\nMin: {min(water_levels):.3f}\n"
                      f"Max: {max(water_levels):.3f}\nRozstęp: {max(water_levels) - min(water_levels):.3f}")
    ax.text(0.02, 0.98, stats_box_text, transform=ax.transAxes, verticalalignment='top',
            horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return fig, ax