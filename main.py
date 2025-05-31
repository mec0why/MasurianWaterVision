import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from src.data_preparation import load_water_boundary, create_bounding_box, create_geodataframe
from src.water_detection import create_workflow
from src.visualization import (
    plot_rgb_w_water, get_water_level_data, plot_water_levels,
    plot_ndvi, plot_seasonal_water_levels, plot_water_mask_comparison, plot_water_level_histogram
)


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


def save_plot(filename):
    plots_dir_from_env = os.getenv("PLOTS_FOLDER", "plots")
    actual_plots_dir = create_directory(plots_dir_from_env)
    plt.savefig(os.path.join(actual_plots_dir, filename), dpi=300, bbox_inches='tight')


def main():
    load_dotenv()

    data_dir_env = os.getenv("DATA_FOLDER", "data")
    create_directory(data_dir_env)

    wkt_file_path = os.getenv("WKT_FILE_PATH")
    buffer_percentage_env = float(os.getenv("BOUNDARY_BUFFER_PERCENTAGE", 0.1))
    cloud_threshold_env = float(os.getenv("CLOUD_THRESHOLD", 0.05))
    ice_threshold_env = float(os.getenv("ICE_THRESHOLD", 0.6))
    max_cloud_coverage_download_env = float(os.getenv("MAX_CLOUD_COVERAGE", 0.8))
    resolution_env = int(os.getenv("SATELLITE_IMAGE_RESOLUTION", 10))
    time_range_years_env = int(os.getenv("TIME_RANGE_YEARS", 10))
    predict_months_env = int(os.getenv("PREDICTION_MONTHS", 24))
    water_level_median_threshold_env = float(os.getenv("WATER_LEVEL_MEDIAN_THRESHOLD", 0.96))

    print("Używane parametry zmiennych środowiskowych:")
    print(f"  WKT_FILE_PATH: {wkt_file_path}")
    print(f"  BOUNDARY_BUFFER_PERCENTAGE: {buffer_percentage_env}")
    print(f"  CLOUD_THRESHOLD: {cloud_threshold_env}")
    print(f"  ICE_THRESHOLD: {ice_threshold_env}")
    print(f"  MAX_CLOUD_COVERAGE: {max_cloud_coverage_download_env}")
    print(f"  SATELLITE_IMAGE_RESOLUTION: {resolution_env}")
    print(f"  TIME_RANGE_YEARS: {time_range_years_env}")
    print(f"  PREDICTION_MONTHS: {predict_months_env}")
    print(f"  WATER_LEVEL_MEDIAN_THRESHOLD: {water_level_median_threshold_env}")
    print("-" * 50)

    if not wkt_file_path or not os.path.exists(wkt_file_path):
        print(f"Błąd: Plik WKT nie został znaleziony pod ścieżką: {wkt_file_path}.")
        print("Upewnij się, że zmienna WKT_FILE_PATH jest poprawnie ustawiona w pliku .env i wskazuje na istniejący plik .wkt.")
        return

    print("Wczytywanie granic jeziora i konfiguracja przepływu pracy...")
    lake_boundary = load_water_boundary(wkt_file_path)
    lake_bbox = create_bounding_box(lake_boundary, buffer_percentage=buffer_percentage_env)
    lake_gdf = create_geodataframe(lake_boundary)
    lake_gdf.plot()
    save_plot("lake_boundary.png")

    workflow, download_node = create_workflow(
        lake_gdf,
        cloud_threshold=cloud_threshold_env,
        ice_threshold=ice_threshold_env,
        max_cloud_coverage_download=max_cloud_coverage_download_env,
        resolution=resolution_env
    )

    current_date = datetime.now()
    ten_years_ago = current_date - timedelta(days=365 * time_range_years_env)
    time_range = [ten_years_ago.strftime("%Y-%m-%d"), current_date.strftime("%Y-%m-%d")]

    print("Rozpoczynanie pobierania i przetwarzania zdjęć satelitarnych...")
    result = workflow.execute({
        download_node: {
            "bbox": lake_bbox,
            "time_interval": time_range
        },
    })

    patch = result.outputs["final_eopatch"]
    print(f"Pobieranie i przetwarzanie zakończone. Znaleziono {len(patch.timestamps)} pasujących zdjęć satelitarnych.")

    if len(patch.timestamps) == 0:
        print("Nie znaleziono żadnych zdjęć dla podanego zakresu i obszaru. Zakończono działanie programu.")
        return

    print("Generowanie wykresów wizualizacyjnych...")
    plot_rgb_w_water(patch, 0, title=f"Pierwsze zdjęcie - {patch.timestamps[0]}")
    save_plot("first_image.png")
    print(f"Zapisano wykres: first_image.png - Pierwsze zdjęcie z dnia {patch.timestamps[0].strftime('%Y-%m-%d')}.")

    plot_rgb_w_water(patch, -1, title=f"Ostatnie zdjęcie - {patch.timestamps[-1]}")
    save_plot("last_image.png")
    print(f"Zapisano wykres: last_image.png - Ostatnie zdjęcie z dnia {patch.timestamps[-1].strftime('%Y-%m-%d')}.")

    dates, valid_data, valid_indices = get_water_level_data(patch, median_threshold_multiplier=water_level_median_threshold_env)

    if len(valid_indices) == 0:
        print("Brak wystarczających danych do wygenerowania pozostałych wykresów po zastosowaniu filtrów (np. lód).")
        print("Możliwe przyczyny: wszystkie zdjęcia zostały odrzucone przez filtr lodu, lub inne kryteria filtracji uniemożliwiające analizę.")
        if len(patch.scalar["WATER_LEVEL"]) > 0:
            plot_water_levels(patch, predict_months=predict_months_env, median_threshold_multiplier=water_level_median_threshold_env)
            save_plot("water_levels_timeline.png")
            print(f"Zapisano wykres: water_levels_timeline.png - Oś czasu poziomu wody z predykcją AI.")
        plt.show()
        return

    water_levels = patch.scalar["WATER_LEVEL"][valid_data, 0]

    if len(water_levels) == 0:
        print("Brak danych o poziomach wody po filtracji. Nie można wygenerować wykresów przedstawiających minima i maksima.")
        plt.show()
        return

    max_idx_local = np.argmax(water_levels)
    min_idx_local = np.argmin(water_levels)
    max_idx = valid_indices[max_idx_local]
    min_idx = valid_indices[min_idx_local]

    plot_rgb_w_water(patch, max_idx, title=f"Najwyższy poziom wody - {patch.timestamps[max_idx]}")
    save_plot("highest_water_level.png")
    print(f"Zapisano wykres: highest_water_level.png - Zdjęcie z najwyższym poziomem wody z dnia {patch.timestamps[max_idx].strftime('%Y-%m-%d')}.")

    plot_rgb_w_water(patch, min_idx, title=f"Najniższy poziom wody - {patch.timestamps[min_idx]}")
    save_plot("lowest_water_level.png")
    print(f"Zapisano wykres: lowest_water_level.png - Zdjęcie z najniższym poziomem wody z dnia {patch.timestamps[min_idx].strftime('%Y-%m-%d')}.")

    plot_seasonal_water_levels(patch, median_threshold_multiplier=water_level_median_threshold_env)
    save_plot("seasonal_water_levels.png")
    print(f"Zapisano wykres: seasonal_water_levels.png - Sezonowa analiza poziomów wody.")

    plot_water_level_histogram(patch, median_threshold_multiplier=water_level_median_threshold_env)
    save_plot("water_level_histogram.png")
    print(f"Zapisano wykres: water_level_histogram.png - Histogram poziomów wody.")

    plot_water_mask_comparison(
        patch,
        min_idx,
        max_idx,
        titles=[
            f"Najniższy poziom - {patch.timestamps[min_idx].strftime('%Y-%m-%d')}",
            f"Najwyższy poziom - {patch.timestamps[max_idx].strftime('%Y-%m-%d')}",
            "Porównanie - Najniższy vs Najwyższy poziom wody"
        ]
    )
    save_plot("water_mask_comparison.png")
    print(f"Zapisano wykres: water_mask_comparison.png - Porównanie masek wodnych z dnia {patch.timestamps[min_idx].strftime('%Y-%m-%d')} i {patch.timestamps[max_idx].strftime('%Y-%m-%d')}.")

    plot_ndvi(
        patch,
        min_idx,
        title=f"Porównanie NDVI: Najniższy vs Najwyższy poziom wody",
        compare_idx=max_idx
    )
    save_plot("ndvi_comparison.png")
    print(f"Zapisano wykres: ndvi_comparison.png - Porównanie NDVI z dnia {patch.timestamps[min_idx].strftime('%Y-%m-%d')} i {patch.timestamps[max_idx].strftime('%Y-%m-%d')}.")

    plot_water_levels(patch, predict_months=predict_months_env, median_threshold_multiplier=water_level_median_threshold_env)
    save_plot("water_levels_timeline.png")
    print(f"Zapisano wykres: water_levels_timeline.png - Oś czasu poziomu wody z predykcją AI.")

    print("Wszystkie wykresy zostały zapisane w folderze 'plots'.")
    print("Wyświetlanie wizualizacji. Zamknij okna, aby zakończyć program.")
    plt.show()


if __name__ == "__main__":
    main()