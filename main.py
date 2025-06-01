import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta
from config import settings
from src.data_handling.geo_preparation import load_water_boundary, create_bounding_box, create_geodataframe
from src.eolearn.pipeline import create_workflow
from src.visualization.image_display import plot_rgb_w_water, plot_ndvi
from src.visualization.stats_display import get_water_level_data, plot_water_levels, plot_seasonal_water_levels, plot_water_level_histogram
from src.visualization.comparison_display import plot_water_mask_comparison
from src.utils.file_operations import create_directory, save_plot


def main():
    create_directory(settings.DATA_FOLDER)

    print("Używane parametry zmiennych środowiskowych:")
    print(f"  WKT_FILE_PATH: {settings.WKT_FILE_PATH}")
    print(f"  BOUNDARY_BUFFER_PERCENTAGE: {settings.BOUNDARY_BUFFER_PERCENTAGE}")
    print(f"  CLOUD_THRESHOLD: {settings.CLOUD_THRESHOLD}")
    print(f"  ICE_THRESHOLD: {settings.ICE_THRESHOLD}")
    print(f"  MAX_CLOUD_COVERAGE_DOWNLOAD: {settings.MAX_CLOUD_COVERAGE_DOWNLOAD}")
    print(f"  SATELLITE_IMAGE_RESOLUTION: {settings.SATELLITE_IMAGE_RESOLUTION}")
    print(f"  TIME_RANGE_YEARS: {settings.TIME_RANGE_YEARS}")
    print(f"  PREDICTION_MONTHS: {settings.PREDICTION_MONTHS}")
    print(f"  WATER_LEVEL_MEDIAN_THRESHOLD: {settings.WATER_LEVEL_MEDIAN_THRESHOLD}")
    print("-" * 50)

    if not settings.WKT_FILE_PATH or not os.path.exists(settings.WKT_FILE_PATH):
        print(f"Błąd: Plik WKT nie został znaleziony pod ścieżką: {settings.WKT_FILE_PATH}.")
        print("Upewnij się, że zmienna WKT_FILE_PATH jest poprawnie ustawiona w pliku .env i wskazuje na istniejący plik .wkt.")
        return

    print("Wczytywanie granic jeziora i konfiguracja przepływu pracy...")
    lake_boundary = load_water_boundary(settings.WKT_FILE_PATH)
    lake_bbox = create_bounding_box(lake_boundary, buffer_percentage=settings.BOUNDARY_BUFFER_PERCENTAGE)
    lake_gdf = create_geodataframe(lake_boundary)
    lake_gdf.plot()
    save_plot("lake_boundary.png")

    workflow, download_node = create_workflow(
        lake_gdf,
        cloud_threshold=settings.CLOUD_THRESHOLD,
        ice_threshold=settings.ICE_THRESHOLD,
        max_cloud_coverage_download=settings.MAX_CLOUD_COVERAGE_DOWNLOAD,
        resolution=settings.SATELLITE_IMAGE_RESOLUTION
    )

    current_date = datetime.now()
    ten_years_ago = current_date - timedelta(days=365 * settings.TIME_RANGE_YEARS)
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

    dates, valid_data, valid_indices = get_water_level_data(patch, median_threshold_multiplier=settings.WATER_LEVEL_MEDIAN_THRESHOLD)

    if len(valid_indices) == 0:
        print("Brak wystarczających danych do wygenerowania pozostałych wykresów po zastosowaniu filtrów (np. lód).")
        print("Możliwe przyczyny: wszystkie zdjęcia zostały odrzucone przez filtr lodu, lub inne kryteria filtracji uniemożliwiające analizę.")
        if len(patch.scalar["WATER_LEVEL"]) > 0:
            plot_water_levels(patch, predict_months=settings.PREDICTION_MONTHS, median_threshold_multiplier=settings.WATER_LEVEL_MEDIAN_THRESHOLD)
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

    plot_seasonal_water_levels(patch, median_threshold_multiplier=settings.WATER_LEVEL_MEDIAN_THRESHOLD)
    save_plot("seasonal_water_levels.png")
    print(f"Zapisano wykres: seasonal_water_levels.png - Sezonowa analiza poziomów wody.")

    plot_water_level_histogram(patch, median_threshold_multiplier=settings.WATER_LEVEL_MEDIAN_THRESHOLD)
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

    plot_water_levels(patch, predict_months=settings.PREDICTION_MONTHS, median_threshold_multiplier=settings.WATER_LEVEL_MEDIAN_THRESHOLD)
    save_plot("water_levels_timeline.png")
    print(f"Zapisano wykres: water_levels_timeline.png - Oś czasu poziomu wody z predykcją AI.")

    print("Wszystkie wykresy zostały zapisane w folderze 'plots'.")
    print("Wyświetlanie wizualizacji. Zamknij okna, aby zakończyć program.")
    plt.show()


if __name__ == "__main__":
    main()