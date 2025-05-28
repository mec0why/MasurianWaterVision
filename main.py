import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta
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
    plots_dir = create_directory("plots")
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')


def main():
    create_directory("data")
    create_directory("plots")

    wkt_file_path = "data/lake_boundary.wkt"

    if not os.path.exists(wkt_file_path):
        print(f"Błąd: Plik {wkt_file_path} nie istnieje!")
        print("Umieść plik WKT w folderze data.")
        return

    print("Wczytywanie granic jeziora i konfiguracja przepływu pracy...")
    lake_boundary = load_water_boundary(wkt_file_path)
    lake_bbox = create_bounding_box(lake_boundary, buffer_percentage=0.1)
    lake_gdf = create_geodataframe(lake_boundary)
    lake_gdf.plot()
    save_plot("lake_boundary.png")

    workflow, download_node = create_workflow(lake_gdf, cloud_threshold=0.05)

    current_date = datetime.now()
    ten_years_ago = current_date - timedelta(days=365 * 10)
    time_range = [ten_years_ago.strftime("%Y-%m-%d"), current_date.strftime("%Y-%m-%d")]

    print("Pobieranie i przetwarzanie zdjęć satelitarnych...")
    result = workflow.execute({
        download_node: {
            "bbox": lake_bbox,
            "time_interval": time_range
        },
    })

    patch = result.outputs["final_eopatch"]
    print(f"Znaleziono {len(patch.timestamps)} pasujących zdjęć satelitarnych.")

    plot_rgb_w_water(patch, 0, title=f"Pierwsze zdjęcie - {patch.timestamps[0]}")
    save_plot("first_image.png")

    plot_rgb_w_water(patch, -1, title=f"Ostatnie zdjęcie - {patch.timestamps[-1]}")
    save_plot("last_image.png")

    dates, valid_data, valid_indices = get_water_level_data(patch, 1.0)
    water_levels = patch.scalar["WATER_LEVEL"][valid_data, 0]
    max_idx = valid_indices[np.argmax(water_levels)]
    min_idx = valid_indices[np.argmin(water_levels)]

    plot_rgb_w_water(patch, max_idx, title=f"Najwyższy poziom wody - {patch.timestamps[max_idx]}")
    save_plot("highest_water_level.png")

    plot_rgb_w_water(patch, min_idx, title=f"Najniższy poziom wody - {patch.timestamps[min_idx]}")
    save_plot("lowest_water_level.png")

    plot_water_levels(patch, 1.0)
    save_plot("water_levels_timeline.png")

    plot_seasonal_water_levels(patch, max_coverage=1.0)
    save_plot("seasonal_water_levels.png")

    plot_water_level_histogram(patch, max_coverage=1.0)
    save_plot("water_level_histogram.png")

    plot_water_mask_comparison(
        patch,
        min_idx,
        max_idx,
        titles=[
            f"Najniższy poziom - {patch.timestamps[min_idx]}", 
            f"Najwyższy poziom - {patch.timestamps[max_idx]}", 
            "Porównanie - Najniższy vs Najwyższy poziom wody"
        ]
    )
    save_plot("water_mask_comparison.png")

    plot_ndvi(
        patch,
        min_idx,
        title=f"Porównanie NDVI: Najniższy vs Najwyższy poziom wody",
        compare_idx=max_idx
    )
    save_plot("ndvi_comparison.png")

    print("Wykresy zostały zapisane w folderze plots.")

    plt.show()
    print("Wyświetlanie wizualizacji...")


if __name__ == "__main__":
    main()