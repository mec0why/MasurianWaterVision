import matplotlib.pyplot as plt
import numpy as np
import os
from src.data_preparation import load_water_boundary, create_bounding_box, create_geodataframe
from src.water_detection import create_workflow
from src.visualization import (
    plot_rgb_w_water, get_water_level_data, plot_water_levels, 
    plot_ndvi, plot_water_coverage_heatmap, plot_water_mask_comparison, plot_water_level_histogram
)
from datetime import datetime, timedelta


def main():
    if not os.path.exists("data"):
        os.makedirs("data")

    wkt_file_path = "data/swietajno_nominal.wkt"

    if not os.path.exists(wkt_file_path):
        print(f"Błąd: Plik {wkt_file_path} nie istnieje!")
        print("Umieść plik WKT w folderze data.")
        return

    print("Wczytywanie granic jeziora i konfiguracja przepływu pracy...")
    dam_nominal = load_water_boundary(wkt_file_path)
    dam_bbox = create_bounding_box(dam_nominal, buffer_percentage=0.1)
    dam_gdf = create_geodataframe(dam_nominal)
    dam_gdf.plot()
    workflow, download_node = create_workflow(dam_gdf, cloud_threshold=0.05)

    current_date = datetime.now()
    ten_years_ago = current_date - timedelta(days=365*10)
    time_interval = [ten_years_ago.strftime("%Y-%m-%d"), current_date.strftime("%Y-%m-%d")]

    print("Pobieranie i przetwarzanie zdjęć satelitarnych...")
    result = workflow.execute({
        download_node: {
            "bbox": dam_bbox,
            "time_interval": time_interval
        },
    })

    patch = result.outputs["final_eopatch"]
    print(f"Znaleziono {len(patch.timestamps)} pasujących zdjęć satelitarnych.")

    print("Generowanie i wyświetlanie wizualizacji...")
    plot_rgb_w_water(patch, 0, title=f"Pierwsze zdjęcie - {patch.timestamps[0]}")
    plot_rgb_w_water(patch, -1, title=f"Ostatnie zdjęcie - {patch.timestamps[-1]}")

    dates, valid_data, valid_indices = get_water_level_data(patch, 1.0)
    water_levels = patch.scalar["WATER_LEVEL"][valid_data, 0]
    max_idx = valid_indices[np.argmax(water_levels)]
    min_idx = valid_indices[np.argmin(water_levels)]
    
    plot_rgb_w_water(patch, max_idx, title=f"Najwyższy poziom wody - {patch.timestamps[max_idx]}")
    plot_rgb_w_water(patch, min_idx, title=f"Najniższy poziom wody - {patch.timestamps[min_idx]}")

    plot_water_levels(patch, 1.0)
    
    plot_water_coverage_heatmap(patch, max_coverage=1.0)
    
    plot_water_level_histogram(patch, max_coverage=1.0)
    
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
    
    plot_ndvi(
        patch, 
        min_idx, 
        title=f"Porównanie NDVI: Najniższy vs Najwyższy poziom wody",
        compare_idx=max_idx
    )
    
    plt.show()


if __name__ == "__main__":
    main()