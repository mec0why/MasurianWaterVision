import matplotlib.pyplot as plt
import os
from src.data_preparation import load_water_boundary, create_bounding_box, create_geodataframe
from src.water_detection import create_workflow
from src.visualization import plot_rgb_w_water, plot_water_levels
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
    plot_water_levels(patch, 1.0)
    plt.show()


if __name__ == "__main__":
    main()