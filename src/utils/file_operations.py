import os
import matplotlib.pyplot as plt
from config import settings


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


def save_plot(filename):
    plots_dir_from_env = settings.PLOTS_FOLDER
    actual_plots_dir = create_directory(plots_dir_from_env)
    plt.savefig(os.path.join(actual_plots_dir, filename), dpi=300, bbox_inches='tight')