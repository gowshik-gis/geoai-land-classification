"""
Data Loader Module
==================
Handles loading and basic validation of satellite imagery data.
All data sources are FREE — no API keys required.
"""

import numpy as np
import pandas as pd
from pathlib import Path

class SatelliteDataLoader:
    """
    Loads satellite imagery from free sources.

    Supported Sources (FREE):
    - Sentinel-2 (ESA) - https://scihub.copernicus.eu
    - Landsat-9 (NASA/USGS) - https://earthexplorer.usgs.gov
    - MODIS (NASA) - https://modis.gsfc.nasa.gov
    - OpenStreetMap - https://www.openstreetmap.org
    """

    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_sample_data(self, region="amazon"):
        """
        Generate simulated satellite data for demonstration.
        In production, this would load actual satellite imagery.

        Parameters:
        -----------
        region : str
            Region name (amazon, sahara, ganges, europe, australia)

        Returns:
        --------
        dict : Contains 'rgb', 'nir', 'swir' bands and metadata
        """
        np.random.seed(42)

        # Simulate a 100x100 pixel satellite image
        size = (100, 100)

        data = {
            'rgb': np.random.randint(0, 255, (*size, 3), dtype=np.uint8),
            'nir': np.random.randint(0, 255, size, dtype=np.uint8),
            'swir': np.random.randint(0, 255, size, dtype=np.uint8),
            'metadata': {
                'region': region,
                'resolution': 30,  # meters
                'crs': 'EPSG:4326',
                'bands': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
                'source': 'Sentinel-2 (Simulated)',
                'date': '2024-01-01'
            }
        }

        return data

    def calculate_ndvi(self, red_band, nir_band):
        """
        Calculate Normalized Difference Vegetation Index.

        NDVI = (NIR - Red) / (NIR + Red)

        Range: -1 to 1
        - Negative: Water, clouds, snow
        - 0 to 0.2: Barren rock, sand, urban
        - 0.2 to 0.5: Sparse vegetation
        - 0.5 to 0.8: Moderate vegetation
        - 0.8 to 1.0: Dense vegetation

        Parameters:
        -----------
        red_band : np.ndarray
            Red band reflectance (0-255)
        nir_band : np.ndarray
            Near-infrared band reflectance (0-255)

        Returns:
        --------
        np.ndarray : NDVI values (-1 to 1)
        """
        # Convert to float and normalize
        red = red_band.astype(float) / 255.0
        nir = nir_band.astype(float) / 255.0

        # Avoid division by zero
        denominator = nir + red
        denominator[denominator == 0] = 0.001

        ndvi = (nir - red) / denominator
        return np.clip(ndvi, -1, 1)

    def calculate_ndwi(self, green_band, nir_band):
        """
        Calculate Normalized Difference Water Index.

        NDWI = (Green - NIR) / (Green + NIR)

        Used to detect water bodies.
        """
        green = green_band.astype(float) / 255.0
        nir = nir_band.astype(float) / 255.0

        denominator = green + nir
        denominator[denominator == 0] = 0.001

        ndwi = (green - nir) / denominator
        return np.clip(ndwi, -1, 1)

    def get_data_info(self):
        """Return information about available free data sources."""
        sources = {
            'Sentinel-2': {
                'provider': 'European Space Agency (ESA)',
                'resolution': '10m (RGB), 20m (NIR/SWIR), 60m (atmospheric)',
                'revisit': '5 days',
                'bands': 13,
                'cost': 'FREE',
                'url': 'https://scihub.copernicus.eu'
            },
            'Landsat-9': {
                'provider': 'NASA / USGS',
                'resolution': '30m (multispectral), 100m (thermal)',
                'revisit': '16 days',
                'bands': 11,
                'cost': 'FREE',
                'url': 'https://earthexplorer.usgs.gov'
            },
            'MODIS': {
                'provider': 'NASA',
                'resolution': '250m - 1000m',
                'revisit': 'Daily',
                'bands': 36,
                'cost': 'FREE',
                'url': 'https://modis.gsfc.nasa.gov'
            },
            'OpenStreetMap': {
                'provider': 'OpenStreetMap Contributors',
                'resolution': 'Variable (vector data)',
                'revisit': 'Continuous updates',
                'bands': 'N/A (vector)',
                'cost': 'FREE',
                'url': 'https://www.openstreetmap.org'
            }
        }
        return sources


if __name__ == "__main__":
    # Test the data loader
    loader = SatelliteDataLoader()
    data = loader.load_sample_data("amazon")
    print(f"Loaded data shape: {data['rgb'].shape}")
    print(f"Metadata: {data['metadata']}")

    ndvi = loader.calculate_ndvi(data['rgb'][:,:,0], data['nir'])
    print(f"NDVI range: {ndvi.min():.2f} to {ndvi.max():.2f}")

    sources = loader.get_data_info()
    print("\nAvailable FREE data sources:")
    for name, info in sources.items():
        print(f"  {name}: {info['cost']} | {info['resolution']}")
