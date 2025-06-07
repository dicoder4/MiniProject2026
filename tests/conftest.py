import pytest
import os
import geopandas as gpd
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory"""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def sample_edges(test_data_dir):
    """Load sample road network edges"""
    return gpd.read_file(test_data_dir / "sample_edges.geojson")

@pytest.fixture
def sample_centers(test_data_dir):
    """Load sample evacuation centers"""
    return gpd.read_file(test_data_dir / "sample_centers.geojson")

@pytest.fixture
def mock_flood_polygon():
    """Create a mock flood polygon"""
    from shapely.geometry import Polygon
    return Polygon([[73.856, 18.516], [73.857, 18.516], 
                   [73.857, 18.517], [73.856, 18.517]])