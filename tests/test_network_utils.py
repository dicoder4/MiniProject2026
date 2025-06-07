import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import Point, LineString
from network_utils import prepare_safe_centers

@pytest.fixture
def mock_edges():
    return gpd.GeoDataFrame({
        'geometry': [LineString([(0, 0), (0, 1)]), LineString([(1, 1), (1, 2)])]
    }, crs="EPSG:4326")

@pytest.fixture
def mock_hospitals():
    return gpd.GeoDataFrame({
        'name': ['Test Hospital'],
        'geometry': [Point(0.01, 0.01)]
    }, crs="EPSG:4326")

def test_prepare_safe_centers(mock_hospitals, mock_edges):
    centers = prepare_safe_centers(mock_hospitals, None, mock_edges, flood_poly=None)
    assert not centers.empty
    assert 'center_id' in centers.columns
