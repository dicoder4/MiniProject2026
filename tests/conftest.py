import pytest
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point, Polygon, LineString
import networkx as nx

@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory"""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def mock_flood_polygon():
    """Create a mock flood polygon"""
    return Polygon([
        (73.856, 18.516), 
        (73.857, 18.516),
        (73.857, 18.517), 
        (73.856, 18.517)
    ])

@pytest.fixture
def mock_network():
    """Create a test road network"""
    G = nx.Graph()
    # Add nodes with coordinates
    G.add_node(0, x=73.856, y=18.516)
    G.add_node(1, x=73.857, y=18.517)
    G.add_node(2, x=73.858, y=18.518)
    
    # Add edges with length attributes
    G.add_edge(0, 1, length=100, travel_time=2)
    G.add_edge(1, 2, length=100, travel_time=2)
    return G

@pytest.fixture
def mock_people_gdf():
    """Create test people GeoDataFrame"""
    return gpd.GeoDataFrame(
        {
            'person_id': [1, 2],
            'geometry': [
                Point(73.856, 18.516),
                Point(73.857, 18.517)
            ]
        },
        crs="EPSG:4326"
    )

@pytest.fixture
def mock_safe_centers():
    """Create test evacuation centers"""
    return gpd.GeoDataFrame(
        {
            'center_id': ['H1', 'P1'],
            'center_type': ['hospital', 'police'],
            'name': ['Test Hospital', 'Test Police'],
            'geometry': [
                Point(73.858, 18.518),
                Point(73.859, 18.519)
            ]
        },
        crs="EPSG:4326"
    )