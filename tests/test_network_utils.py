import pytest
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, Polygon
from network_utils import (
    validate_safe_centers_against_flood,
    prepare_safe_centers,
    setup_graph_for_evacuation
)

@pytest.fixture
def sample_flood_polygon():
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

@pytest.fixture
def sample_centers():
    return gpd.GeoDataFrame(
        {
            'center_id': ['H1', 'H2'],
            'center_type': ['hospital', 'hospital']
        },
        geometry=[Point(2, 2), Point(0.5, 0.5)],  # One safe, one in flood zone
        crs="EPSG:4326"
    )

def test_validate_safe_centers(sample_centers, sample_flood_polygon):
    edges = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])],
        crs="EPSG:4326"
    )
    
    safe_centers = validate_safe_centers_against_flood(
        sample_centers,
        sample_flood_polygon,
        edges
    )
    
    assert len(safe_centers) == 1  # Only one center should be safe
    assert safe_centers.iloc[0]['center_id'] == 'H1'

def test_setup_graph_for_evacuation():
    G = nx.Graph()
    G.add_edge(0, 1, length=1000)
    
    G_setup = setup_graph_for_evacuation(G, walking_speed_kmph=5)
    
    assert 'travel_time' in G_setup.edges[0, 1]
    assert 'weight' in G_setup.edges[0, 1]
    