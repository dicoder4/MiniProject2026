import pytest
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from evacuation_algorithms import (
    dijkstra_evacuation,
    astar_evacuation,
    generate_evacuation_summary
)

@pytest.fixture
def sample_network():
    G = nx.Graph()
    G.add_edge(0, 1, weight=1, length=100)
    G.add_edge(1, 2, weight=1, length=100)
    return G

@pytest.fixture
def sample_people():
    return gpd.GeoDataFrame(
        geometry=[Point(0, 0)],
        crs="EPSG:4326"
    )

@pytest.fixture
def sample_centers():
    return gpd.GeoDataFrame(
        {
            'center_id': ['C1'],
            'center_type': ['hospital']
        },
        geometry=[Point(1, 1)],
        crs="EPSG:4326"
    )

def test_dijkstra_evacuation(sample_network, sample_people, sample_centers):
    result = dijkstra_evacuation(sample_network, sample_people, sample_centers)
    
    assert 'evacuated' in result
    assert 'unreachable' in result
    assert 'times' in result
    assert 'execution_time' in result

def test_astar_evacuation(sample_network, sample_people, sample_centers):
    result = astar_evacuation(sample_network, sample_people, sample_centers)
    
    assert 'evacuated' in result
    assert 'unreachable' in result
    assert 'times' in result
    assert 'execution_time' in result