import pytest
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
from evacuation_algorithms import dijkstra_evacuation, astar_evacuation

@pytest.fixture
def simple_graph():
    G = nx.MultiDiGraph()
    G.add_node(1, x=0, y=0)
    G.add_node(2, x=1, y=1)
    G.add_edge(1, 2, length=100)
    return G

@pytest.fixture
def people():
    return gpd.GeoDataFrame({
        'person_id': [1],
        'geometry': [Point(0, 0)]
    }, crs="EPSG:4326")

@pytest.fixture
def safe_centers():
    return gpd.GeoDataFrame({
        'center_id': ['center1'],
        'geometry': [Point(1, 1)],
        'type': ['hospital']
    }, crs="EPSG:4326")

def test_dijkstra_evacuation(simple_graph, people, safe_centers):
    result = dijkstra_evacuation(simple_graph, people, safe_centers)
    assert 'routes' in result
    assert len(result['routes']) == 1
    assert result['evacuated']

def test_astar_evacuation(simple_graph, people, safe_centers):
    result = astar_evacuation(simple_graph, people, safe_centers)
    assert 'routes' in result
    assert len(result['routes']) == 1
