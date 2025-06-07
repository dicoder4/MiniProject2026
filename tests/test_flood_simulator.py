import pytest
from flood_simulator import DynamicFloodSimulator, create_elevation_grid
import geopandas as gpd
from shapely.geometry import Point, Polygon

def test_create_elevation_grid():
    """Test elevation grid creation"""
    edges = gpd.GeoDataFrame(
        geometry=[LineString([
            (73.856, 18.516),
            (73.857, 18.517)
        ])],
        crs="EPSG:4326"
    )
    
    elev_gdf = create_elevation_grid(edges, resolution=10)
    
    assert isinstance(elev_gdf, gpd.GeoDataFrame)
    assert 'elevation' in elev_gdf.columns
    assert len(elev_gdf) == 100  # 10x10 grid

def test_flood_simulator_initialization(mock_network):
    """Test flood simulator initialization"""
    # Create test data
    elev_gdf = gpd.GeoDataFrame(
        {'elevation': [0, 1]},
        geometry=[
            Point(73.856, 18.516),
            Point(73.857, 18.517)
        ],
        crs="EPSG:4326"
    )
    
    edges = gpd.GeoDataFrame(
        geometry=[LineString([
            (73.856, 18.516),
            (73.857, 18.517)
        ])],
        crs="EPSG:4326"
    )
    
    nodes = gpd.GeoDataFrame(
        geometry=[
            Point(73.856, 18.516),
            Point(73.857, 18.517)
        ],
        crs="EPSG:4326"
    )
    
    simulator = DynamicFloodSimulator(
        elev_gdf=elev_gdf,
        edges=edges,
        nodes=nodes,
        station="Test Station",
        lat=18.516,
        lon=73.856,
        initial_people=10
    )
    
    assert simulator.people_gdf is not None
    assert len(simulator.people_gdf) == 10