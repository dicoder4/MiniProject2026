import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from flood_simulator import DynamicFloodSimulator, create_elevation_grid

def test_create_elevation_grid():
    # Create mock edges GeoDataFrame
    geometry = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    edges = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")
    
    # Test grid creation
    elev_gdf = create_elevation_grid(edges, resolution=10)
    
    assert isinstance(elev_gdf, gpd.GeoDataFrame)
    assert 'elevation' in elev_gdf.columns
    assert len(elev_gdf) == 100  # 10x10 grid

def test_flood_simulator_initialization():
    # Create mock data
    elev_gdf = gpd.GeoDataFrame(
        {'elevation': [0]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326"
    )
    edges = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:4326"
    )
    nodes = gpd.GeoDataFrame(
        geometry=[Point(0, 0)],
        crs="EPSG:4326"
    )
    
    simulator = DynamicFloodSimulator(
        elev_gdf=elev_gdf,
        edges=edges,
        nodes=nodes,
        station="Test Station",
        lat=0,
        lon=0,
        initial_people=10
    )
    
    assert simulator.people_gdf is not None
    assert len(simulator.people_gdf) == 10