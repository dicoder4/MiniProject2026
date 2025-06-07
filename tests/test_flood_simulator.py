import pytest
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString
from flood_simulator import create_elevation_grid, DynamicFloodSimulator

@pytest.fixture
def dummy_edges():
    return gpd.GeoDataFrame({
        'geometry': [LineString([(0, 0), (0, 1)]), LineString([(1, 0), (1, 1)])]
    }, crs="EPSG:4326")

def test_create_elevation_grid(dummy_edges):
    elev_gdf = create_elevation_grid(dummy_edges, resolution=10)
    assert not elev_gdf.empty
    assert 'elevation' in elev_gdf.columns

def test_flood_simulator_initialization(dummy_edges):
    elev_gdf = create_elevation_grid(dummy_edges)
    nodes = gpd.GeoDataFrame({'x': [0, 1], 'y': [0, 1]}, geometry=[LineString([(0, 0), (0, 1)]).interpolate(0.5)]*2, crs="EPSG:4326")
    sim = DynamicFloodSimulator(elev_gdf, dummy_edges, nodes, "StationX", 0, 0, initial_people=10)
    assert sim.people_gdf is not None
    assert len(sim.people_gdf) == 10
