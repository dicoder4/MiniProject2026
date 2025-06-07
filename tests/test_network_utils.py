import pytest
from network_utils import (
    validate_safe_centers_against_flood,
    prepare_safe_centers,
    setup_graph_for_evacuation
)

def test_validate_safe_centers(mock_safe_centers, mock_flood_polygon):
    """Test safe center validation against flood zones"""
    edges = gpd.GeoDataFrame(
        geometry=[LineString([
            (73.856, 18.516),
            (73.857, 18.517)
        ])],
        crs="EPSG:4326"
    )
    
    safe_centers = validate_safe_centers_against_flood(
        mock_safe_centers,
        mock_flood_polygon,
        edges
    )
    
    # Both centers should be safe (outside flood zone)
    assert len(safe_centers) == 2
    assert set(safe_centers['center_id'].tolist()) == {'H1', 'P1'}

def test_setup_graph_for_evacuation(mock_network):
    """Test graph preparation for evacuation"""
    G_setup = setup_graph_for_evacuation(mock_network, walking_speed_kmph=5)
    
    # Check edge attributes
    for u, v, data in G_setup.edges(data=True):
        assert 'travel_time' in data
        assert 'weight' in data
        assert 'base_cost' in data
        assert 'penalty' in data
        assert data['penalty'] >= 0