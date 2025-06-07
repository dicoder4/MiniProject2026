import pytest
from evacuation_algorithms import (
    dijkstra_evacuation,
    astar_evacuation,
    quanta_adaptive_routing_evacuation,
    bidirectional_evacuation
)

@pytest.mark.evacuation
def test_dijkstra_evacuation(mock_network, mock_people_gdf, mock_safe_centers):
    """Test Dijkstra evacuation algorithm"""
    result = dijkstra_evacuation(mock_network, mock_people_gdf, mock_safe_centers)
    
    assert result is not None
    assert 'evacuated' in result
    assert 'routes' in result
    assert 'times' in result
    assert 'execution_time' in result
    assert result['algorithm'] == 'Dijkstra'
    
    # Check if routes are valid
    for route in result['routes']:
        assert 'person_id' in route
        assert 'path' in route
        assert 'time' in route
        assert route['time'] > 0

@pytest.mark.evacuation
def test_astar_evacuation(mock_network, mock_people_gdf, mock_safe_centers):
    """Test A* evacuation algorithm"""
    result = astar_evacuation(mock_network, mock_people_gdf, mock_safe_centers)
    
    assert result is not None
    assert 'evacuated' in result
    assert 'routes' in result
    assert 'times' in result
    assert result['algorithm'] == 'A*'