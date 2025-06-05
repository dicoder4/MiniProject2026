"""
Dijkstra's Algorithm for Flood Evacuation Routing
================================================

This module implements Dijkstra's shortest path algorithm for evacuating people
from flooded areas to safe centers (hospitals, police stations, etc.).

Author: Flood Simulation Team
Date: 2025
"""

import networkx as nx
import osmnx as ox
import time
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import numpy as np


def prepare_safe_centers(hospitals_gdf, police_gdf, edges, flood_gdf=None):
    """
    Prepare safe evacuation centers from OSM data (hospitals + police stations).
    
    Parameters:
    -----------
    hospitals_gdf : GeoDataFrame
        Hospital locations from OSM
    police_gdf : GeoDataFrame  
        Police station locations from OSM
    edges : GeoDataFrame
        Road network edges
    flood_gdf : GeoDataFrame, optional
        Current flood boundaries to exclude flooded centers
    
    Returns:
    --------
    GeoDataFrame: Safe centers with center_id and geometry columns
    """
    safe_centers = []
    
    # Collect hospitals
    if hospitals_gdf is not None and not hospitals_gdf.empty:
        for _, row in hospitals_gdf.iterrows():
            safe_centers.append({
                'name': row.get('name', 'Hospital'),
                'geometry': row.geometry,
                'type': 'hospital'
            })
    
    # Collect police stations
    if police_gdf is not None and not police_gdf.empty:
        for _, row in police_gdf.iterrows():
            safe_centers.append({
                'name': row.get('name', 'Police Station'),
                'geometry': row.geometry,
                'type': 'police'
            })
    
    # Fallback to mock centers if no real centers found
    if not safe_centers:
        print("⚠️ No hospitals or police stations found. Creating mock evacuation centers.")
        sample_lines = edges.sample(min(5, len(edges))).geometry
        for i, line in enumerate(sample_lines):
            if isinstance(line, LineString):
                safe_centers.append({
                    'name': f'Mock Center {i+1}',
                    'geometry': line.interpolate(0.5, normalized=True),
                    'type': 'mock'
                })
    
    # Create GeoDataFrame
    safe_centers_gdf = gpd.GeoDataFrame(safe_centers, crs=edges.crs).reset_index(drop=True)
    safe_centers_gdf['center_id'] = safe_centers_gdf['name'].fillna("Unnamed")
    
    # Convert geometries to Point (use centroid if not Point)
    safe_centers_gdf['geometry'] = safe_centers_gdf.geometry.apply(
        lambda g: g if isinstance(g, Point) else g.centroid
    )
    
    # Exclude centers inside flood zone
    if flood_gdf is not None and not flood_gdf.empty and not safe_centers_gdf.empty:
        safe_centers_gdf = safe_centers_gdf[
            ~safe_centers_gdf.geometry.apply(lambda g: flood_gdf.intersects(g).any())
        ].copy()
    
    if safe_centers_gdf.empty:
        print("⚠️ All safe centers are flooded. No evacuation centers available.")
    
    return safe_centers_gdf


def prepare_graph_for_routing(G, walking_speed_kmph=10):
    """
    Prepare the road network graph by adding travel time weights to edges.
    
    Parameters:
    -----------
    G : NetworkX Graph
        Road network graph from OSMnx
    walking_speed_kmph : float
        Walking speed in kilometers per hour
    
    Returns:
    --------
    NetworkX Graph: Graph with travel_time weights on edges
    """
    walking_speed_mpm = walking_speed_kmph * 1000 / 60  # meters per minute
    
    # Add travel time to each edge
    for u, v, k, data in G.edges(keys=True, data=True):
        if 'length' in data:
            data['travel_time'] = data['length'] / walking_speed_mpm
        else:
            # Default length if not available
            data['travel_time'] = 5.0  # 5 minutes default
    
    return G


def evacuate_people_with_dijkstra(G, flooded_people, safe_centers_gdf, walking_speed_kmph=10):
    """
    Evacuate people using Dijkstra's shortest path algorithm.
    
    Parameters:
    -----------
    G : NetworkX Graph
        Road network graph
    flooded_people : GeoDataFrame
        People in flooded areas with person_id and geometry
    safe_centers_gdf : GeoDataFrame
        Safe evacuation centers
    walking_speed_kmph : float
        Walking speed in kilometers per hour
    
    Returns:
    --------
    dict: Results containing evacuated people, routes, times, unreachable, and log
    """
    # Prepare graph with travel times
    G = prepare_graph_for_routing(G, walking_speed_kmph)
    
    # Initialize results
    evacuated_people = []
    evac_routes = []
    evac_times = []
    unreachable = []
    evac_log = []
    
    start_time = time.time()
    
    print(f"🚶 Starting evacuation for {len(flooded_people)} people using Dijkstra's algorithm...")
    
    # Process each person
    for idx, row in flooded_people.iterrows():
        person_id = row.get('person_id', f'Person_{idx}')
        person_point = row.geometry
        
        try:
            # Find nearest node to person's location
            orig_node = ox.distance.nearest_nodes(G, person_point.x, person_point.y)
        except Exception as e:
            unreachable.append(person_point)
            evac_log.append(f"❌ Person {person_id}: Could not find nearest node ({str(e)})")
            continue
        
        # Find best evacuation center
        best_route = None
        best_time = float('inf')
        best_center_id = None
        best_dest_node = None
        
        for _, center_row in safe_centers_gdf.iterrows():
            dest_point = center_row.geometry
            center_id = center_row.center_id
            
            try:
                # Find nearest node to evacuation center
                dest_node = ox.distance.nearest_nodes(G, dest_point.x, dest_point.y)
                
                # Calculate shortest path using Dijkstra's algorithm
                path = nx.shortest_path(G, orig_node, dest_node, weight='travel_time')
                
                # Calculate total travel time
                total_time = 0
                for u, v in zip(path[:-1], path[1:]):
                    # Handle multi-edge case
                    edge_data = G[u][v]
                    if isinstance(edge_data, dict):
                        if 'travel_time' in edge_data:
                            total_time += edge_data['travel_time']
                        else:
                            # Take first edge if multiple
                            first_edge = list(edge_data.values())[0]
                            total_time += first_edge.get('travel_time', 5.0)
                    else:
                        total_time += edge_data.get('travel_time', 5.0)
                
                # Update best route if this is faster
                if total_time < best_time:
                    best_route = path
                    best_time = total_time
                    best_center_id = center_id
                    best_dest_node = dest_node
                    
            except nx.NetworkXNoPath:
                continue
            except Exception as e:
                continue
        
        # Record results
        if best_route is not None:
            evac_routes.append({
                'person_id': person_id,
                'origin': person_point,
                'path': best_route,
                'time': best_time,
                'destination': best_center_id
            })
            evac_times.append(best_time)
            evacuated_people.append(person_point)
            evac_log.append(f"✅ Person {person_id}: Evacuated to '{best_center_id}' in {best_time:.2f} min")
        else:
            unreachable.append(person_point)
            evac_log.append(f"❌ Person {person_id}: No evacuation path found")
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Summary statistics
    total_people = len(flooded_people)
    evacuated_count = len(evacuated_people)
    unreachable_count = len(unreachable)
    success_rate = (evacuated_count / total_people * 100) if total_people > 0 else 0
    
    print(f"\n📊 Dijkstra's Algorithm Results:")
    print(f"   Total people: {total_people}")
    print(f"   ✅ Evacuated: {evacuated_count}")
    print(f"   ❌ Unreachable: {unreachable_count}")
    print(f"   📈 Success rate: {success_rate:.1f}%")
    
    if evac_times:
        print(f"   ⏱️ Evacuation times:")
        print(f"      Average: {np.mean(evac_times):.2f} min")
        print(f"      Maximum: {np.max(evac_times):.2f} min")
        print(f"      Minimum: {np.min(evac_times):.2f} min")
    
    print(f"   🕒 Algorithm execution time: {execution_time:.2f} seconds")
    
    return {
        'algorithm': 'Dijkstra',
        'evacuated': evacuated_people,
        'routes': evac_routes,
        'times': evac_times,
        'unreachable': unreachable,
        'log': evac_log,
        'execution_time': execution_time,
        'stats': {
            'total_people': total_people,
            'evacuated_count': evacuated_count,
            'unreachable_count': unreachable_count,
            'success_rate': success_rate,
            'avg_time': np.mean(evac_times) if evac_times else 0,
            'max_time': np.max(evac_times) if evac_times else 0,
            'min_time': np.min(evac_times) if evac_times else 0
        }
    }


def get_route_coordinates(G, route_path):
    """
    Convert a route path (list of node IDs) to coordinates for plotting.
    
    Parameters:
    -----------
    G : NetworkX Graph
        Road network graph
    route_path : list
        List of node IDs forming the route
    
    Returns:
    --------
    list: List of (longitude, latitude) coordinate tuples
    """
    coordinates = []
    for node_id in route_path:
        if node_id in G.nodes:
            node_data = G.nodes[node_id]
            coordinates.append((node_data['x'], node_data['y']))
    return coordinates


# Example usage and testing
if __name__ == "__main__":
    print("🔄 Dijkstra Evacuation Algorithm Module")
    print("This module provides evacuation routing using Dijkstra's shortest path algorithm.")
    print("Import this module in your main application to use the evacuation functions.")
