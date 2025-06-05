"""
Evacuation Algorithms Module for Flood Evacuation System
Enhanced with detailed logging, center assignment tracking, and downloadable logs
"""

import networkx as nx
import osmnx as ox
import time
import numpy as np
from shapely.geometry import Point
from haversine import haversine
import logging
from collections import deque, defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_nearest_node_robust(G, point, max_distance=1000):
    """Enhanced nearest node finding with multiple fallback strategies"""
    try:
        # Strategy 1: Direct OSMnx nearest_nodes
        node = ox.distance.nearest_nodes(G, point.x, point.y)
        return node
    except:
        try:
            # Strategy 2: Manual search with distance threshold
            min_dist = float('inf')
            nearest_node = None
            px, py = point.x, point.y
            
            for node, data in G.nodes(data=True):
                if 'x' in data and 'y' in data:
                    nx_coord, ny_coord = data['x'], data['y']
                    # Use Euclidean distance for speed
                    dist = ((px - nx_coord)**2 + (py - ny_coord)**2)**0.5
                    if dist < min_dist and dist < max_distance:
                        min_dist = dist
                        nearest_node = node
            
            if nearest_node is not None:
                return nearest_node
            else:
                raise Exception(f"No nodes within {max_distance} units")
        except:
            # Strategy 3: Find any node (last resort)
            nodes_list = list(G.nodes())
            if nodes_list:
                return nodes_list[0]  # Return first available node
            else:
                raise Exception("No nodes in graph")

def generate_detailed_evacuation_log(evacuation_result, safe_centers_gdf, location_name, algorithm_name):
    """Generate comprehensive evacuation log with center-wise statistics"""
    
    # Initialize center statistics
    center_stats = defaultdict(lambda: {
        'count': 0,
        'people_ids': [],
        'avg_time': 0,
        'min_time': float('inf'),
        'max_time': 0,
        'total_time': 0,
        'center_type': 'unknown',
        'center_name': 'Unknown'
    })
    
    # Populate center information
    for _, center in safe_centers_gdf.iterrows():
        center_id = center.get('center_id', 'Unknown')
        center_stats[center_id]['center_type'] = center.get('type', 'unknown')
        center_stats[center_id]['center_name'] = center.get('name', center_id)
    
    # Process evacuation routes
    for route in evacuation_result['routes']:
        destination = route.get('destination', 'Unknown')
        person_id = route.get('person_id', 'Unknown')
        time_taken = route.get('time', 0)
        
        if destination in center_stats:
            center_stats[destination]['count'] += 1
            center_stats[destination]['people_ids'].append(person_id)
            center_stats[destination]['total_time'] += time_taken
            center_stats[destination]['min_time'] = min(center_stats[destination]['min_time'], time_taken)
            center_stats[destination]['max_time'] = max(center_stats[destination]['max_time'], time_taken)
            center_stats[destination]['avg_time'] = center_stats[destination]['total_time'] / center_stats[destination]['count']
    
    # Generate detailed log
    log_lines = []
    log_lines.append("="*80)
    log_lines.append("DETAILED EVACUATION LOG")
    log_lines.append("="*80)
    log_lines.append(f"Location: {location_name}")
    log_lines.append(f"Algorithm Used: {algorithm_name}")
    log_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Execution Time: {evacuation_result['execution_time']:.2f} seconds")
    log_lines.append("")
    
    # Overall statistics
    total_people_in_danger = len(evacuation_result['evacuated']) + len(evacuation_result['unreachable'])
    evacuated_count = len(evacuation_result['evacuated'])
    unreachable_count = len(evacuation_result['unreachable'])
    success_rate = (evacuated_count / total_people_in_danger * 100) if total_people_in_danger > 0 else 0
    
    log_lines.append("EVACUATION SUMMARY")
    log_lines.append("-" * 40)
    log_lines.append(f"Total People in Danger: {total_people_in_danger}")
    log_lines.append(f"Successfully Evacuated: {evacuated_count}")
    log_lines.append(f"Unreachable: {unreachable_count}")
    log_lines.append(f"Success Rate: {success_rate:.1f}%")
    log_lines.append("")
    
    # Center-wise evacuation statistics
    log_lines.append("CENTER-WISE EVACUATION STATISTICS")
    log_lines.append("-" * 50)
    
    # Sort centers by evacuation count (descending)
    sorted_centers = sorted(center_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    for center_id, stats in sorted_centers:
        if stats['count'] > 0:
            log_lines.append(f"\n🏥 {center_id} ({stats['center_type'].upper()})")
            log_lines.append(f"   Center Name: {stats['center_name']}")
            log_lines.append(f"   People Evacuated: {stats['count']}")
            log_lines.append(f"   Average Time: {stats['avg_time']:.1f} minutes")
            log_lines.append(f"   Min Time: {stats['min_time']:.1f} minutes")
            log_lines.append(f"   Max Time: {stats['max_time']:.1f} minutes")
            log_lines.append(f"   People IDs: {', '.join(map(str, stats['people_ids']))}")
    
    # Centers with no evacuations
    unused_centers = [center_id for center_id, stats in center_stats.items() if stats['count'] == 0]
    if unused_centers:
        log_lines.append(f"\nUNUSED EVACUATION CENTERS: {', '.join(unused_centers)}")
    
    log_lines.append("")
    log_lines.append("DETAILED EVACUATION LOG ENTRIES")
    log_lines.append("-" * 50)
    
    # Add individual evacuation log entries
    for entry in evacuation_result['log']:
        log_lines.append(entry)
    
    # Time distribution analysis
    if evacuation_result['times']:
        log_lines.append("")
        log_lines.append("EVACUATION TIME ANALYSIS")
        log_lines.append("-" * 40)
        times = evacuation_result['times']
        log_lines.append(f"Minimum Time: {min(times):.1f} minutes")
        log_lines.append(f"Maximum Time: {max(times):.1f} minutes")
        log_lines.append(f"Average Time: {np.mean(times):.1f} minutes")
        log_lines.append(f"Median Time: {np.median(times):.1f} minutes")
        log_lines.append(f"Standard Deviation: {np.std(times):.1f} minutes")
    
    log_lines.append("")
    log_lines.append("="*80)
    log_lines.append("END OF EVACUATION LOG")
    log_lines.append("="*80)
    
    return "\n".join(log_lines), center_stats

def generate_evacuation_summary(evacuation_result, safe_centers_gdf):
    """Generate detailed summary of evacuations per center"""
    center_assignments = {}
    
    # Initialize all centers with zero count
    for _, center in safe_centers_gdf.iterrows():
        center_id = center.get('center_id', 'Unknown')
        center_assignments[center_id] = {
            'count': 0,
            'people_ids': [],
            'avg_time': 0,
            'center_type': center.get('type', 'unknown'),
            'center_name': center.get('name', center_id)
        }
    
    # Count evacuations per center
    for route in evacuation_result['routes']:
        destination = route.get('destination', 'Unknown')
        person_id = route.get('person_id', 'Unknown')
        time_taken = route.get('time', 0)
        
        if destination in center_assignments:
            center_assignments[destination]['count'] += 1
            center_assignments[destination]['people_ids'].append(person_id)
            
            # Update average time
            current_times = [r.get('time', 0) for r in evacuation_result['routes'] 
                           if r.get('destination') == destination]
            center_assignments[destination]['avg_time'] = np.mean(current_times)
    
    return center_assignments

def track_evacuation_progress(evacuation_result, safe_centers_gdf):
    """Real-time progress tracking"""
    
    progress_data = {
        'total_people_in_danger': len(evacuation_result['unreachable']) + len(evacuation_result['evacuated']),
        'successfully_evacuated': len(evacuation_result['evacuated']),
        'still_unreachable': len(evacuation_result['unreachable']),
        'evacuation_rate': len(evacuation_result['evacuated']) / (len(evacuation_result['unreachable']) + len(evacuation_result['evacuated'])) * 100 if (len(evacuation_result['unreachable']) + len(evacuation_result['evacuated'])) > 0 else 0,
        'centers_utilized': len(set(route.get('destination') for route in evacuation_result['routes'])),
        'total_centers_available': len(safe_centers_gdf),
        'center_utilization_rate': len(set(route.get('destination') for route in evacuation_result['routes'])) / len(safe_centers_gdf) * 100 if len(safe_centers_gdf) > 0 else 0
    }
    
    return progress_data

def generate_emergency_alerts(evacuation_result, risk_level, population_at_risk_pct):
    """Generate emergency alerts based on situation"""
    
    alerts = []
    
    # Critical alerts
    if "HIGH RISK" in risk_level:
        alerts.extend([
            "🚨 EMERGENCY EVACUATION ORDER",
            "🔴 IMMEDIATE DANGER - EVACUATE NOW",
            "📢 All residents in flood zones must evacuate immediately",
            "🚑 Emergency services have been notified",
            "📱 Follow official emergency channels for updates"
        ])
    
    # Evacuation status alerts
    unreachable_count = len(evacuation_result['unreachable'])
    if unreachable_count > 0:
        alerts.append(f"⚠️ {unreachable_count} people remain unreachable and need rescue")
    
    # Center capacity alerts
    evacuated_count = len(evacuation_result['evacuated'])
    if evacuated_count > 100:
        alerts.append("🏥 Evacuation centers may be reaching capacity")
    
    # Time-based alerts
    if evacuation_result['times']:
        max_time = max(evacuation_result['times'])
        if max_time > 60:  # More than 1 hour
            alerts.append(f"⏰ Some evacuations taking over {max_time:.0f} minutes")
    
    return alerts

def dijkstra_evacuation(G, flooded_people, safe_centers_gdf, walking_speed_kmph=5):
    """Enhanced Dijkstra evacuation with detailed logging"""
    
    # Validate inputs
    if flooded_people.empty:
        return {
            'evacuated': [],
            'routes': [],
            'times': [],
            'unreachable': [],
            'log': ['No people in flood zone'],
            'execution_time': 0.0,
            'algorithm': 'Dijkstra'
        }
    
    if safe_centers_gdf.empty:
        return {
            'evacuated': [],
            'routes': [],
            'times': [],
            'unreachable': flooded_people.geometry.tolist(),
            'log': ['No safe centers available'],
            'execution_time': 0.0,
            'algorithm': 'Dijkstra'
        }
    
    # Step 1: Assign travel time to each edge
    walking_speed_mpm = walking_speed_kmph * 1000 / 60  # meters per minute
    
    for u, v, k, data in G.edges(keys=True, data=True):
        if 'length' in data:
            data['travel_time'] = data['length'] / walking_speed_mpm

    # Step 2: Evacuation Routing
    evacuated_people, evac_routes, evac_times, unreachable, evac_log = [], [], [], [], []
    start_time = time.time()

    for idx, row in flooded_people.iterrows():
        person_id = row.get('person_id', idx)
        person_point = row.geometry

        try:
            orig_node = find_nearest_node_robust(G, person_point)
        except Exception as e:
            unreachable.append(person_point)
            evac_log.append(f"Person {person_id} could NOT be evacuated (no node found): {e}")
            continue

        best_route, best_time, best_center_id = None, float('inf'), None

        # Try each safe center
        for _, center_row in safe_centers_gdf.iterrows():
            dest_point = center_row.geometry
            center_id = center_row.get('center_id', 'Unknown')

            try:
                dest_node = find_nearest_node_robust(G, dest_point)
                path = nx.shortest_path(G, orig_node, dest_node, weight='travel_time')
                
                # Calculate total time
                time_taken = 0
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    # Handle multiple edges between nodes
                    edge_data = G[u][v]
                    if isinstance(edge_data, dict) and 'travel_time' in edge_data:
                        time_taken += edge_data['travel_time']
                    else:
                        # Multiple edges, take the first one with travel_time
                        for key, data in edge_data.items():
                            if 'travel_time' in data:
                                time_taken += data['travel_time']
                                break

                if time_taken < best_time:
                    best_route = path
                    best_time = time_taken
                    best_center_id = center_id
            except Exception as e:
                evac_log.append(f"No path found from Person {person_id} to {center_id}: {e}")
                continue

        if best_route:
            evac_routes.append({
                'person_id': person_id,
                'origin': person_point, 
                'path': best_route, 
                'time': best_time,
                'destination': best_center_id
            })
            evac_times.append(best_time)
            evacuated_people.append(person_point)
            evac_log.append(f"Person {person_id} evacuated to '{best_center_id}' in {best_time:.2f} min")
        else:
            unreachable.append(person_point)
            evac_log.append(f"Person {person_id} could NOT be evacuated (no path to any safe center)")

    end_time = time.time()

    return {
        'evacuated': evacuated_people,
        'routes': evac_routes,
        'times': evac_times,
        'unreachable': unreachable,
        'log': evac_log,
        'execution_time': end_time - start_time,
        'algorithm': 'Dijkstra'
    }

def astar_evacuation(G, flooded_people, safe_centers_gdf, walking_speed_kmph=5):
    """Enhanced A* evacuation with Euclidean heuristic"""
    
    # Validate inputs
    if flooded_people.empty:
        return {
            'evacuated': [],
            'routes': [],
            'times': [],
            'unreachable': [],
            'log': ['No people in flood zone'],
            'execution_time': 0.0,
            'algorithm': 'A*'
        }
    
    if safe_centers_gdf.empty:
        return {
            'evacuated': [],
            'routes': [],
            'times': [],
            'unreachable': flooded_people.geometry.tolist(),
            'log': ['No safe centers available'],
            'execution_time': 0.0,
            'algorithm': 'A*'
        }

    evacuated_people = []
    evac_routes = []
    evac_times = []
    unreachable = []
    evac_log = []
    start_time = time.time()

    # Assign travel time to edges
    walking_speed_mpm = walking_speed_kmph * 1000 / 60
    for u, v, k, data in G.edges(keys=True, data=True):
        if 'length' in data:
            data['travel_time'] = data['length'] / walking_speed_mpm

    # EUCLIDEAN HEURISTIC
    def euclidean_heuristic(u, v):
        try:
            x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
            x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
            return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
        except:
            return 0

    for i, row in flooded_people.iterrows():
        person_id = row.get('person_id', i)
        person = row.geometry

        try:
            orig_node = find_nearest_node_robust(G, person)
        except Exception as e:
            unreachable.append(person)
            evac_log.append(f"Person {person_id} could NOT be evacuated (no node found): {e}")
            continue

        best_time = float('inf')
        best_route = None
        best_center_id = None

        for idx, center_row in safe_centers_gdf.iterrows():
            try:
                dest_node = find_nearest_node_robust(G, center_row.geometry)
                path = nx.astar_path(G, orig_node, dest_node,
                                   heuristic=euclidean_heuristic,
                                   weight='travel_time')
                
                # Calculate time
                time_taken = 0
                for u, v in zip(path[:-1], path[1:]):
                    edge_data = G.get_edge_data(u, v)
                    if isinstance(edge_data, dict):
                        # Get first edge with travel_time
                        for k, data in edge_data.items():
                            travel_time = data.get('travel_time')
                            if travel_time is not None:
                                time_taken += travel_time
                                break
                    else:
                        time_taken += edge_data.get('travel_time', 1.0)
                
                if time_taken < best_time and len(path) > 1:
                    best_time = time_taken
                    best_route = path
                    best_center_id = center_row.get('center_id', idx)
            except:
                continue

        if best_route:
            evac_routes.append({
                'person_id': person_id,
                'origin': person, 
                'path': best_route, 
                'time': best_time,
                'destination': best_center_id
            })
            evac_times.append(best_time)
            evacuated_people.append(person)
            evac_log.append(f"Person {person_id} evacuated to '{best_center_id}' in {best_time:.2f} min")
        else:
            unreachable.append(person)
            evac_log.append(f"Person {person_id} could NOT be evacuated (no path)")

    end_time = time.time()

    return {
        'evacuated': evacuated_people,
        'routes': evac_routes,
        'times': evac_times,
        'unreachable': unreachable,
        'log': evac_log,
        'execution_time': end_time - start_time,
        'algorithm': 'A*'
    }

def quanta_adaptive_routing_evacuation(G, flooded_people, safe_centers_gdf, walking_speed_kmph=5):
    """Enhanced Quanta Adaptive Routing with dynamic weights"""
    
    # Validate inputs
    if flooded_people.empty:
        return {
            'evacuated': [],
            'routes': [],
            'times': [],
            'unreachable': [],
            'log': ['No people in flood zone'],
            'execution_time': 0.0,
            'algorithm': 'Quanta Adaptive Routing'
        }
    
    if safe_centers_gdf.empty:
        return {
            'evacuated': [],
            'routes': [],
            'times': [],
            'unreachable': flooded_people.geometry.tolist(),
            'log': ['No safe centers available'],
            'execution_time': 0.0,
            'algorithm': 'Quanta Adaptive Routing'
        }

    # DYNAMIC WEIGHT UPDATE
    def update_dynamic_weights(G, congestion_factor=1.0):
        base_speed = walking_speed_kmph * 1000 / 60  # m/min
        for u, v, k, data in G.edges(keys=True, data=True):
            if 'length' in data:
                speed = base_speed
                data['travel_time'] = data['length'] / (speed * congestion_factor)

    # QUANTA ADAPTIVE ROUTING FUNCTION
    def quanta_adaptive_routing(G, origin_node, target_node, max_steps=500):
        visited = set()
        queue = deque()
        queue.append((origin_node, [origin_node], 0))  # (node, path, time)

        best_path = None
        best_time = float('inf')

        while queue and max_steps > 0:
            current, path, time_so_far = queue.popleft()
            visited.add(current)

            if current == target_node:
                if time_so_far < best_time:
                    best_time = time_so_far
                    best_path = path
                continue

            for neighbor in G.neighbors(current):
                if neighbor not in visited:
                    edge_data = G.get_edge_data(current, neighbor)
                    if edge_data:
                        # Get first key or default
                        edge_info = list(edge_data.values())[0]
                        travel_time = edge_info.get('travel_time', float('inf'))
                        queue.append((neighbor, path + [neighbor], time_so_far + travel_time))

            max_steps -= 1

        return best_path, best_time

    start_time = time.time()
    evac_log = []
    evac_routes = []
    evacuated_points = []
    unreachable_points = []
    evac_times = []

    # Assign base costs to edges
    walking_speed_mpm = walking_speed_kmph * 1000 / 60
    for u, v, data in G.edges(data=True):
        if 'length' in data:
            data['base_cost'] = data['length'] / walking_speed_mpm
            data['travel_time'] = data['length'] / walking_speed_mpm
        # Add random penalty for simulation
        data['penalty'] = np.random.uniform(0, 0.5)

    for idx, person_row in flooded_people.iterrows():
        person_id = person_row.get('person_id', idx)
        person_point = person_row.geometry

        try:
            origin_node = find_nearest_node_robust(G, person_point)
            update_dynamic_weights(G, congestion_factor=1.0 + idx * 0.01)
        except:
            unreachable_points.append(person_point)
            evac_log.append(f"Person {person_id} unreachable (no origin node)")
            continue

        best_path = None
        best_cost = float('inf')
        best_center = None

        for center_idx, center_row in safe_centers_gdf.iterrows():
            try:
                center_node = find_nearest_node_robust(G, center_row.geometry)
                center_id = center_row.get('center_id', f'Center_{center_idx}')
                
                path, cost = quanta_adaptive_routing(G, origin_node, center_node)
                if path and cost < best_cost:
                    best_path = path
                    best_cost = cost
                    best_center = center_id
            except:
                continue

        if best_path:
            evac_routes.append({
                'person_id': person_id,
                'origin': person_point,
                'path': best_path, 
                'time': best_cost,
                'destination': best_center
            })
            evacuated_points.append(person_point)
            evac_log.append(f"Person {person_id} evacuated to center '{best_center}' with cost {best_cost:.2f}")
            evac_times.append(best_cost)
        else:
            unreachable_points.append(person_point)
            evac_log.append(f"Person {person_id} unreachable")

    end_time = time.time()

    return {
        'evacuated': evacuated_points,
        'routes': evac_routes,
        'times': evac_times,
        'unreachable': unreachable_points,
        'log': evac_log,
        'execution_time': end_time - start_time,
        'algorithm': 'Quanta Adaptive Routing'
    }

def bidirectional_evacuation(G, flooded_people, safe_centers_gdf, walking_speed_kmph=5):
    """Enhanced Bidirectional Dijkstra"""
    
    # Validate inputs
    if flooded_people.empty:
        return {
            'evacuated': [],
            'routes': [],
            'times': [],
            'unreachable': [],
            'log': ['No people in flood zone'],
            'execution_time': 0.0,
            'algorithm': 'Bidirectional Dijkstra'
        }
    
    if safe_centers_gdf.empty:
        return {
            'evacuated': [],
            'routes': [],
            'times': [],
            'unreachable': flooded_people.geometry.tolist(),
            'log': ['No safe centers available'],
            'execution_time': 0.0,
            'algorithm': 'Bidirectional Dijkstra'
        }

    start_time = time.time()
    routes = []
    evacuated = []
    unreachable = []
    evac_times = []
    log = []

    # Assign travel time to edges
    walking_speed_mpm = walking_speed_kmph * 1000 / 60
    for u, v, k, data in G.edges(keys=True, data=True):
        if 'length' in data:
            data['travel_time'] = data['length'] / walking_speed_mpm

    for person_idx, person_row in flooded_people.iterrows():
        person_point = person_row.geometry
        person_id = person_row.get('person_id', person_idx)

        try:
            source_node = find_nearest_node_robust(G, person_point)
        except Exception:
            unreachable.append(person_point)
            log.append(f"Person {person_id} location node not found in graph.")
            continue

        # Find shortest bidirectional path to any safe center
        shortest_path = None
        shortest_cost = float('inf')
        chosen_center = None

        for center_idx, center_row in safe_centers_gdf.iterrows():
            try:
                target_node = find_nearest_node_robust(G, center_row.geometry)
                center_id = center_row.get('center_id', f'Center_{center_idx}')
                
                if source_node == target_node:
                    shortest_path = [source_node]
                    shortest_cost = 0
                    chosen_center = center_id
                    break
                
                # BIDIRECTIONAL DIJKSTRA
                length, path = nx.bidirectional_dijkstra(G, source_node, target_node, weight='travel_time')
                if length < shortest_cost:
                    shortest_cost = length
                    shortest_path = path
                    chosen_center = center_id
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        if shortest_path is not None:
            routes.append({
                'person_id': person_id,
                'origin': person_point,
                'path': shortest_path,
                'time': shortest_cost,
                'destination': chosen_center
            })
            evac_times.append(shortest_cost)
            evacuated.append(person_point)
            log.append(f"Person {person_id} evacuated to {chosen_center} in {shortest_cost:.2f} min")
        else:
            unreachable.append(person_point)
            log.append(f"Person {person_id} unreachable from any safe center.")

    end_time = time.time()

    return {
        'evacuated': evacuated,
        'routes': routes,
        'times': evac_times,
        'unreachable': unreachable,
        'log': log,
        'execution_time': end_time - start_time,
        'algorithm': 'Bidirectional Dijkstra'
    }
