import networkx as nx
import osmnx as ox
import time
import geopandas as gpd
from shapely.geometry import Point, LineString

def prepare_safe_centers(hospitals_gdf, police_gdf, edges, flood_poly):
    """
    Prepare safe evacuation centers from hospitals and police stations.
    Excludes centers that are in flood zones.
    """
    safe_centers = []

    # Collect hospitals
    if hospitals_gdf is not None and not hospitals_gdf.empty:
        for _, row in hospitals_gdf.iterrows():
            safe_centers.append({
                'name': row.get('name', 'Unnamed Hospital'),
                'geometry': row.geometry,
                'type': 'hospital'
            })

    # Collect police stations
    if police_gdf is not None and not police_gdf.empty:
        for _, row in police_gdf.iterrows():
            safe_centers.append({
                'name': row.get('name', 'Unnamed Police Station'),
                'geometry': row.geometry,
                'type': 'police'
            })

    # Fallback to road midpoints if no centers found
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
    if safe_centers:
        safe_centers_gdf = gpd.GeoDataFrame(safe_centers, crs=edges.crs).reset_index(drop=True)
        safe_centers_gdf['center_id'] = safe_centers_gdf['name'].fillna("Unnamed")
        
        # Convert all geometries to Point (use centroid if not already a Point)
        safe_centers_gdf['geometry'] = safe_centers_gdf.geometry.apply(
            lambda g: g if isinstance(g, Point) else g.centroid
        )

        # Exclude centers inside flood zone
        if flood_poly is not None:
            safe_centers_gdf = safe_centers_gdf[
                ~safe_centers_gdf.geometry.apply(lambda g: flood_poly.contains(g))
            ].copy()

        if safe_centers_gdf.empty:
            print("⚠️ All safe centers are flooded. No evacuation centers available.")
    else:
        safe_centers_gdf = gpd.GeoDataFrame(columns=['name', 'geometry', 'type', 'center_id'], crs=edges.crs)

    return safe_centers_gdf

def evacuate_people_with_shortest_path(G, flooded_people, safe_centers_gdf, walking_speed_kmph=5):
    """
    Calculate evacuation routes using shortest path algorithm.
    """
    # Step 1: Assign travel time to each edge
    walking_speed_mpm = walking_speed_kmph * 1000 / 60  # meters per minute

    for u, v, k, data in G.edges(keys=True, data=True):
        if 'length' in data:
            data['travel_time'] = data['length'] / walking_speed_mpm

    # Step 2: Evacuation Routing
    evacuated_people, evac_routes, evac_times, unreachable, evac_log = [], [], [], [], []
    start_time = time.time()

    if safe_centers_gdf.empty:
        evac_log.append("❌ No safe centers available for evacuation")
        return {
            'evacuated': [],
            'routes': [],
            'times': [],
            'unreachable': flooded_people.geometry.tolist(),
            'log': evac_log,
            'execution_time': time.time() - start_time
        }

    for idx, row in flooded_people.iterrows():
        person_id = row['person_id']
        person_point = row.geometry

        try:
            orig_node = ox.distance.nearest_nodes(G, person_point.x, person_point.y)
        except Exception as e:
            unreachable.append(person_point)
            evac_log.append(f"Person {person_id} could NOT be evacuated (no node found): {e}")
            continue

        best_route, best_time, best_center_id = None, float('inf'), None

        # Try each safe center
        for _, center_row in safe_centers_gdf.iterrows():
            dest_point = center_row.geometry
            center_id = center_row.center_id

            try:
                dest_node = ox.distance.nearest_nodes(G, dest_point.x, dest_point.y)
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
        'execution_time': end_time - start_time
    }
