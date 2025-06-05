import networkx as nx
import osmnx as ox
import time
import geopandas as gpd
from shapely.geometry import Point, LineString
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_safe_centers_against_flood(safe_centers_gdf, flood_poly, edges):
    """Enhanced validation ensuring centers are truly safe - MISSING FROM STREAMLIT"""
    
    if safe_centers_gdf.empty or flood_poly is None:
        return safe_centers_gdf
    
    # Check each center against flood zones
    safe_centers = []
    flooded_centers = []
    
    for idx, center in safe_centers_gdf.iterrows():
        center_point = center.geometry
        center_id = center.get('center_id', f'Center_{idx}')
        
        # Multiple validation checks
        is_safe = True
        
        # Check 1: Point not in flood polygon
        if flood_poly.contains(center_point):
            is_safe = False
            flooded_centers.append(center_id)
        
        # Check 2: Buffer around center (50m) not intersecting flood
        try:
            buffer_zone = center_point.buffer(0.0005)  # ~50m buffer
            if flood_poly.intersects(buffer_zone):
                is_safe = False
                if center_id not in flooded_centers:
                    flooded_centers.append(center_id)
        except:
            pass
        
        if is_safe:
            safe_centers.append(center.to_dict())
    
    if flooded_centers:
        print(f"⚠️ Excluded {len(flooded_centers)} centers in flood zones: {', '.join(flooded_centers)}")
    
    if safe_centers:
        return gpd.GeoDataFrame(safe_centers, crs=safe_centers_gdf.crs)
    else:
        print("❌ No safe centers remain after flood validation")
        return gpd.GeoDataFrame(columns=safe_centers_gdf.columns, crs=safe_centers_gdf.crs)

def prepare_safe_centers(hospitals_gdf, police_gdf, edges, flood_poly):
    """
    Prepare safe evacuation centers from hospitals and police stations.
    Excludes centers that are in flood zones with advanced fallback logic.
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

        # Enhanced validation against flood zones
        safe_centers_gdf = validate_safe_centers_against_flood(safe_centers_gdf, flood_poly, edges)

        # ADVANCED FALLBACK: If all centers are flooded, generate mock centers outside flood zones
        if safe_centers_gdf.empty:
            print("⚠️ All safe centers are flooded. Generating random mock centers outside flood zones.")
            
            # Expand bounding box by 5% in each direction
            minx, miny, maxx, maxy = edges.total_bounds
            dx = (maxx - minx) * 0.05
            dy = (maxy - miny) * 0.05
            bounds = (minx - dx, miny - dy, maxx + dx, maxy + dy)
            
            mock_centers = []
            attempts = 0
            required = 5
            max_attempts = 200
            
            while len(mock_centers) < required and attempts < max_attempts:
                x = random.uniform(bounds[0], bounds[2])
                y = random.uniform(bounds[1], bounds[3])
                pt = Point(x, y)
                
                # Check if point is outside flood zone
                if flood_poly is None or not flood_poly.contains(pt):
                    mock_centers.append({
                        'name': f'Mock Center {len(mock_centers)+1}',
                        'geometry': pt,
                        'type': 'mock'
                    })
                attempts += 1
            
            if mock_centers:
                safe_centers_gdf = gpd.GeoDataFrame(mock_centers, crs=edges.crs).reset_index(drop=True)
                safe_centers_gdf['center_id'] = safe_centers_gdf['name']
                print(f"✅ Placed {len(mock_centers)} mock centers after {attempts} attempts.")
            else:
                print("❌ Could not place mock centers outside flood zones after max attempts.")
    else:
        safe_centers_gdf = gpd.GeoDataFrame(columns=['name', 'geometry', 'type', 'center_id'], crs=edges.crs)

    return safe_centers_gdf

def assign_people_to_centers_with_capacity(evacuation_result, safe_centers_gdf, max_capacity_per_center=50):
    """Smart assignment considering center capacity - MISSING FROM STREAMLIT"""
    
    # Initialize center capacities
    center_capacities = {}
    for _, center in safe_centers_gdf.iterrows():
        center_id = center.get('center_id')
        center_type = center.get('type', 'unknown')
        
        # Different capacities based on center type
        if center_type == 'hospital':
            capacity = max_capacity_per_center * 2  # Hospitals can handle more
        elif center_type == 'police':
            capacity = max_capacity_per_center * 1.5  # Police stations moderate capacity
        else:
            capacity = max_capacity_per_center  # Mock centers basic capacity
            
        center_capacities[center_id] = {
            'max_capacity': capacity,
            'current_occupancy': 0,
            'assigned_people': []
        }
    
    # Reassign people based on capacity
    for route in evacuation_result['routes']:
        destination = route.get('destination')
        person_id = route.get('person_id')
        
        if destination in center_capacities:
            if center_capacities[destination]['current_occupancy'] < center_capacities[destination]['max_capacity']:
                center_capacities[destination]['current_occupancy'] += 1
                center_capacities[destination]['assigned_people'].append(person_id)
            else:
                # Find alternative center
                for alt_center, capacity_info in center_capacities.items():
                    if capacity_info['current_occupancy'] < capacity_info['max_capacity']:
                        route['destination'] = alt_center
                        capacity_info['current_occupancy'] += 1
                        capacity_info['assigned_people'].append(person_id)
                        break
    
    return center_capacities

def setup_graph_for_evacuation(G: nx.Graph, walking_speed_kmph: float = 5) -> nx.Graph:
    """
    Prepare and configure the graph for evacuation routing by adding necessary edge attributes.
    """
    try:
        # Convert walking speed to meters per minute
        walking_speed_mpm = walking_speed_kmph * 1000 / 60
        
        # Add travel time to edges
        for u, v, k, data in G.edges(keys=True, data=True):
            if 'length' in data:
                # Calculate travel time in minutes
                data['travel_time'] = data['length'] / walking_speed_mpm
                
                # Set weight for general pathfinding (using length)
                data['weight'] = data['length']
                
                # Set base cost for adaptive routing
                data['base_cost'] = data['length']
                
                # Initialize penalty (can be modified during flood simulation)
                data['penalty'] = 0
                
                # Calculate adaptive cost
                data['adaptive_cost'] = data['base_cost'] + data['penalty']
            else:
                # Default values if length is missing
                logger.warning(f"Missing length for edge ({u}, {v}, {k}). Using default values.")
                data['travel_time'] = 1.0  # 1 minute default
                data['weight'] = 100.0     # 100 meters default
                data['base_cost'] = 100.0
                data['penalty'] = 0
                data['adaptive_cost'] = 100.0
        
        logger.info(f"Graph prepared for evacuation: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
        
    except Exception as e:
        logger.error(f"Failed to setup graph for evacuation: {str(e)}")
        return G

def get_nearest_node_robust(G, point, max_distance=1000):
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
