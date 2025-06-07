import folium
import geopandas as gpd
from shapely.geometry import Point, LineString

def create_flood_folium_map(lat, lon, people_gdf, impact, edges):
    """
    Create a Folium map showing flood simulation results with gradient colors.
    """
    # Create base map
    m = folium.Map(location=[lat, lon], zoom_start=13)
    
    # Add road network
    for idx, row in edges.iterrows():
        folium.PolyLine(
            locations=[[coord[1], coord[0]] for coord in row.geometry.coords],
            color='gray',
            weight=1,
            opacity=0.7
        ).add_to(m)
    
    # Add flood zones with gradient colors (ENHANCED FROM COLAB)
    if not impact['flood_gdf'].empty:
        for idx, row in impact['flood_gdf'].iterrows():
            # Convert geometry to coordinates
            if hasattr(row.geometry, 'exterior'):
                # Polygon
                coords = [[coord[1], coord[0]] for coord in row.geometry.exterior.coords]
                # Use gradient colors from the flood simulation
                color = row.get('color', 'blue')
                folium.Polygon(
                    locations=coords,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6,
                    weight=2,
                    popup="Flood Zone"
                ).add_to(m)
    
    # Add blocked roads
    if not impact['blocked_edges'].empty:
        for idx, row in impact['blocked_edges'].iterrows():
            folium.PolyLine(
                locations=[[coord[1], coord[0]] for coord in row.geometry.coords],
                color='red',
                weight=4,
                opacity=0.8,
                popup="Blocked Road"
            ).add_to(m)
    
    # Add safe people (green dots)
    if not impact['safe_people'].empty:
        for idx, row in impact['safe_people'].iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=8,
                popup=f"Safe Person {row['person_id']}",
                color='green',
                fill=True,
                fillColor='green',
                fillOpacity=0.7
            ).add_to(m)
    
    # Add flooded people (red X marks)
    if not impact['flooded_people'].empty:
        for idx, row in impact['flooded_people'].iterrows():
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=f"Person {row['person_id']} - IN DANGER",
                icon=folium.Icon(color='red', icon='exclamation-sign')
            ).add_to(m)
    
    # Add center marker
    folium.Marker(
        [lat, lon],
        popup="Flood Center",
        icon=folium.Icon(color='blue', icon='tint')
    ).add_to(m)
    
    return m

def create_evacuation_folium_map(lat, lon, evacuation_result, safe_centers_gdf, impact, G):
    """
    Create a Folium map showing evacuation routes and results.
    ENHANCED VERSION with better error handling and annotations
    """
    # Create base map
    m = folium.Map(location=[lat, lon], zoom_start=13)
    
    # Add flood zones (lighter for background) with gradient colors
    if not impact['flood_gdf'].empty:
        for idx, row in impact['flood_gdf'].iterrows():
            if hasattr(row.geometry, 'exterior'):
                coords = [[coord[1], coord[0]] for coord in row.geometry.exterior.coords]
                color = row.get('color', 'lightblue')
                folium.Polygon(
                    locations=coords,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.3,
                    weight=1,
                    popup="Flood Zone"
                ).add_to(m)
    
    # Add safe centers with enhanced icons
    if not safe_centers_gdf.empty:
        for idx, row in safe_centers_gdf.iterrows():
            center_type = row.get('type', 'unknown')
            if center_type == 'hospital':
                icon_color = 'red'
                icon_name = 'plus'
            elif center_type == 'police':
                icon_color = 'blue'
                icon_name = 'shield'
            else:
                icon_color = 'green'
                icon_name = 'home'
            
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=f"Safe Center: {row['center_id']}",
                icon=folium.Icon(color=icon_color, icon=icon_name)
            ).add_to(m)
    
    # Add evacuation routes with enhanced visualization
    colors = ['purple', 'orange', 'darkgreen', 'pink', 'darkblue', 'cadetblue', 'red', 'lightred', 'beige', 'darkpurple']
    
    for i, route in enumerate(evacuation_result['routes']):
        color = colors[i % len(colors)]
        
        # SAFE ACCESS: Check if path exists
        if 'path' not in route:
            continue
            
        path = route['path']
        
        # Convert node path to coordinates
        route_coords = []
        for node in path:
            if node in G.nodes:
                node_data = G.nodes[node]
                route_coords.append([node_data['y'], node_data['x']])
        
        if len(route_coords) > 1:
            # SAFE ACCESS: Get route time with fallback
            route_time = route.get('time', route.get('time_min', 0))
            
            folium.PolyLine(
                locations=route_coords,
                color=color,
                weight=4,
                opacity=0.8,
                popup=f"Evacuation Route {i+1} - {route_time:.1f} min"
            ).add_to(m)
        
        # Add start point (person location) with enhanced markers
        if 'origin' in route:
            origin = route['origin']
            
            # SAFE ACCESS: Get person ID with multiple fallback options
            person_id = route.get('person_id', 
                                route.get('person_idx', 
                                        route.get('id', f'Person_{i+1}')))
            
            # SAFE ACCESS: Get route time with fallback
            route_time = route.get('time', route.get('time_min', 0))
            
            folium.CircleMarker(
                location=[origin.y, origin.x],
                radius=6,
                popup=f"Person {person_id} - Evacuated in {route_time:.1f} min",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
    
    # Add unreachable people with enhanced markers
    if 'unreachable' in evacuation_result:
        for i, point in enumerate(evacuation_result['unreachable']):
            # Check if point has the required attributes
            if hasattr(point, 'y') and hasattr(point, 'x'):
                folium.Marker(
                    location=[point.y, point.x],
                    popup=f"UNREACHABLE Person {i+1} - No evacuation path found",
                    icon=folium.Icon(color='black', icon='remove')
                ).add_to(m)
    
    # Enhanced legend with more details
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 250px; height: 150px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Evacuation Map Legend</b></p>
    <p><i class="fa fa-plus" style="color:red"></i> Hospitals</p>
    <p><i class="fa fa-shield" style="color:blue"></i> Police Stations</p>
    <p><span style="color:purple">━━━</span> Evacuation Routes</p>
    <p><i class="fa fa-remove" style="color:black"></i> Unreachable</p>
    <p><span style="color:blue">▓▓▓</span> Flood Zones (Gradient)</p>
    <p><span style="color:red">━━━</span> Blocked Roads</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m