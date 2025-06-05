"""
Enhanced Citizen Interface for Emergency Evacuation
Click map to get coordinates, enter them, and find best evacuation route
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import geopandas as gpd
import time
import osmnx as ox
import networkx as nx
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from shapely.geometry import Point, LineString

# Import your existing modules
from flood_simulator import DynamicFloodSimulator, create_elevation_grid
from osm_features import get_osm_features, load_road_network_with_filtering
from evacuation_algorithms import (
    dijkstra_evacuation, 
    astar_evacuation, 
    quanta_adaptive_routing_evacuation, 
    bidirectional_evacuation,
    generate_detailed_evacuation_log
)
from network_utils import prepare_safe_centers
from visualization_utils import create_flood_folium_map, create_evacuation_folium_map
from risk_assessment import calculate_risk_level

def sync_coordinates():
    """Synchronize coordinates between clicked map and input fields"""
    # Get coordinates from various sources
    clicked_coords = st.session_state.simulation_data.get('clicked_coordinates', {})
    
    # Use clicked coordinates if available, otherwise use defaults
    if clicked_coords:
        return clicked_coords['lat'], clicked_coords['lon']
    else:
        # Return default coordinates from selected station
        return st.session_state.simulation_data.get('lat', 19.0760), st.session_state.simulation_data.get('lon', 72.8777)

def show_citizen_interface():
    """Enhanced citizen interface with coordinate input and smart evacuation"""
    
    st.markdown("""
    <div style='background: linear-gradient(90deg, #dc3545 0%, #fd7e14 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;'>
        <h2>🚨 Emergency Flood Evacuation System</h2>
        <p><strong>🏠 Citizen Portal - Emergency Planning & Evacuation</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for simulation data
    if 'simulation_data' not in st.session_state:
        st.session_state.simulation_data = {}
    
    # Initialize geolocator
    @st.cache_resource
    def get_geolocator():
        geolocator = Nominatim(user_agent="flood_sim_mapper")
        return RateLimiter(geolocator.geocode, min_delay_seconds=1)

    geocode = get_geolocator()
    
    @st.cache_data
    def load_and_validate_stations(file_path, state_name):
        """Load CSV and return only geocodable stations"""
        try:
            df = pd.read_csv(file_path)
            if 'Station' not in df.columns or 'District' not in df.columns:
                st.error("CSV must contain both 'Station' and 'District' columns.")
                return [], df
            
            station_options = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            unique_stations = sorted(df[['Station', 'District']].drop_duplicates().dropna().values.tolist())
            
            for i, (station, district) in enumerate(unique_stations):
                progress_bar.progress((i + 1) / len(unique_stations))
                status_text.text(f"Validating station {i + 1}/{len(unique_stations)}: {station}")
                
                location_name = f"{station}, {district}, {state_name}, India"
                try:
                    loc = geocode(location_name)
                    if loc:
                        station_options.append((station, district, loc.latitude, loc.longitude))
                    time.sleep(0.1)  # Rate limiting
                except:
                    continue
            
            progress_bar.empty()
            status_text.empty()
            
            return station_options, df
        except Exception as e:
            st.error(f"Error loading or validating stations: {e}")
            return [], pd.DataFrame()

    # Sidebar for citizens
    st.sidebar.title("🏠 Citizen Controls")
    st.sidebar.markdown("---")

    # Location setup with CSV data
    st.sidebar.subheader("📍 Location Setup")

    # State selection
    state_options = {
        "Maharashtra": "floods_with_districts_mh.csv",
        "Karnataka": "floods_with_districts_ka.csv"
    }

    selected_state = st.sidebar.selectbox("🏛️ Select State:", list(state_options.keys()))
    file_path = state_options[selected_state]

    # Load and validate stations
    with st.spinner("Loading and validating stations..."):
        station_options, df = load_and_validate_stations(file_path, selected_state)

    if not station_options:
        st.error("❌ No valid geocodable locations found in dataset.")
        st.stop()

    st.sidebar.success(f"✅ Found {len(station_options)} valid stations")

    # Station selection dropdown
    station_labels = [f"{station} ({district})" for station, district, _, _ in station_options]
    selected_index = st.sidebar.selectbox("📍 Select Station:", range(len(station_labels)), 
                                        format_func=lambda x: station_labels[x])

    if selected_index is not None:
        station_name, district_name, lat, lon = station_options[selected_index]
        location_name = f"{station_name}, {district_name}, {selected_state}, India"
        
        st.sidebar.write(f"**Selected:** {station_name}")
        st.sidebar.write(f"**District:** {district_name}")
        st.sidebar.write(f"**Coordinates:** {lat:.4f}, {lon:.4f}")

        # Get flood level from CSV
        flood_level_data = df[df['Station'] == station_name]["Peak Flood Level (m)"]
        peak_flood_level = flood_level_data.max() if not flood_level_data.empty else 5.0
        
        st.sidebar.write(f"**Peak Flood Level:** {peak_flood_level:.1f}m")

    # Manual coordinates override (optional)
    st.sidebar.subheader("🎯 Manual Override (Optional)")
    use_manual = st.sidebar.checkbox("Use manual coordinates")
    if use_manual:
        lat = st.sidebar.number_input("Latitude:", value=lat if 'lat' in locals() else 19.0760, format="%.6f")
        lon = st.sidebar.number_input("Longitude:", value=lon if 'lon' in locals() else 72.8777, format="%.6f")
        peak_flood_level = st.sidebar.number_input("Peak flood level (m):", value=peak_flood_level if 'peak_flood_level' in locals() else 3.0, min_value=0.1, max_value=10.0)
        station_name = st.sidebar.text_input("Station name:", value=station_name if 'station_name' in locals() else "Manual Location")
        location_name = f"{station_name}, Manual Location"

    # Create tabs (same as researcher but without Algorithm Comparison and Analytics)
    tab1, tab2, tab3 = st.tabs([
        "🗺️ Network Setup", 
        "🌊 Flood Simulation", 
        "🚶 Evacuation Planning"
    ])
    
    # --- Tab 1: Network Setup (Same as researcher) ---
    with tab1:
        st.header("🗺️ Road Network & Infrastructure")
        
        if 'location_name' in locals():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Network Parameters")
                
                # Network distance
                network_dist = st.slider("Network Radius (meters)", 1000, 5000, 2000, 100)
                
                # Filter minor roads option
                filter_minor = st.checkbox("Filter Minor Roads", value=True)
                
                # Load network button
                if st.button("🔄 Load Road Network", type="primary"):
                    with st.spinner("Loading road network..."):
                        try:
                            # Use your custom function
                            G = load_road_network_with_filtering(location_name, lat, lon, network_dist, filter_minor)
                            
                            if G is not None:
                                nodes, edges = ox.graph_to_gdfs(G)
                                
                                # Add travel time to edges for evacuation algorithms
                                walking_speed_mpm = 5 * 1000 / 60  # 5 km/h in meters per minute
                                for u, v, k, data in G.edges(keys=True, data=True):
                                    if 'length' in data:
                                        data['travel_time'] = data['length'] / walking_speed_mpm
                                        data['weight'] = data['length']
                                        data['base_cost'] = data['length']
                                        data['penalty'] = 0
                                
                                st.session_state.simulation_data.update({
                                    'G': G,
                                    'nodes': nodes,
                                    'edges': edges,
                                    'location_name': location_name,
                                    'lat': lat,
                                    'lon': lon,
                                    'station_name': station_name,
                                    'peak_flood_level': peak_flood_level
                                })
                                st.success(f"✅ Loaded network with {len(nodes)} nodes and {len(edges)} edges")
                            else:
                                st.error("❌ Failed to load road network")
                        except Exception as e:
                            st.error(f"❌ Failed to load road network: {e}")
                
                # Load OSM features
                if st.button("🏥 Load Infrastructure"):
                    if 'edges' in st.session_state.simulation_data:
                        with st.spinner("Loading hospitals and police stations..."):
                            try:
                                # Load hospitals using your custom function
                                hospital_tags = {"amenity": "hospital"}
                                hospitals_gdf = get_osm_features(location_name, hospital_tags, "hospital")
                                
                                # Load police stations
                                police_tags = {"amenity": "police"}
                                police_gdf = get_osm_features(location_name, police_tags, "police station")
                                
                                st.session_state.simulation_data.update({
                                    'hospitals_gdf': hospitals_gdf,
                                    'police_gdf': police_gdf
                                })
                                
                                # Show results
                                if hospitals_gdf is not None and not hospitals_gdf.empty:
                                    st.success(f"✅ Found {len(hospitals_gdf)} hospitals")
                                else:
                                    st.warning("⚠️ No hospitals found")
                                
                                if police_gdf is not None and not police_gdf.empty:
                                    st.success(f"✅ Found {len(police_gdf)} police stations")
                                else:
                                    st.warning("⚠️ No police stations found")
                            except Exception as e:
                                st.error(f"Error loading infrastructure: {e}")
                    else:
                        st.warning("⚠️ Please load road network first")
            
            with col2:
                st.subheader("Network Visualization")
                
                if 'edges' in st.session_state.simulation_data:
                    # Create Folium map
                    m = folium.Map(location=[lat, lon], zoom_start=13)
                    
                    # Add road network
                    edges_data = st.session_state.simulation_data['edges']
                    for idx, row in edges_data.iterrows():
                        coords = list(row.geometry.coords)
                        folium.PolyLine(
                            locations=[[coord[1], coord[0]] for coord in coords],
                            color='blue',
                            weight=2,
                            opacity=0.7
                        ).add_to(m)
                    
                    # Add infrastructure if loaded
                    if 'hospitals_gdf' in st.session_state.simulation_data:
                        hospitals = st.session_state.simulation_data['hospitals_gdf']
                        if hospitals is not None and not hospitals.empty:
                            for idx, row in hospitals.iterrows():
                                if hasattr(row.geometry, 'centroid'):
                                    point = row.geometry.centroid
                                else:
                                    point = row.geometry
                                
                                folium.Marker(
                                    [point.y, point.x],
                                    popup=f"Hospital: {row.get('name', 'Unnamed')}",
                                    icon=folium.Icon(color='red', icon='plus')
                                ).add_to(m)
                    
                    if 'police_gdf' in st.session_state.simulation_data:
                        police = st.session_state.simulation_data['police_gdf']
                        if police is not None and not police.empty:
                            for idx, row in police.iterrows():
                                if hasattr(row.geometry, 'centroid'):
                                    point = row.geometry.centroid
                                else:
                                    point = row.geometry
                                
                                folium.Marker(
                                    [point.y, point.x],
                                    popup=f"Police: {row.get('name', 'Unnamed')}",
                                    icon=folium.Icon(color='blue', icon='shield')
                                ).add_to(m)
                    
                    # Add center marker
                    folium.Marker(
                        [lat, lon],
                        popup=f"{station_name}",
                        icon=folium.Icon(color='green', icon='home')
                    ).add_to(m)
                    
                    st_folium(m, width=700, height=500)
                else:
                    st.info("👆 Load road network to see visualization")

    # --- Tab 2: Flood Simulation ---
    with tab2:
        st.header("🌊 Dynamic Flood Simulation")
        
        if 'edges' in st.session_state.simulation_data:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Simulation Parameters")
                
                # Initialize simulator button
                if st.button("🌊 Initialize Flood Simulator"):
                    with st.spinner("Creating elevation grid and initializing simulator..."):
                        try:
                            edges = st.session_state.simulation_data['edges']
                            nodes = st.session_state.simulation_data['nodes']
                            
                            # Create elevation grid using your function
                            elev_gdf = create_elevation_grid(edges)
                            
                            # Initialize simulator using your class
                            simulator = DynamicFloodSimulator(
                                elev_gdf=elev_gdf,
                                edges=edges,
                                nodes=nodes,
                                station=station_name,
                                lat=lat,
                                lon=lon,
                                initial_people=50
                            )
                            
                            st.session_state.simulation_data['simulator'] = simulator
                            st.session_state.simulation_data['elev_gdf'] = elev_gdf
                            
                            st.success("✅ Flood simulator initialized!")
                        except Exception as e:
                            st.error(f"Error initializing simulator: {e}")
                
                # Simulation controls
                if 'simulator' in st.session_state.simulation_data:
                    st.subheader("Live Controls")
                    
                    # Flood level slider
                    flood_level = st.slider(
                        "Flood Spread (0-100%)",
                        min_value=0,
                        max_value=100,
                        value=20,
                        step=5
                    ) / 100.0
                    
                    # Number of people
                    num_people = st.slider(
                        "Population Count",
                        min_value=10,
                        max_value=200,
                        value=50,
                        step=10
                    )
                    
                    # Update simulation
                    if st.button("🔄 Update Simulation"):
                        simulator = st.session_state.simulation_data['simulator']
                        
                        with st.spinner("Running flood simulation..."):
                            # Update people count
                            simulator.update_people_count(num_people)
                            
                            # Calculate flood impact
                            impact = simulator._calculate_flood_impact(flood_level)
                            
                            st.session_state.simulation_data['current_impact'] = impact
                            st.session_state.simulation_data['flood_level'] = flood_level
                            st.session_state.simulation_data['num_people'] = num_people
                            
                            # Prepare safe centers IMMEDIATELY after flood simulation
                            hospitals_gdf = st.session_state.simulation_data.get('hospitals_gdf')
                            police_gdf = st.session_state.simulation_data.get('police_gdf')
                            edges = st.session_state.simulation_data['edges']
                            
                            # Prepare safe centers (exclude flooded ones)
                            safe_centers_gdf = prepare_safe_centers(hospitals_gdf, police_gdf, edges, impact['flood_poly'])
                            
                            # If no safe centers found, generate mock centers
                            if safe_centers_gdf.empty:
                                st.warning("⚠️ No hospitals or police stations found outside flood zone. Generating emergency centers...")
                                safe_centers_gdf = generate_mock_centers(edges, impact['flood_poly'])
                            
                            st.session_state.simulation_data['safe_centers_gdf'] = safe_centers_gdf
                            
                            # Display statistics
                            st.write("### 📊 Current Simulation Stats")
                            total_people = len(simulator.people_gdf)
                            flooded_people = len(impact['flooded_people'])
                            safe_people = len(impact['safe_people'])
                            
                            # Use sequential layout instead of nested columns
                            st.metric("Total People", total_people)
                            st.metric("In Flood Zone", flooded_people, 
                                    f"{flooded_people/total_people*100:.1f}%" if total_people > 0 else "0%")
                            st.metric("Safe", safe_people,
                                    f"{safe_people/total_people*100:.1f}%" if total_people > 0 else "0%")
                            
                            # Enhanced risk assessment
                            risk_pct = flooded_people / total_people * 100 if total_people > 0 else 0
                            risk_level, _ = calculate_risk_level(flooded_people, total_people)
                            
                            if "HIGH RISK" in risk_level:
                                st.markdown(f'<div style="background-color: #dc3545; color: white; padding: 1rem; border-radius: 5px; margin: 1rem 0;">{risk_level}: {risk_pct:.1f}% of population in flood zone</div>', unsafe_allow_html=True)
                            elif "MEDIUM RISK" in risk_level:
                                st.markdown(f'<div style="background-color: #ffc107; color: black; padding: 1rem; border-radius: 5px; margin: 1rem 0;">{risk_level}: {risk_pct:.1f}% of population in flood zone</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div style="background-color: #28a745; color: white; padding: 1rem; border-radius: 5px; margin: 1rem 0;">{risk_level}: {risk_pct:.1f}% of population in flood zone</div>', unsafe_allow_html=True)
            
            with col2:
                st.subheader("Flood Simulation Map")
                
                if 'current_impact' in st.session_state.simulation_data:
                    impact = st.session_state.simulation_data['current_impact']
                    simulator = st.session_state.simulation_data['simulator']
                    
                    # Create flood simulation map using your function
                    flood_map = create_flood_folium_map(
                        lat, lon, 
                        simulator.people_gdf, 
                        impact,
                        st.session_state.simulation_data['edges']
                    )
                    
                    st_folium(flood_map, width=700, height=500)
                else:
                    st.info("👆 Initialize and run simulation to see results")
        else:
            st.warning("⚠️ Please load road network in the Setup tab first")

    # --- Tab 3: Enhanced Evacuation Planning ---
    with tab3:
        st.header("🚶 Smart Evacuation Planning")
        
        if 'current_impact' in st.session_state.simulation_data:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📍 Your Location")

                # Walking speed
                walking_speed = st.slider("Walking Speed (km/h)", 3, 15, 5, 1)

                # Initialize coordinates
                if 'user_coordinates' not in st.session_state.simulation_data:
                    st.session_state.simulation_data['user_coordinates'] = {'lat': lat, 'lon': lon}

                #st.write("**Step 1: Click on the map to get coordinates**")
                # Display current coordinates
                current_coords = st.session_state.simulation_data['user_coordinates']
                st.success(f"📍 **Current Location:** {current_coords['lat']:.6f}, {current_coords['lon']:.6f}")

                st.write("**Step 1: Enter or adjust your coordinates**")

                # Coordinate inputs
                col_lat, col_lon = st.columns(2)
                with col_lat:
                    new_lat = st.number_input(
                        "Latitude:", 
                        value=current_coords['lat'], 
                        format="%.6f",
                        step=0.000001,
                        key="lat_input"
                    )
                with col_lon:
                    new_lon = st.number_input(
                        "Longitude:", 
                        value=current_coords['lon'], 
                        format="%.6f",
                        step=0.000001,
                        key="lon_input"
                    )

                # Update coordinates if changed
                if new_lat != current_coords['lat'] or new_lon != current_coords['lon']:
                    st.session_state.simulation_data['user_coordinates'] = {'lat': new_lat, 'lon': new_lon}
                    st.rerun()

                st.write("**Step 2: Find evacuation route**")

                if st.button("🚨 FIND BEST EVACUATION ROUTE", type="primary", use_container_width=True):
                    coords = st.session_state.simulation_data['user_coordinates']
                    find_best_evacuation_route(coords['lat'], coords['lon'], walking_speed)

            with col2:
                st.subheader("Evacuation Map")

                if 'current_impact' in st.session_state.simulation_data:
                    impact = st.session_state.simulation_data['current_impact']
                    m = folium.Map(location=[lat, lon], zoom_start=13)

                    # Add road network
                    edges_data = st.session_state.simulation_data['edges']
                    for idx, row in edges_data.iterrows():
                        coords = list(row.geometry.coords)
                        folium.PolyLine(
                            locations=[[coord[1], coord[0]] for coord in coords],
                            color='gray',
                            weight=1,
                            opacity=0.3
                        ).add_to(m)

                    # Flood polygons
                    if not impact['flood_gdf'].empty:
                        for idx, row in impact['flood_gdf'].iterrows():
                            if hasattr(row.geometry, 'exterior'):
                                coords = [[coord[1], coord[0]] for coord in row.geometry.exterior.coords]
                                color = row.get('color', 'red')
                                folium.Polygon(
                                    locations=coords,
                                    color=color,
                                    fill=True,
                                    fillColor=color,
                                    fillOpacity=0.4,
                                    weight=2
                                ).add_to(m)

                    # Safe centers
                    if 'safe_centers_gdf' in st.session_state.simulation_data:
                        safe_centers_gdf = st.session_state.simulation_data['safe_centers_gdf']
                        for idx, row in safe_centers_gdf.iterrows():
                            icon_color = 'green'
                            icon_name = 'home'
                            if row.get('type') == 'hospital':
                                icon_color = 'red'
                                icon_name = 'plus'
                            elif row.get('type') == 'police':
                                icon_color = 'blue'
                                icon_name = 'shield'
                            folium.Marker(
                                [row.geometry.y, row.geometry.x],
                                popup=row.get('center_id', f'Center {idx+1}'),
                                icon=folium.Icon(color=icon_color, icon=icon_name)
                            ).add_to(m)

                    # Current user marker
                    folium.Marker(
                        [current_coords['lat'], current_coords['lon']],
                        popup=f"Your Location: {current_coords['lat']:.6f}, {current_coords['lon']:.6f}",
                        icon=folium.Icon(color='orange', icon='map-pin')
                    ).add_to(m)

                    # Add route if available
                    if 'evacuation_result' in st.session_state.simulation_data:
                        evac_result = st.session_state.simulation_data['evacuation_result']
                        if evac_result.get('best_algorithm') and 'algorithm_results' in evac_result:
                            best_result = evac_result['algorithm_results'][evac_result['best_algorithm']]
                            if best_result.get('routes'):
                                try:
                                    G = st.session_state.simulation_data['G']
                                    route = best_result['routes'][0]
                                    if 'path' in route:
                                        route_coords = [
                                            [G.nodes[n]['y'], G.nodes[n]['x']] for n in route['path'] if n in G.nodes
                                        ]
                                        if len(route_coords) > 1:
                                            folium.PolyLine(
                                                locations=route_coords,
                                                color='#00ff00',
                                                weight=6,
                                                opacity=0.8,
                                                popup=f"Evacuation Route"
                                            ).add_to(m)
                                            folium.Marker(
                                                route_coords[-1],
                                                popup=f"Destination: {route.get('destination', 'Safe Center')}",
                                                icon=folium.Icon(color='green', icon='flag')
                                            ).add_to(m)
                                except Exception as e:
                                    pass

                    # Make map clickable
                    map_data = st_folium(m, width=700, height=500, key="main_evacuation_map")
                    if map_data and map_data.get('last_clicked'):
                        clicked_lat = map_data['last_clicked']['lat']
                        clicked_lon = map_data['last_clicked']['lng']
                        st.session_state.simulation_data['user_coordinates'] = {
                            'lat': clicked_lat,
                            'lon': clicked_lon
                        }
                        st.rerun()

                    # Evacuation result section
                    if 'evacuation_result' in st.session_state.simulation_data:
                        st.markdown("---")
                        show_evacuation_results_below_map()
                else:
                    st.info("👆 Run flood simulation first to see the map")

        else:
            st.warning("⚠️ Please run flood simulation first")

    # Add citizen-specific footer
    show_citizen_footer()
    
    # Additional sidebar information for citizens
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🆘 Emergency Guide")
        st.markdown("""
        **Quick Steps:**
        1. 🗺️ Load your area network
        2. 🌊 Check flood simulation
        3. 🚶 Find evacuation route
        
        **In Emergency:**
        - 📞 Call 112 immediately
        - 🏃 Follow evacuation route
        - 📻 Listen to official updates
        """)
        
        st.markdown("### ⚠️ Safety Status")
        
        # System status for citizens
        status_items = [
            ("Network Loaded", 'edges' in st.session_state.simulation_data),
            ("Flood Data Ready", 'current_impact' in st.session_state.simulation_data),
            ("Evacuation Ready", 'evacuation_result' in st.session_state.simulation_data)
        ]
        
        for item_name, status in status_items:
            icon = "✅" if status else "❌"
            color = "green" if status else "red"
            st.markdown(f"<span style='color: {color};'>{icon} {item_name}</span>", unsafe_allow_html=True)

def find_best_evacuation_route(user_lat, user_lon, walking_speed):
    """Find the best evacuation route by comparing all algorithms"""
    
    with st.spinner("🔍 Finding your best evacuation route..."):
        try:
            # Get simulation data
            G = st.session_state.simulation_data['G']
            impact = st.session_state.simulation_data['current_impact']
            edges = st.session_state.simulation_data['edges']
            
            # Get safe centers (should already be prepared)
            safe_centers_gdf = st.session_state.simulation_data.get('safe_centers_gdf')
            
            if safe_centers_gdf is None or safe_centers_gdf.empty:
                # Prepare safe centers first
                hospitals_gdf = st.session_state.simulation_data.get('hospitals_gdf')
                police_gdf = st.session_state.simulation_data.get('police_gdf')
                
                # Prepare safe centers (exclude flooded ones)
                safe_centers_gdf = prepare_safe_centers(hospitals_gdf, police_gdf, edges, impact['flood_poly'])
                
                # If no safe centers found, generate mock centers
                if safe_centers_gdf.empty:
                    st.warning("⚠️ No hospitals or police stations found outside flood zone. Generating emergency centers...")
                    safe_centers_gdf = generate_mock_centers(edges, impact['flood_poly'])
                
                if safe_centers_gdf.empty:
                    st.error("❌ Could not find any safe evacuation centers")
                    return
                
                st.session_state.simulation_data['safe_centers_gdf'] = safe_centers_gdf
            
            st.success(f"✅ Found {len(safe_centers_gdf)} safe evacuation centers")
            
            # Check if user is in flood zone
            user_point = Point(user_lon, user_lat)
            in_flood_zone = False
            if impact['flood_poly'] and impact['flood_poly'].contains(user_point):
                in_flood_zone = True
            
            if not in_flood_zone:
                st.success("✅ **You are in a SAFE ZONE!**")
                st.info("No evacuation needed, but here are the nearest safe centers:")
                
                # Show nearest centers
                for idx, center in safe_centers_gdf.head(3).iterrows():
                    center_type = center.get('type', 'unknown')
                    center_name = center.get('center_id', f'Center {idx+1}')
                    icon = "🏥" if center_type == "hospital" else "🚓" if center_type == "police" else "🏠"
                    
                    # Calculate distance
                    distance = user_point.distance(center.geometry) * 111000  # Rough conversion to meters
                    st.write(f"{icon} **{center_name}** ({center_type.title()}) - ~{distance:.0f}m away")
                
                # Store result for map display
                st.session_state.simulation_data['evacuation_result'] = {
                    'status': 'safe',
                    'user_location': user_point,
                    'safe_centers_gdf': safe_centers_gdf
                }
                return
            
            st.error("🚨 **You are in a FLOOD ZONE!** Finding evacuation route...")
            
            # Create a person at user's location
            user_gdf = gpd.GeoDataFrame({
                'person_id': ['YOU'],
                'geometry': [user_point]
            }, crs=edges.crs)
            
            # Run all algorithms to find the fastest route
            algorithms = {
                "Dijkstra": dijkstra_evacuation,
                "A*": astar_evacuation,
                "Quanta Adaptive Routing": quanta_adaptive_routing_evacuation,
                "Bidirectional": bidirectional_evacuation
            }
            
            algorithm_results = {}
            best_algorithm = None
            best_time = float('inf')
            
            st.write("### 🔄 Testing All Algorithms...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (alg_name, alg_func) in enumerate(algorithms.items()):
                status_text.text(f"Testing {alg_name}...")
                progress_bar.progress((i + 1) / len(algorithms))
                
                try:
                    result = alg_func(G, user_gdf, safe_centers_gdf, walking_speed)
                    algorithm_results[alg_name] = result
                    
                    # Check if this algorithm found a faster route
                    if result['times'] and len(result['times']) > 0:
                        avg_time = np.mean(result['times'])
                        if avg_time < best_time:
                            best_time = avg_time
                            best_algorithm = alg_name
                    
                except Exception as e:
                    st.warning(f"⚠️ {alg_name} failed: {e}")
                    algorithm_results[alg_name] = {'error': str(e)}
            
            progress_bar.empty()
            status_text.empty()
            
            # Store results
            st.session_state.simulation_data['evacuation_result'] = {
                'algorithm_results': algorithm_results,
                'best_algorithm': best_algorithm,
                'best_time': best_time,
                'safe_centers_gdf': safe_centers_gdf,
                'user_location': user_point,
                'walking_speed': walking_speed,
                'routes': algorithm_results.get(best_algorithm, {}).get('routes', []) if best_algorithm else []
            }
            
            # Show results summary
            if best_algorithm:
                st.success(f"✅ **Best Route Found:** {best_algorithm} algorithm")
                st.write(f"⏱️ **Estimated Time:** {best_time:.1f} minutes")
                
                # Show algorithm comparison
                st.write("### 🏆 Algorithm Performance")
                for alg_name, result in algorithm_results.items():
                    if 'error' not in result and result.get('times'):
                        time_taken = np.mean(result['times'])
                        if alg_name == best_algorithm:
                            st.success(f"🏆 **{alg_name}**: {time_taken:.1f} min (FASTEST)")
                        else:
                            st.info(f"⏱️ **{alg_name}**: {time_taken:.1f} min")
                    elif 'error' in result:
                        st.error(f"❌ **{alg_name}**: Failed")
            else:
                st.error("❌ No evacuation routes found. All routes may be blocked.")
                
        except Exception as e:
            st.error(f"❌ Error finding evacuation route: {e}")

def generate_mock_centers(edges, flood_poly):
    """Generate mock evacuation centers outside flood zones"""
    try:
        # Get bounds and expand them
        minx, miny, maxx, maxy = edges.total_bounds
        dx = (maxx - minx) * 0.1
        dy = (maxy - miny) * 0.1
        bounds = (minx - dx, miny - dy, maxx + dx, maxy + dy)
        
        mock_centers = []
        attempts = 0
        required = 5
        max_attempts = 100
        
        while len(mock_centers) < required and attempts < max_attempts:
            # Generate random point
            x = np.random.uniform(bounds[0], bounds[2])
            y = np.random.uniform(bounds[1], bounds[3])
            pt = Point(x, y)
            
            # Check if point is outside flood zone
            if flood_poly is None or not flood_poly.contains(pt):
                mock_centers.append({
                    'name': f'Emergency Center {len(mock_centers)+1}',
                    'geometry': pt,
                    'type': 'emergency',
                    'center_id': f'Emergency Center {len(mock_centers)+1}'
                })
            attempts += 1
        
        if mock_centers:
            return gpd.GeoDataFrame(mock_centers, crs=edges.crs)
        else:
            return gpd.GeoDataFrame(columns=['name', 'geometry', 'type', 'center_id'], crs=edges.crs)
            
    except Exception as e:
        st.error(f"Error generating mock centers: {e}")
        return gpd.GeoDataFrame(columns=['name', 'geometry', 'type', 'center_id'], crs=edges.crs)

def show_evacuation_results_below_map():
    """Display the evacuation results below the main map"""
    
    evac_result = st.session_state.simulation_data['evacuation_result']
    
    # Handle safe zone case
    if evac_result.get('status') == 'safe':
        st.success("✅ You are in a safe zone!")
        return
    
    best_algorithm = evac_result.get('best_algorithm')
    
    if not best_algorithm:
        st.error("❌ No evacuation routes available")
        return
    
    best_result = evac_result['algorithm_results'][best_algorithm]
    
    # Show best route details
    if best_result.get('routes'):
        route = best_result['routes'][0]  # First (and likely only) route
        destination = route.get('destination', 'Safe Center')
        time_taken = route.get('time', evac_result.get('best_time', 0))
        
        st.markdown("""
        <div style='background: #28a745; color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
            <h3>✅ EVACUATION ROUTE DISPLAYED ON MAP</h3>
            <p><strong>Follow the green route shown above to reach safety</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Use sequential layout instead of columns to avoid nesting
        st.metric("🏆 Best Algorithm", best_algorithm)
        st.metric("⏱️ Evacuation Time", f"{time_taken:.0f} min")
        st.metric("🏥 Destination", destination)
        
        # Step-by-step instructions
        st.subheader("📋 Your Evacuation Instructions")
        
        with st.expander(f"🚶 Route to {destination} ({time_taken:.0f} min)", expanded=True):
            st.write(f"**🏥 Destination:** {destination}")
            st.write(f"**⏱️ Estimated Time:** {time_taken:.0f} minutes")
            st.write(f"**🚶 Walking Speed:** {evac_result['walking_speed']} km/h")
            
            st.write("**📍 Directions:**")
            st.write("1. 🚶 Follow the GREEN route on the map above")
            st.write("2. 🏥 Head towards the FLAG marker (destination)")
            st.write("3. 📱 Keep this map open for navigation")
            st.write("4. ⚠️ Stay alert and follow emergency instructions")
            st.write("5. 📞 Call 112 if you encounter any problems")
        
        # Emergency contacts
        st.subheader("📞 Emergency Contacts")
        
        st.markdown("""
        **🚨 Emergency Services**
        - Emergency: **112**
        - Police: **100**
        - Fire: **101**
        - Medical: **108**
        
        **📻 Stay Informed**
        - Local radio stations
        - Emergency broadcasts
        - Official government alerts
        - Social media updates
        """)

def show_citizen_footer():
    """Show footer with emergency information for citizens"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #dc3545; color: white; border-radius: 10px; margin-top: 20px;'>
        <h4>🚨 EMERGENCY EVACUATION SYSTEM</h4>
        <p><strong>⚠️ FOR EMERGENCY USE ONLY</strong></p>
        <p>This system provides evacuation guidance during flood emergencies. 
        Always follow official emergency services instructions and evacuation orders.</p>
        <div style='margin-top: 15px; font-size: 18px;'>
            <strong>🆘 EMERGENCY NUMBERS:</strong><br>
            <span style='background: white; color: red; padding: 5px 10px; border-radius: 5px; margin: 5px;'>📞 112 - Emergency</span>
            <span style='background: white; color: red; padding: 5px 10px; border-radius: 5px; margin: 5px;'>🚓 100 - Police</span>
            <span style='background: white; color: red; padding: 5px 10px; border-radius: 5px; margin: 5px;'>🚑 108 - Medical</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Make sure this is properly exported
__all__ = ['show_citizen_interface', 'find_best_evacuation_route', 'generate_mock_centers', 'show_evacuation_results_below_map', 'show_citizen_footer', 'sync_coordinates']
