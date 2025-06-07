"""
Enhanced Disaster Response Authority Interface for Emergency Evacuation
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
# Add this import at the top with other imports
from emergency_notifications import get_flood_alert_email
import os
import json
from dotenv import load_dotenv
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Import your existing modules
from flood_simulator import DynamicFloodSimulator, create_elevation_grid
from osm_features import get_osm_features, load_road_network_with_filtering
from evacuation_algorithms import (
    dijkstra_evacuation, 
    astar_evacuation, 
    quanta_adaptive_routing_evacuation, 
    bidirectional_evacuation,
    generate_detailed_evacuation_log,
    generate_evacuation_summary
)
from network_utils import prepare_safe_centers
from visualization_utils import create_flood_folium_map, create_evacuation_folium_map
from risk_assessment import calculate_risk_level


def show_authority_interface():
    """Disaster Response Authority interface with coordinate input and smart evacuation"""
    
    st.markdown("""
    <div style='background: linear-gradient(90deg, #dc3545 0%, #fd7e14 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;'>
        <h2>🚨 Emergency Flood Evacuation System</h2>
        <p><strong>🏢 Disaster Response Authority Portal - Emergency Planning & Evacuation</strong></p>
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

    # Sidebar for authorities
    st.sidebar.title("🏢 Authority Controls")
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Network Setup", 
        "🌊 Flood Simulation", 
        "🚶 Evacuation Planning",
        "📊 Analytics",
        "🆘 Mass SOS & Mock Centers"
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

    # --- Tab 3: Evacuation Planning (Auto Best Algorithm) ---
    with tab3:
        st.header("🚶 Evacuation Route Planning (Automatic Best Algorithm)")

        if 'current_impact' in st.session_state.simulation_data:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Evacuation Parameters")

                # Walking speed
                walking_speed = st.slider("Walking Speed (km/h)", 3, 15, 5, 1)

                # Prepare safe centers if not already
                if st.button("🏥 Prepare Safe Centers"):
                    with st.spinner("Identifying safe evacuation centers..."):
                        try:
                            hospitals_gdf = st.session_state.simulation_data.get('hospitals_gdf')
                            police_gdf = st.session_state.simulation_data.get('police_gdf')
                            edges = st.session_state.simulation_data['edges']
                            impact = st.session_state.simulation_data['current_impact']

                            safe_centers_gdf = prepare_safe_centers(
                                hospitals_gdf, police_gdf, edges, impact['flood_poly']
                            )

                            st.session_state.simulation_data['safe_centers_gdf'] = safe_centers_gdf

                            if not safe_centers_gdf.empty:
                                st.success(f"✅ Found {len(safe_centers_gdf)} safe evacuation centers")
                                st.write("**Safe Centers:**")
                                for idx, center in safe_centers_gdf.iterrows():
                                    st.write(f"- {center.get('center_id', f'Center {idx+1}')}")
                            else:
                                st.warning("⚠️ No safe centers found outside flood zone")
                        except Exception as e:
                            st.error(f"Error preparing safe centers: {e}")

                # Run all algorithms and select the best
                if st.button("🚁 Calculate Evacuation Routes (Auto-Select Best)", type="primary"):
                    if 'safe_centers_gdf' in st.session_state.simulation_data:
                        with st.spinner("Running all evacuation algorithms and selecting the best..."):
                            try:
                                G = st.session_state.simulation_data['G']
                                impact = st.session_state.simulation_data['current_impact']
                                safe_centers_gdf = st.session_state.simulation_data['safe_centers_gdf']
                                flooded_people = impact['flooded_people']
                                location_name = st.session_state.simulation_data.get('location_name', 'Unknown Location')

                                # Run all algorithms
                                algorithms = {
                                    "Dijkstra": dijkstra_evacuation,
                                    "A*": astar_evacuation,
                                    "Quanta Adaptive Routing": quanta_adaptive_routing_evacuation,
                                    "Bidirectional": bidirectional_evacuation
                                }
                                results = {}
                                for name, func in algorithms.items():
                                    try:
                                        result = func(G, flooded_people, safe_centers_gdf, walking_speed)
                                        results[name] = result
                                    except Exception as e:
                                        results[name] = {'evacuated': [], 'times': [], 'unreachable': [], 'execution_time': float('inf'), 'error': str(e)}

                                # Select best: most evacuated, then fastest avg time
                                best_algorithm = None
                                best_result = None
                                max_evacuated = -1
                                min_avg_time = float('inf')
                                for name, result in results.items():
                                    evacuated = len(result.get('evacuated', []))
                                    avg_time = np.mean(result.get('times', [])) if result.get('times') else float('inf')
                                    if evacuated > max_evacuated or (evacuated == max_evacuated and avg_time < min_avg_time):
                                        max_evacuated = evacuated
                                        min_avg_time = avg_time
                                        best_algorithm = name
                                        best_result = result

                                # Save to session state
                                best_result['algorithm'] = best_algorithm
                                st.session_state.simulation_data['evacuation_result'] = best_result

                                # Generate detailed log
                                detailed_log, center_stats = generate_detailed_evacuation_log(
                                    best_result, safe_centers_gdf, location_name, best_algorithm
                                )
                                st.session_state.simulation_data['detailed_log'] = detailed_log
                                st.session_state.simulation_data['center_stats'] = center_stats

                                # Display results
                                st.write("### 🚨 Evacuation Results")
                                total_flooded = len(impact['flooded_people'])
                                evacuated = len(best_result['evacuated'])
                                unreachable = len(best_result['unreachable'])

                                st.metric("People in Danger", total_flooded)
                                st.metric("Successfully Evacuated", evacuated,
                                        f"{evacuated/total_flooded*100:.1f}%" if total_flooded > 0 else "0%")
                                st.metric("Unreachable", unreachable,
                                        f"{unreachable/total_flooded*100:.1f}%" if total_flooded > 0 else "0%")

                                st.write("### 🏥 Evacuation Center Assignments")
                                center_assignments = generate_evacuation_summary(best_result, safe_centers_gdf)
                                for center_id, data in center_assignments.items():
                                    if data['count'] > 0:
                                        st.write(f"**{center_id}** ({data['center_type'].title()})")
                                        st.write(f"👥 {data['count']} people | ⏱️ {data['avg_time']:.1f} min avg")

                                if best_result['times'] and len(best_result['times']) > 0:
                                    avg_time = np.mean(best_result['times'])
                                    max_time = max(best_result['times'])
                                    st.write(f"**Average Evacuation Time:** {avg_time:.1f} minutes")
                                    st.write(f"**Maximum Evacuation Time:** {max_time:.1f} minutes")
                                else:
                                    st.warning("⚠️ No successful evacuations - all people were unreachable")
                                    st.write("**Average Evacuation Time:** N/A")
                                    st.write("**Maximum Evacuation Time:** N/A")

                                st.write(f"**Algorithm Used:** {best_algorithm}")
                                st.write(f"**Execution Time:** {best_result['execution_time']:.2f} seconds")
                            except Exception as e:
                                st.error(f"Error calculating evacuation routes: {e}")
                    else:
                        st.warning("⚠️ Please prepare safe centers first")

            with col2:
                st.subheader("Evacuation Routes Map")

                if 'evacuation_result' in st.session_state.simulation_data:
                    evacuation_result = st.session_state.simulation_data['evacuation_result']
                    safe_centers_gdf = st.session_state.simulation_data['safe_centers_gdf']
                    impact = st.session_state.simulation_data['current_impact']
                    G = st.session_state.simulation_data['G']

                    # Create evacuation map
                    evac_map = create_evacuation_folium_map(
                        lat, lon, evacuation_result, safe_centers_gdf, impact, G
                    )

                    st_folium(evac_map, width=700, height=500)

                    # Enhanced evacuation log with download option
                    with st.expander("📋 Detailed Evacuation Log"):
                        # Show first 20 entries
                        for log_entry in evacuation_result['log'][:20]:
                            st.text(log_entry)
                        if len(evacuation_result['log']) > 20:
                            st.text(f"... and {len(evacuation_result['log']) - 20} more entries")

                        # Download detailed log button
                        if 'detailed_log' in st.session_state.simulation_data:
                            st.markdown("---")
                            st.write("**📥 Download Options:**")

                            # Download detailed log
                            detailed_log = st.session_state.simulation_data['detailed_log']
                            station_name = st.session_state.simulation_data.get('station_name', 'location')
                            filename = f"evacuation_log_{station_name.replace(' ', '_')}_{evacuation_result.get('algorithm', 'best').replace(' ', '_')}.txt"

                            st.download_button(
                                label="📄 Download Detailed Log",
                                data=detailed_log,
                                file_name=filename,
                                mime="text/plain",
                                help="Download comprehensive evacuation log with center-wise statistics"
                            )

                            # Download CSV summary
                            if 'center_stats' in st.session_state.simulation_data:
                                center_stats = st.session_state.simulation_data['center_stats']

                                # Create CSV data
                                csv_data = []
                                csv_data.append("Center_ID,Center_Name,Center_Type,People_Evacuated,Avg_Time_Min,Min_Time_Min,Max_Time_Min,People_IDs")

                                for center_id, stats in center_stats.items():
                                    if stats['count'] > 0:
                                        people_ids = ';'.join(map(str, stats['people_ids']))
                                        csv_data.append(f"{center_id},{stats['center_name']},{stats['center_type']},{stats['count']},{stats['avg_time']:.1f},{stats['min_time']:.1f},{stats['max_time']:.1f},{people_ids}")

                                csv_content = "\n".join(csv_data)
                                csv_filename = f"evacuation_summary_{station_name.replace(' ', '_')}.csv"

                                st.download_button(
                                    label="📊 Download CSV Summary",
                                    data=csv_content,
                                    file_name=csv_filename,
                                    mime="text/csv",
                                    help="Download center-wise evacuation statistics as CSV"
                                )
                else:
                    st.info("👆 Calculate evacuation routes to see visualization")
        else:
            st.warning("⚠️ Please run flood simulation first")

    # --- Tab 4: Analytics ---
    with tab4:
        st.header("📊 Comprehensive Analytics Dashboard")
        
        # CHECK IF EVACUATION RESULT EXISTS IN SESSION STATE
        if 'evacuation_result' in st.session_state.simulation_data:
            evacuation_result = st.session_state.simulation_data['evacuation_result']
            
            # Check if there are any successful evacuations
            if evacuation_result['times'] and len(evacuation_result['times']) > 0:
                # Time series analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Evacuation Timeline")
                    
                    times = sorted(evacuation_result['times'])
                    
                    # Create cumulative evacuation chart
                    timeline_data = []
                    for i, time_val in enumerate(times):
                        timeline_data.append({
                            'Time (minutes)': time_val,
                            'People Evacuated': i + 1,
                            'Cumulative %': (i + 1) / len(times) * 100
                        })
                    
                    timeline_df = pd.DataFrame(timeline_data)
                    
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    fig.add_trace(
                        go.Scatter(x=timeline_df['Time (minutes)'], y=timeline_df['People Evacuated'],
                                  name="People Evacuated", line=dict(color='blue')),
                        secondary_y=False,
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=timeline_df['Time (minutes)'], y=timeline_df['Cumulative %'],
                                  name="Completion %", line=dict(color='red')),
                        secondary_y=True,
                    )
                    
                    fig.update_layout(title="Evacuation Progress Over Time")
                    fig.update_xaxes(title_text="Time (Minutes)")
                    fig.update_yaxes(title_text="People Evacuated", secondary_y=False)
                    fig.update_yaxes(title_text="Completion Percentage", secondary_y=True)
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("🎯 Performance Metrics")
                    
                    impact = st.session_state.simulation_data['current_impact']
                    
                    # Key metrics
                    total_people = st.session_state.simulation_data['num_people']
                    flooded_people = len(impact['flooded_people'])
                    evacuated_people = len(evacuation_result['evacuated'])
                    
                    # Performance metrics with safe calculations
                    st.metric("Evacuation Success Rate", 
                             f"{evacuated_people/flooded_people*100:.1f}%" if flooded_people > 0 else "0%",
                             f"{evacuated_people}/{flooded_people}")
                    
                    # Safe calculation for evacuation times
                    avg_time = np.mean(evacuation_result['times'])
                    max_time = max(evacuation_result['times'])
                    st.metric("Average Evacuation Time", f"{avg_time:.1f} min")
                    st.metric("Maximum Evacuation Time", f"{max_time:.1f} min")
                    
                    st.metric("Algorithm Performance", f"{evacuation_result['execution_time']:.2f} sec")
                    st.metric("Algorithm Used", evacuation_result.get('algorithm', 'Unknown'))
            else:
                # No successful evacuations
                st.error("❌ No successful evacuations found!")
                st.warning("All people were unreachable. This could be due to:")
                st.write("- Network connectivity issues")
                st.write("- All safe centers being in flood zones")
                st.write("- Graph coordinate system mismatches")
                
                # Show basic statistics
                impact = st.session_state.simulation_data['current_impact']
                total_people = st.session_state.simulation_data['num_people']
                flooded_people_count = len(impact['flooded_people'])
                
                # Use sequential layout instead of nested columns
                st.metric("Total People", total_people)
                st.metric("People in Flood Zone", flooded_people_count)
                st.metric("Unreachable", len(evacuation_result['unreachable']))

            # Detailed Analytics Section (only if we have data)
            if 'evacuation_result' in st.session_state.simulation_data and 'current_impact' in st.session_state.simulation_data:
                st.subheader("🔍 Detailed Risk Analysis")
                
                impact = st.session_state.simulation_data['current_impact']
                total_people = st.session_state.simulation_data['num_people']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Population distribution by risk
                    safe_people_count = len(impact['safe_people'])
                    flooded_people_count = len(impact['flooded_people'])
                    evacuated_count = len(evacuation_result['evacuated'])
                    unreachable_count = len(evacuation_result['unreachable'])
                    
                    risk_data = {
                        'Risk Level': ['Safe', 'In Flood Zone', 'Evacuated', 'Unreachable'],
                        'Population': [safe_people_count, flooded_people_count, evacuated_count, unreachable_count],
                        'Percentage': [
                            safe_people_count/total_people*100 if total_people > 0 else 0,
                            flooded_people_count/total_people*100 if total_people > 0 else 0,
                            evacuated_count/total_people*100 if total_people > 0 else 0,
                            unreachable_count/total_people*100 if total_people > 0 else 0
                        ]
                    }
                    risk_df = pd.DataFrame(risk_data)
                    
                    fig = px.pie(risk_df, values='Population', names='Risk Level',
                                title='Population Distribution by Risk Status',
                                color_discrete_map={
                                    'Safe': '#28a745',
                                    'In Flood Zone': '#fd7e14', 
                                    'Evacuated': '#007bff',
                                    'Unreachable': '#dc3545'
                                })
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Infrastructure status
                    blocked_roads = len(impact['blocked_edges'])
                    total_roads = len(st.session_state.simulation_data['edges'])
                    safe_centers_count = len(st.session_state.simulation_data.get('safe_centers_gdf', []))
                    
                    infra_data = {
                        'Infrastructure': ['Roads (Blocked)', 'Roads (Clear)', 'Safe Centers'],
                        'Count': [blocked_roads, total_roads - blocked_roads, safe_centers_count],
                        'Status': ['Critical', 'Good', 'Available']
                    }
                    infra_df = pd.DataFrame(infra_data)
                    
                    fig = px.bar(infra_df, x='Infrastructure', y='Count',
                                title='Infrastructure Status Overview',
                                color='Status',
                                color_discrete_map={
                                    'Critical': '#dc3545',
                                    'Good': '#28a745',
                                    'Available': '#007bff'
                                })
                    st.plotly_chart(fig, use_container_width=True)
                
                # Summary report
                st.subheader("📋 Comprehensive Situation Report")
                
                flood_level_pct = st.session_state.simulation_data.get('flood_level', 0) * 100
                location_name = st.session_state.simulation_data.get('location_name', 'Unknown Location')
                station_name = st.session_state.simulation_data.get('station_name', 'Unknown Station')
                
                # Safe calculations for times
                if evacuation_result['times']:
                    avg_evac_time = np.mean(evacuation_result['times'])
                    max_evac_time = max(evacuation_result['times'])
                else:
                    avg_evac_time = 0
                    max_evac_time = 0
                
                st.markdown(f"""
                **📍 Location Details:**
                - **Location:** {location_name}
                - **Station:** {station_name}
                - **Current Flood Level:** {flood_level_pct:.0f}% of maximum spread
                
                **👥 Population Impact:**
                - **Total Population:** {total_people:,} people
                - **People at Risk:** {flooded_people_count:,} ({flooded_people_count/total_people*100:.1f}%)
                - **Successfully Evacuated:** {evacuated_count:,} ({evacuated_count/flooded_people_count*100:.1f}% of at-risk if flooded_people_count > 0 else 0)
                - **Unreachable/Stranded:** {unreachable_count:,} people
                
                **🛣️ Infrastructure Status:**
                - **Total Roads:** {total_roads:,}
                - **Roads Blocked:** {blocked_roads:,} ({blocked_roads/total_roads*100:.1f}%)
                - **Safe Centers Available:** {safe_centers_count}
                
                **⏱️ Evacuation Performance:**
                - **Algorithm Used:** {evacuation_result.get('algorithm', 'Unknown')}
                - **Average Evacuation Time:** {avg_evac_time:.1f} minutes
                - **Maximum Evacuation Time:** {max_evac_time:.1f} minutes
                - **Calculation Time:** {evacuation_result['execution_time']:.2f} seconds
                """)
                
                # Risk assessment
                population_at_risk_pct = flooded_people_count / total_people * 100 if total_people > 0 else 0
                evacuation_success_rate = evacuated_count / flooded_people_count * 100 if flooded_people_count > 0 else 0
                
                if population_at_risk_pct > 50 or evacuation_success_rate < 70:
                    st.markdown('<div class="alert-high">🔴 HIGH RISK: Immediate evacuation required</div>', unsafe_allow_html=True)
                elif population_at_risk_pct > 20 or evacuation_success_rate < 85:
                    st.markdown('<div class="alert-medium">🟡 MEDIUM RISK: Monitor situation closely</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-low">🟢 LOW RISK: Situation under control</div>', unsafe_allow_html=True)
        else:
            st.info("🏃‍♂️ Complete the evacuation planning to see comprehensive analytics")

    # --- Tab 5: Mass SOS & Mock Centers ---
    with tab5:
        st.header("🆘 Mass SOS Alert & Mock Center Directions")

        users_path = os.path.join(os.path.dirname(__file__), "users.json")
        try:
            with open(users_path, "r", encoding="utf-8") as f:
                users = json.load(f)
        except Exception as e:
            st.error(f"Could not load users.json: {e}")
            users = []
        # Get mock centers from session state
        safe_centers_gdf = st.session_state.simulation_data.get('safe_centers_gdf')
        if safe_centers_gdf is None or safe_centers_gdf.empty:
            st.warning("⚠️ No mock/safe centers found. Please run the flood simulation and evacuation planning first.")
        else:
            # Prepare mock center info with Google Maps links
            mock_centers_info = []
            for idx, row in safe_centers_gdf.iterrows():
                lat, lon = row.geometry.y, row.geometry.x
                gmaps_link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
                mock_centers_info.append({
                    "name": row.get('center_id', f'Center {idx+1}'),
                    "type": row.get('type', 'emergency'),
                    "lat": lat,
                    "lon": lon,
                    "gmaps_link": gmaps_link
                })

            st.write("### 🏥 Mock/Emergency Centers to be sent:")
            for c in mock_centers_info:
                st.markdown(f"- **{c['name']}** ({c['type']}) [Google Maps]({c['gmaps_link']})")

            # Compose the message
          
            subject = "🚨 FLOOD SOS: Emergency Centers & Directions"
            html_centers = "".join([
                f"<li><b>{c['name']}</b> ({c['type']}) - <a href='{c['gmaps_link']}'>Directions</a> ({c['lat']:.5f}, {c['lon']:.5f})</li>"
                for c in mock_centers_info
            ])

            html_message = f"""
            <ul>
                {html_centers}
            </ul>
            </div>
            </body>
            </html>
            """

            if st.button("🚨 SEND MASS SOS TO ALL USERS", type="primary"):
                from emergency_notifications import notification_system
                sent_count = 0
                failed_count = 0
                
                print("Users data:", users)
                
                # Extract emails from users dictionary
                user_emails = []
                for username, user_data in users.items():
                    if isinstance(user_data, dict) and 'email' in user_data:
                        email = user_data['email']
                        if email:  # Check if email is not empty
                            user_emails.append({
                                'email': email,
                                'name': user_data.get('name', username),
                                'username': username
                            })
                            print(f"Found email for {username}: {email}")
                
                print(f"Total emails found: {len(user_emails)}")
                
                if not user_emails:
                    st.warning("⚠️ No user emails found in the system!")
                else:
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Send emails to all users
                    for i, user_info in enumerate(user_emails):
                        email = user_info['email']
                        name = user_info['name']
                        username = user_info['username']
                        user_sub, user_message = get_flood_alert_email(name, selected_state)
                        try:
                            status_text.text(f"Sending to {name} ({email})...")
                            print(email,user_sub, user_message)
                            # Send the email
                            notification_system.send_email_alert(
                                email, 
                                user_sub, 
                                user_message+html_message, 
                                is_html=True
                            )
                            sent_count += 1
                            st.success(f"✅ Sent to {name} ({email})")
                            
                        except Exception as e:
                            failed_count += 1
                            st.error(f"❌ Failed to send to {name} ({email}): {str(e)}")
                            print(f"Error sending to {email}: {e}")
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(user_emails))
                    
                    # Final status
                    status_text.text("Mass notification complete!")
                    
                    if sent_count > 0:
                        st.success(f"✅ Successfully sent SOS alerts to {sent_count} users!")
                    
                    if failed_count > 0:
                        st.warning(f"⚠️ Failed to send to {failed_count} users. Check logs for details.")
                    
                    # Summary
                    st.info(f"📊 Summary: {sent_count} successful, {failed_count} failed out of {len(user_emails)} total emails")


    # Add authority-specific footer
    show_authority_footer()
    
    # Additional sidebar information for authorities
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🆘 Emergency Guide")
        st.markdown("""
        **Quick Steps:**
        1. 🗺️ Load your area network
        2. 🌊 Check flood simulation
        3. 🚶 Find evacuation route
                    
        
 
        """)
        
        st.markdown("### ⚠️ Safety Status")
        
        # System status for authorities
        status_items = [
            ("Network Loaded", 'edges' in st.session_state.simulation_data),
            ("Flood Data Ready", 'current_impact' in st.session_state.simulation_data),
            ("Evacuation Ready", 'evacuation_result' in st.session_state.simulation_data)
        ]
        
        for item_name, status in status_items:
            icon = "✅" if status else "❌"
            color = "green" if status else "red"
            st.markdown(f"<span style='color: {color};'>{icon} {item_name}</span>", unsafe_allow_html=True)


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


def show_authority_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #dc3545; color: white; border-radius: 10px; margin-top: 20px;'>
        <h4>🚨 EMERGENCY EVACUATION SYSTEM</h4>
        <p><strong>⚠️ FOR EMERGENCY USE ONLY</strong></p>
        <p>DISASTER RESPONSE AUTHORITY (FLOODS)</p>
        <p>© 2025 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)

# Make sure this is properly exported
__all__ = ['show_authority_interface',  'generate_mock_centers',  'show_authority_footer']
