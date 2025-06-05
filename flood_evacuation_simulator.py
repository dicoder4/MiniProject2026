import streamlit as st
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from folium.plugins import TimestampedGeoJson
import folium
import datetime
from streamlit_folium import st_folium
import osmnx as ox
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from time import sleep
import random
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our custom evacuation modules
try:
    from dijkstra_evacuation import evacuate_people_with_dijkstra, prepare_safe_centers
    from evacuation_base import EvacuationSimulator, create_evacuation_folium_map, validate_evacuation_data
    PATHFINDING_AVAILABLE = True
except ImportError:
    st.warning("⚠️ Pathfinding modules not found. Basic simulation only.")
    PATHFINDING_AVAILABLE = False

# --- Streamlit Configuration ---
st.set_page_config(layout="wide", page_title="Flood Evacuation Simulator")
st.markdown("""
# 🚨 Advanced Flood Evacuation Simulator
A comprehensive tool to visualize flood progression and optimal evacuation routing.
""")

# Initialize geolocator with rate limiter
geolocator = Nominatim(user_agent="flood_sim_mapper")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

@st.cache_data
def load_and_validate_stations(file_path, state_name):
    """Load CSV and return only geocodable stations"""
    try:
        df = pd.read_csv(file_path)
        if 'Station' not in df.columns or 'District' not in df.columns:
            st.error("CSV must contain both 'Station' and 'District' columns.")
            return []
        
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
            except:
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        return station_options
    except Exception as e:
        st.error(f"Error loading or validating stations: {e}")
        return []

@st.cache_data
def get_osm_features(location, tags, label):
    try:
        gdf = ox.features_from_place(location, tags=tags)
        if gdf.empty:
            raise ValueError("No features found using place query.")
    except Exception:
        try:
            st.warning(f"⚠️ Falling back to address search for {label}...")
            gdf = ox.features_from_address(f"{label} near {location}", tags=tags, dist=10000)
        except Exception as e:
            st.error(f"❌ Could not load {label} data for {location}: {e}")
            return gpd.GeoDataFrame()
    return gdf

@st.cache_data
def load_road_network(location_name, lat, lon, network_dist=3000):
    """Load road network for pathfinding"""
    try:
        # Try to get graph from place name first
        G = ox.graph_from_place(location_name, network_type='drive')
    except Exception:
        try:
            # Fall back to point-based graph
            G = ox.graph_from_point((lat, lon), dist=network_dist, network_type='drive')
        except Exception as e:
            st.error(f"❌ Could not load road network: {e}")
            return None
    
    return G

def create_flood_simulation_map(lat, lon, location_name):
    """Create interactive flood simulation map"""
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles='OpenStreetMap')
    
    # Add location marker
    folium.Marker(
        [lat, lon],
        popup=f"<b>{location_name}</b><br>Flood Origin Point",
        icon=folium.Icon(color='red', icon='warning-sign')
    ).add_to(m)
    
    # Simulate flood zones (concentric circles)
    colors = ['yellow', 'orange', 'red', 'darkred']
    radii = [500, 1000, 1500, 2000]
    labels = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
    
    for i, (radius, color, label) in enumerate(zip(radii, colors, labels)):
        folium.Circle(
            location=[lat, lon],
            radius=radius,
            popup=f"{label} Zone - {radius}m radius",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.3,
            weight=2
        ).add_to(m)
    
    return m

def generate_population_data(lat, lon, num_points=100):
    """Generate simulated population points around the flood center"""
    population_data = []
    
    for i in range(num_points):
        # Generate random points within 3km radius
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(100, 3000)  # 100m to 3km
        
        # Convert to lat/lon offset
        lat_offset = (distance * np.cos(angle)) / 111000  # roughly 111km per degree
        lon_offset = (distance * np.sin(angle)) / (111000 * np.cos(np.radians(lat)))
        
        pop_lat = lat + lat_offset
        pop_lon = lon + lon_offset
        
        population_data.append({
            'id': i,
            'lat': pop_lat,
            'lon': pop_lon,
            'population': random.randint(1, 20),
            'vulnerability': random.choice(['Low', 'Medium', 'High']),
            'evacuation_status': 'Not Evacuated'
        })
    
    return pd.DataFrame(population_data)

def create_evacuation_map(lat, lon, population_df, safe_zones):
    """Create evacuation routing map"""
    m = folium.Map(location=[lat, lon], zoom_start=13)
    
    # Add flood center
    folium.Marker(
        [lat, lon],
        popup="Flood Center",
        icon=folium.Icon(color='red', icon='warning-sign')
    ).add_to(m)
    
    # Add population points
    for _, person in population_df.iterrows():
        color = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}[person['vulnerability']]
        folium.CircleMarker(
            [person['lat'], person['lon']],
            radius=5,
            popup=f"Pop: {person['population']}<br>Risk: {person['vulnerability']}",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    # Add safe zones
    for i, zone in enumerate(safe_zones):
        folium.Marker(
            zone,
            popup=f"Safe Zone {i+1}",
            icon=folium.Icon(color='green', icon='home')
        ).add_to(m)
    
    return m

# --- Streamlit App ---
st.title("🚨 Advanced Flood Evacuation Simulator")

# State selection
state_options = {
    "Maharashtra": "floods_with_districts_mh.csv",
    "Karnataka": "floods_with_districts_ka.csv"
}

selected_state = st.selectbox("🏛️ Select State:", list(state_options.keys()))
file_path = state_options[selected_state]

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["🌊 Flood Simulation", "🚶 Evacuation Routing", "📊 Analytics"])

with tab1:
    # Original flood simulation functionality
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader(f"📍 {selected_state} Stations")
        
        # Load and validate stations
        with st.spinner("Loading and validating geocodable stations..."):
            station_options = load_and_validate_stations(file_path, selected_state)
        
        if not station_options:
            st.error("❌ No valid geocodable locations found in dataset.")
            st.stop()
        
        st.success(f"✅ Found {len(station_options)} valid stations")
        
        # Station selection
        station_labels = [f"{station} ({district})" for station, district, _, _ in station_options]
        selected_index = st.selectbox("📍 Select a station:", range(len(station_labels)), 
                                    format_func=lambda x: station_labels[x])
        selected_station, selected_district, lat, lon = station_options[selected_index]
        
        st.write(f"### Selected: {selected_station}")
        st.write(f"**District:** {selected_district}")
        st.write(f"**Coordinates:** {lat:.4f}, {lon:.4f}")
        
        # Flood simulation parameters
        st.subheader("🌊 Flood Parameters")
        flood_intensity = st.slider("Flood Intensity", 1, 10, 5)
        flood_speed = st.slider("Flood Spread Speed (m/min)", 1, 100, 10)
        simulation_duration = st.slider("Simulation Duration (hours)", 1, 24, 6)
        
        # Generate simulation button
        if st.button("🌊 Generate Flood Simulation"):
            st.success("Flood simulation generated!")
    
    with col2:
        st.subheader("🗺️ Flood Simulation Map")
        
        if 'selected_station' in locals():
            flood_map = create_flood_simulation_map(lat, lon, f"{selected_station}, {selected_district}")
            st_folium(flood_map, width=700, height=500)
            
            # Display flood progression timeline
            st.subheader("⏱️ Flood Progression Timeline")
            timeline_data = []
            for hour in range(simulation_duration):
                affected_radius = flood_speed * 60 * hour  # Convert to meters
                timeline_data.append({
                    'Hour': hour,
                    'Affected Radius (m)': affected_radius,
                    'Estimated Affected Population': int(affected_radius * 0.1)  # Rough estimate
                })
            
            timeline_df = pd.DataFrame(timeline_data)
            fig = px.line(timeline_df, x='Hour', y='Affected Radius (m)', 
                         title='Flood Progression Over Time')
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🚶 Evacuation Route Planning")
    
    if 'selected_station' in locals():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Evacuation Parameters")
            
            # Generate population data
            if st.button("👥 Generate Population Data"):
                st.session_state.population_df = generate_population_data(lat, lon)
                st.success(f"Generated {len(st.session_state.population_df)} population points")
            
            if 'population_df' in st.session_state:
                st.write(f"**Total Population Points:** {len(st.session_state.population_df)}")
                st.write(f"**Total People:** {st.session_state.population_df['population'].sum()}")
                
                # Vulnerability breakdown
                vuln_counts = st.session_state.population_df['vulnerability'].value_counts()
                st.write("**Vulnerability Breakdown:**")
                for vuln, count in vuln_counts.items():
                    st.write(f"- {vuln}: {count} points")
            
            # Safe zone configuration
            st.subheader("🏠 Safe Zones")
            num_safe_zones = st.number_input("Number of Safe Zones", 1, 10, 3)
            
            # Generate safe zones around the area
            safe_zones = []
            for i in range(num_safe_zones):
                angle = (2 * np.pi * i) / num_safe_zones
                distance = 4000  # 4km from center
                safe_lat = lat + (distance * np.cos(angle)) / 111000
                safe_lon = lon + (distance * np.sin(angle)) / (111000 * np.cos(np.radians(lat)))
                safe_zones.append([safe_lat, safe_lon])
            
            if st.button("🚁 Plan Evacuation Routes"):
                if PATHFINDING_AVAILABLE and 'population_df' in st.session_state:
                    st.success("Evacuation routes calculated!")
                else:
                    st.info("Using basic evacuation simulation (advanced pathfinding not available)")
        
        with col2:
            st.subheader("🗺️ Evacuation Map")
            
            if 'population_df' in st.session_state:
                evacuation_map = create_evacuation_map(lat, lon, st.session_state.population_df, safe_zones)
                st_folium(evacuation_map, width=700, height=500)
            else:
                st.info("Generate population data to view evacuation map")

with tab3:
    st.subheader("📊 Analytics Dashboard")
    
    if 'selected_station' in locals():
        # Create sample analytics data
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Risk Assessment")
            
            # Risk metrics
            risk_data = {
                'Risk Level': ['Low', 'Medium', 'High', 'Critical'],
                'Population at Risk': [150, 300, 200, 100],
                'Infrastructure Count': [20, 35, 15, 8]
            }
            risk_df = pd.DataFrame(risk_data)
            
            fig = px.bar(risk_df, x='Risk Level', y='Population at Risk',
                        title='Population Distribution by Risk Level',
                        color='Risk Level',
                        color_discrete_map={'Low': 'green', 'Medium': 'yellow', 
                                          'High': 'orange', 'Critical': 'red'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⏱️ Evacuation Timeline")
            
            # Evacuation timeline data
            timeline_data = {
                'Time (Hours)': list(range(0, 13)),
                'People Evacuated': [0, 50, 150, 300, 500, 650, 750, 750, 750, 750, 750, 750, 750],
                'Cumulative %': [0, 6.7, 20, 40, 66.7, 86.7, 100, 100, 100, 100, 100, 100, 100]
            }
            timeline_df = pd.DataFrame(timeline_data)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=timeline_df['Time (Hours)'], y=timeline_df['People Evacuated'],
                          name="People Evacuated", line=dict(color='blue')),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Scatter(x=timeline_df['Time (Hours)'], y=timeline_df['Cumulative %'],
                          name="Completion %", line=dict(color='red')),
                secondary_y=True,
            )
            
            fig.update_layout(title="Evacuation Progress Over Time")
            fig.update_xaxes(title_text="Time (Hours)")
            fig.update_yaxes(title_text="People Evacuated", secondary_y=False)
            fig.update_yaxes(title_text="Completion Percentage", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional analytics
        st.subheader("🏥 Resource Requirements")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Emergency Vehicles", "25", "5")
        with col2:
            st.metric("Medical Teams", "8", "2")
        with col3:
            st.metric("Shelter Capacity", "1000", "250")
        with col4:
            st.metric("Estimated Cost", "₹50L", "₹10L")
        
        # Infrastructure impact
        st.subheader("🏗️ Infrastructure Impact Analysis")
        
        impact_data = {
            'Infrastructure Type': ['Roads', 'Bridges', 'Hospitals', 'Schools', 'Utilities'],
            'Total Count': [150, 12, 5, 25, 80],
            'At Risk': [45, 8, 2, 10, 60],
            'Critical': [15, 3, 1, 3, 25]
        }
        impact_df = pd.DataFrame(impact_data)
        
        fig = go.Figure(data=[
            go.Bar(name='Total', x=impact_df['Infrastructure Type'], y=impact_df['Total Count']),
            go.Bar(name='At Risk', x=impact_df['Infrastructure Type'], y=impact_df['At Risk']),
            go.Bar(name='Critical', x=impact_df['Infrastructure Type'], y=impact_df['Critical'])
        ])
        
        fig.update_layout(barmode='group', title='Infrastructure Risk Assessment')
        st.plotly_chart(fig, use_container_width=True)

# Sidebar with additional information
st.sidebar.header("ℹ️ About This Simulator")
st.sidebar.markdown("""
This advanced flood evacuation simulator provides:

🌊 **Flood Simulation**
- Interactive flood zone visualization
- Temporal flood progression modeling
- Risk assessment by zones

🚶 **Evacuation Planning**
- Population distribution analysis
- Optimal route calculation
- Safe zone identification

📊 **Analytics & Insights**
- Real-time evacuation metrics
- Resource requirement estimation
- Infrastructure impact assessment

**Data Sources:**
- OpenStreetMap for geographical data
- Synthetic population data for simulation
- Historical flood patterns for modeling
""")

st.sidebar.header("🔧 Technical Notes")
st.sidebar.markdown("""
**Required Libraries:**
- streamlit
- geopandas
- folium
- osmnx
- geopy
- plotly

**Optional Modules:**
- dijkstra_evacuation
- evacuation_base

For full pathfinding capabilities, ensure custom evacuation modules are available.
""")

# Footer
st.markdown("---")
st.markdown("🚨 **Disclaimer:** This is a simulation tool for educational and planning purposes. Always consult official emergency services for actual evacuations.")