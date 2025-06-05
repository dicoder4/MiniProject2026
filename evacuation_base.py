"""
Base Evacuation Module
======================

Common utilities and base classes for evacuation algorithms.
Provides shared functionality for all pathfinding algorithms.

Author: Flood Simulation Team  
Date: 2025
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import random
import numpy as np
import folium
from folium.plugins import TimestampedGeoJson
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class EvacuationSimulator:
    """Base class for evacuation simulation with common utilities."""
    
    def __init__(self, people_gdf, flood_gdf, safe_centers_gdf, road_graph):
        """
        Initialize evacuation simulator.
        
        Parameters:
        -----------
        people_gdf : GeoDataFrame
            People locations with person_id and geometry
        flood_gdf : GeoDataFrame
            Flood zone boundaries
        safe_centers_gdf : GeoDataFrame
            Safe evacuation centers
        road_graph : NetworkX Graph
            Road network graph
        """
        self.people_gdf = people_gdf
        self.flood_gdf = flood_gdf
        self.safe_centers_gdf = safe_centers_gdf
        self.road_graph = road_graph
        self.results = {}
    
    def get_flooded_people(self):
        """
        Identify people who are in flooded areas.
        
        Returns:
        --------
        GeoDataFrame: People located within flood zones
        """
        if self.flood_gdf.empty:
            return gpd.GeoDataFrame()
        
        flooded_people = []
        for idx, person in self.people_gdf.iterrows():
            person_point = person.geometry
            if self.flood_gdf.intersects(person_point).any():
                flooded_people.append(person)
        
        if flooded_people:
            return gpd.GeoDataFrame(flooded_people).reset_index(drop=True)
        else:
            return gpd.GeoDataFrame()
    
    def generate_mock_people(self, center_lat, center_lon, num_people=50, radius_km=2):
        """
        Generate random people around a center point for simulation.
        
        Parameters:
        -----------
        center_lat : float
            Center latitude
        center_lon : float
            Center longitude
        num_people : int
            Number of people to generate
        radius_km : float
            Radius in kilometers to spread people
        
        Returns:
        --------
        GeoDataFrame: Generated people with person_id and geometry
        """
        people = []
        radius_deg = radius_km / 111  # Rough conversion km to degrees
        
        for i in range(num_people):
            # Generate random offset within radius
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(0, radius_deg)
            
            lat_offset = distance * np.cos(angle)
            lon_offset = distance * np.sin(angle)
            
            person_lat = center_lat + lat_offset
            person_lon = center_lon + lon_offset
            
            people.append({
                'person_id': f'P{i+1:03d}',
                'geometry': Point(person_lon, person_lat)
            })
        
        return gpd.GeoDataFrame(people, crs='EPSG:4326')
    
    def calculate_evacuation_metrics(self, results):
        """
        Calculate comprehensive evacuation metrics.
        
        Parameters:
        -----------
        results : dict
            Results from evacuation algorithm
        
        Returns:
        --------
        dict: Calculated metrics
        """
        total_people = len(self.get_flooded_people())
        evacuated_count = len(results.get('evacuated', []))
        unreachable_count = len(results.get('unreachable', []))
        evac_times = results.get('times', [])
        
        metrics = {
            'total_people': total_people,
            'evacuated_count': evacuated_count,
            'unreachable_count': unreachable_count,
            'success_rate': (evacuated_count / total_people * 100) if total_people > 0 else 0,
            'avg_time': np.mean(evac_times) if evac_times else 0,
            'max_time': np.max(evac_times) if evac_times else 0,
            'min_time': np.min(evac_times) if evac_times else 0,
            'std_time': np.std(evac_times) if evac_times else 0,
            'execution_time': results.get('execution_time', 0)
        }
        
        return metrics
    
    def export_results_to_csv(self, results, filename):
        """
        Export evacuation results to CSV file.
        
        Parameters:
        -----------
        results : dict
            Results from evacuation algorithm
        filename : str
            Output CSV filename
        """
        routes_data = []
        
        for route in results.get('routes', []):
            routes_data.append({
                'person_id': route.get('person_id', 'Unknown'),
                'origin_x': route['origin'].x,
                'origin_y': route['origin'].y,
                'destination': route.get('destination', 'Unknown'), 
                'travel_time_min': route.get('time', 0),
                'path_length': len(route.get('path', []))
            })
        
        if routes_data:
            df = pd.DataFrame(routes_data)
            df.to_csv(filename, index=False)
            print(f"✅ Results exported to {filename}")
        else:
            print("❌ No routes to export")


def create_evacuation_folium_map(center_lat, center_lon, results, flood_gdf, safe_centers_gdf, 
                                 people_gdf, zoom_start=13):
    """
    Create an interactive Folium map showing evacuation results.
    
    Parameters:
    -----------
    center_lat, center_lon : float
        Map center coordinates
    results : dict
        Evacuation algorithm results
    flood_gdf : GeoDataFrame
        Flood zone boundaries
    safe_centers_gdf : GeoDataFrame
        Safe evacuation centers
    people_gdf : GeoDataFrame
        All people locations
    zoom_start : int
        Initial map zoom level
    
    Returns:
    --------
    folium.Map: Interactive map with evacuation visualization
    """
    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
    
    # Add flood zone
    if not flood_gdf.empty:
        for _, row in flood_gdf.iterrows():
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda x: {
                    'fillColor': 'blue',
                    'color': 'blue',
                    'weight': 2,
                    'fillOpacity': 0.3
                },
                popup='Flood Zone'
            ).add_to(m)
    
    # Add safe centers
    for _, center in safe_centers_gdf.iterrows():
        icon_color = 'red' if center.get('type') == 'hospital' else 'blue'
        icon_name = 'plus' if center.get('type') == 'hospital' else 'shield'
        
        folium.Marker(
            location=[center.geometry.y, center.geometry.x],
            popup=f"🏥 {center.get('name', 'Safe Center')}",
            icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa')
        ).add_to(m)
    
    # Add evacuated people (green)
    for person_point in results.get('evacuated', []):
        folium.Marker(
            location=[person_point.y, person_point.x],
            popup='✅ Evacuated',
            icon=folium.Icon(color='green', icon='user', prefix='fa')
        ).add_to(m)
    
    # Add unreachable people (red)
    for person_point in results.get('unreachable', []):
        folium.Marker(
            location=[person_point.y, person_point.x],
            popup='❌ Unreachable',
            icon=folium.Icon(color='red', icon='user', prefix='fa')
        ).add_to(m)
    
    # Add evacuation routes
    colors = ['orange', 'purple', 'darkgreen', 'cadetblue', 'darkred']
    for i, route in enumerate(results.get('routes', [])[:10]):  # Limit to first 10 routes
        if 'coordinates' in route:
            folium.PolyLine(
                locations=[(lat, lon) for lon, lat in route['coordinates']],
                color=colors[i % len(colors)],
                weight=3,
                opacity=0.7,
                popup=f"Route for {route.get('person_id', 'Person')}"
            ).add_to(m)
    
    return m


def compare_algorithms(results_list, algorithm_names):
    """
    Compare multiple evacuation algorithms and create comparison chart.
    
    Parameters:
    -----------
    results_list : list
        List of results dictionaries from different algorithms
    algorithm_names : list
        List of algorithm names
    
    Returns:
    --------
    matplotlib.Figure: Comparison chart
    """
    metrics = []
    
    for results in results_list:
        stats = results.get('stats', {})
        metrics.append([
            stats.get('success_rate', 0),
            stats.get('avg_time', 0),
            stats.get('execution_time', 0)
        ])
    
    # Create comparison chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Success Rate
    axes[0].bar(algorithm_names, [m[0] for m in metrics], color='green', alpha=0.7)
    axes[0].set_title('Success Rate (%)')
    axes[0].set_ylabel('Percentage')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Average Evacuation Time
    axes[1].bar(algorithm_names, [m[1] for m in metrics], color='orange', alpha=0.7)
    axes[1].set_title('Average Evacuation Time (min)')
    axes[1].set_ylabel('Minutes')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Execution Time
    axes[2].bar(algorithm_names, [m[2] for m in metrics], color='blue', alpha=0.7)
    axes[2].set_title('Algorithm Execution Time (sec)')
    axes[2].set_ylabel('Seconds')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


# Utility functions
def validate_evacuation_data(people_gdf, flood_gdf, safe_centers_gdf, road_graph):
    """
    Validate input data for evacuation simulation.
    
    Returns:
    --------
    dict: Validation results with errors and warnings
    """
    errors = []
    warnings = []
    
    # Check people data
    if people_gdf.empty:
        errors.append("People GeoDataFrame is empty")
    elif 'geometry' not in people_gdf.columns:
        errors.append("People GeoDataFrame missing 'geometry' column")
    
    # Check safe centers
    if safe_centers_gdf.empty:
        warnings.append("No safe evacuation centers available")
    
    # Check road network
    if road_graph is None or road_graph.number_of_nodes() == 0:
        errors.append("Road network graph is empty or invalid")
    
    # Check flood data
    if flood_gdf.empty:
        warnings.append("No flood zone defined")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }


# Example usage
if __name__ == "__main__":
    print("🔧 Base Evacuation Module")
    print("This module provides common utilities for evacuation algorithms.")
    print("Import this module along with specific algorithm modules for full functionality.")
