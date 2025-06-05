"""
Evacuation Runner Module
========================
Contains plotting and execution logic for evacuation algorithms.
Supports multiple routing algorithms and provides comprehensive visualization.
Designed for integration with Streamlit app.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import geopandas as gpd
from shapely.geometry import LineString, Point
import contextily as ctx
import networkx as nx
import osmnx as ox
from haversine import haversine
import logging

# Import evacuation algorithms
from evacuation_algorithms import (
    dijkstra_evacuation,
    astar_evacuation,
    quanta_adaptive_routing_evacuation,
    bidirectional_evacuation
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_evacuation_and_plot(algorithm_name, G, flooded_people, flood_gdf, safe_centers_gdf, 
                           edges, people_gdf, station_name, current_flood_level, 
                           walking_speed_kmph=5, show_plot=True):
    """
    Run evacuation algorithm and generate visualization plot.
    
    Parameters:
    -----------
    algorithm_name : str
        Name of algorithm to use ("Dijkstra", "A*", "Quanta Adaptive Routing", "Bidirectional")
    G : networkx.Graph
        NetworkX graph
    flooded_people : geopandas.GeoDataFrame
        GeoDataFrame of people in flood zones
    flood_gdf : geopandas.GeoDataFrame
        GeoDataFrame of flood zones
    safe_centers_gdf : geopandas.GeoDataFrame
        GeoDataFrame of safe centers
    edges : geopandas.GeoDataFrame
        GeoDataFrame of road network edges
    people_gdf : geopandas.GeoDataFrame
        GeoDataFrame of all people
    station_name : str
        Name of the station/location
    current_flood_level : float
        Current flood level
    walking_speed_kmph : float, default=5
        Walking speed in km/h
    show_plot : bool, default=True
        Whether to display the plot
    
    Returns:
    --------
    dict
        Complete evacuation results with statistics
    """
    
    # Algorithm mapping
    algorithms = {
        "Dijkstra": dijkstra_evacuation,
        "A*": astar_evacuation,
        "Quanta Adaptive Routing": quanta_adaptive_routing_evacuation,
        "Bidirectional": bidirectional_evacuation
    }
    
    # Run the selected algorithm
    if algorithm_name not in algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}. Available: {list(algorithms.keys())}")
    
    results = algorithms[algorithm_name](G, flooded_people, safe_centers_gdf, walking_speed_kmph)
    
    # Extract results
    evac_log = results['log']
    evac_routes = results['routes']
    evacuated_people = results['evacuated']
    unreachable = results['unreachable']
    evac_times = results['times']
    exec_time = results['execution_time']
    
    if show_plot:
        # Create visualization
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # Flood zone
        if not flood_gdf.empty:
            flood_gdf.plot(ax=ax, color='blue', alpha=0.3, label='Flood Zone')
        
        # Roads
        edges.plot(ax=ax, linewidth=1, edgecolor="gray", zorder=1)
        
        # Evacuation routes
        for route in evac_routes:
            try:
                if 'path' in route:
                    path = route['path']
                    line = LineString([(G.nodes[n]['x'], G.nodes[n]['y']) for n in path])
                    gpd.GeoSeries([line], crs=edges.crs).plot(ax=ax, color='orange', linewidth=2, alpha=0.7)
            except Exception as e:
                logger.warning(f"Failed to plot route: {e}")
                continue
        
        # Safe centers
        safe_centers_gdf.plot(ax=ax, color="purple", markersize=80, marker="*", label="Safe Center", zorder=3)
        for _, row in safe_centers_gdf.iterrows():
            geom = row.geometry
            if geom.geom_type == 'Point':
                point = geom
            elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                point = geom.centroid
            else:
                continue
            center_id = row.get('center_id', row.get('name', 'Center'))
            ax.annotate(center_id, xy=(point.x, point.y), xytext=(3, 3),
                       textcoords='offset points', fontsize=9, fontweight='bold', color='black',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
        
        # Evacuated people (green)
        if evacuated_people:
            gpd.GeoSeries(evacuated_people, crs=edges.crs).plot(ax=ax, color="green", markersize=40, label="Evacuated")
        
        # Unreachable people (red x)
        unreach_geom = [pt for pt in unreachable if isinstance(pt, Point)]
        if unreach_geom:
            gpd.GeoSeries(unreach_geom, crs=edges.crs).plot(ax=ax, color="red", marker='x', markersize=40, label="Unreachable")
        
        # Annotate all people with their ID
        for _, row in people_gdf.iterrows():
            person_id = row.get('person_id', 'P')
            ax.annotate(str(person_id), xy=(row.geometry.x, row.geometry.y), xytext=(3, 3),
                       textcoords='offset points', fontsize=8, fontweight='bold', color='black',
                       bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="gray", alpha=0.7))
        
        # Basemap
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=edges.crs)
        except:
            pass  # Skip basemap if not available
        
        # Legend
        ax.legend(handles=[
            Line2D([0], [0], color='gray', lw=2, label='Roads'),
            Line2D([0], [0], color='orange', lw=2, label='Evacuation Route'),
            Line2D([0], [0], marker='o', color='green', markersize=10, linestyle='None', label='Evacuated'),
            Line2D([0], [0], marker='x', color='red', markersize=10, linestyle='None', label='Unreachable'),
            Line2D([0], [0], marker='*', color='purple', markersize=12, linestyle='None', label='Safe Center'),
            Line2D([0], [0], color='blue', lw=6, alpha=0.4, label='Flood Zone')
        ], loc='upper left')
        
        algo_title = {
            "Dijkstra": "Dijkstra's Algorithm",
            "A*": "A* Algorithm", 
            "Quanta Adaptive Routing": "Quanta Adaptive Routing",
            "Bidirectional": "Bidirectional Dijkstra's Algorithm"
        }.get(algorithm_name, "Evacuation Algorithm")
        
        ax.set_title(f"{station_name} Evacuation via {algo_title} @ {current_flood_level:.2f}m", fontsize=15)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    # Print summary
    print(f"🚶 Flooded people: {len(flooded_people)}")
    print(f"✅ Evacuated: {len(evacuated_people)}")
    print(f"❌ Unreachable: {len(unreachable)}")
    if evac_times:
        print(f"⏱️ Estimated evacuation time: Avg = {np.mean(evac_times):.2f} min | Max = {np.max(evac_times):.2f} min")
    print(f"🕒 Script execution time: {exec_time:.2f} sec\n")
    
    print("📋 Evacuation Log:")
    for entry in evac_log[:10]:  # Show first 10 entries
        print(entry)
    if len(evac_log) > 10:
        print(f"... and {len(evac_log) - 10} more entries")
    
    return results

def compare_algorithms(G, flooded_people, safe_centers_gdf, walking_speed_kmph=5):
    """
    Compare all available evacuation algorithms.
    
    Parameters:
    -----------
    G : networkx.Graph
        NetworkX graph
    flooded_people : geopandas.GeoDataFrame
        GeoDataFrame of people in flood zones
    safe_centers_gdf : geopandas.GeoDataFrame
        GeoDataFrame of safe centers
    walking_speed_kmph : float, default=5
        Walking speed in km/h
    
    Returns:
    --------
    dict
        Comparison results for all algorithms
    """
    algorithms = {
        "Dijkstra": dijkstra_evacuation,
        "A*": astar_evacuation,
        "Quanta Adaptive Routing": quanta_adaptive_routing_evacuation,
        "Bidirectional": bidirectional_evacuation
    }
    
    comparison_results = {}
    
    for algorithm_name, algorithm_func in algorithms.items():
        try:
            print(f"Running {algorithm_name}...")
            result = algorithm_func(G, flooded_people, safe_centers_gdf, walking_speed_kmph)
            comparison_results[algorithm_name] = result
        except Exception as e:
            print(f"Failed to run {algorithm_name}: {e}")
            comparison_results[algorithm_name] = {
                'evacuated': [],
                'unreachable': flooded_people.geometry.tolist(),
                'times': [],
                'execution_time': float('inf'),
                'error': str(e),
                'algorithm': algorithm_name
            }
    
    return comparison_results

class EvacuationRunner:
    """
    Main class for running evacuation algorithms and generating visualizations.
    Designed for compatibility with Streamlit app.
    """
    
    def __init__(self, walking_speed_kmph=5):
        """
        Initialize the evacuation runner.
        
        Parameters:
        -----------
        walking_speed_kmph : float, default=5
            Default walking speed for evacuation calculations
        """
        self.walking_speed_kmph = walking_speed_kmph
        self.algorithms = {
            "Dijkstra": dijkstra_evacuation,
            "A*": astar_evacuation,
            "Quanta Adaptive Routing": quanta_adaptive_routing_evacuation,
            "Bidirectional": bidirectional_evacuation
        }
    
    def run_algorithm(self, algorithm_name, G, flooded_people, safe_centers_gdf, walking_speed_kmph=None):
        """
        Run a specific evacuation algorithm.
        
        Parameters:
        -----------
        algorithm_name : str
            Name of algorithm to run
        G : networkx.Graph
            NetworkX graph
        flooded_people : geopandas.GeoDataFrame
            People in flood zones
        safe_centers_gdf : geopandas.GeoDataFrame
            Safe evacuation centers
        walking_speed_kmph : float, optional
            Walking speed (uses default if None)
        
        Returns:
        --------
        dict
            Evacuation results
        """
        if walking_speed_kmph is None:
            walking_speed_kmph = self.walking_speed_kmph
        
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        return self.algorithms[algorithm_name](G, flooded_people, safe_centers_gdf, walking_speed_kmph)
    
    def run_evacuation_and_plot(self, algorithm_name, G, flooded_people, flood_gdf, safe_centers_gdf, 
                               edges, people_gdf, station_name, current_flood_level, 
                               walking_speed_kmph=None, show_plot=True):
        """
        Run evacuation algorithm and generate visualization plot.
        
        This method is compatible with the Streamlit app interface.
        """
        if walking_speed_kmph is None:
            walking_speed_kmph = self.walking_speed_kmph
            
        return run_evacuation_and_plot(
            algorithm_name, G, flooded_people, flood_gdf, safe_centers_gdf,
            edges, people_gdf, station_name, current_flood_level,
            walking_speed_kmph, show_plot
        )
    
    def compare_algorithms(self, G, flooded_people, safe_centers_gdf, walking_speed_kmph=None):
        """
        Compare all available evacuation algorithms.
        
        This method is compatible with the Streamlit app interface.
        """
        if walking_speed_kmph is None:
            walking_speed_kmph = self.walking_speed_kmph
            
        return compare_algorithms(G, flooded_people, safe_centers_gdf, walking_speed_kmph)

def main():
    """Interactive main function for algorithm selection."""
    algorithm_choices = {
        1: "Dijkstra", 
        2: "A*", 
        3: "Quanta Adaptive Routing", 
        4: "Bidirectional"
    }
    
    try:
        print("Available Evacuation Algorithms:")
        for key, value in algorithm_choices.items():
            print(f"{key}. {value}")
        
        selected_num = int(input("Enter your choice (1-4): "))
        algorithm_name = algorithm_choices.get(selected_num)
        
        if algorithm_name is None:
            print("Invalid selection. Please enter 1, 2, 3, or 4.")
            return
        
        print(f"Selected algorithm: {algorithm_name}")
        print("Note: This requires proper initialization of G, flooded_people, safe_centers_gdf, etc.")
        print("Use EvacuationRunner class in your main application.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
