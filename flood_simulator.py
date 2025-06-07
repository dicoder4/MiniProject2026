import numpy as np
import geopandas as gpd
import random
from shapely.geometry import Point, LineString
import matplotlib.cm as cm
from matplotlib.colors import to_hex
import osmnx as ox

def create_elevation_grid(edges, resolution=100):
    """Create elevation grid from edges bounds"""
    xmin, ymin, xmax, ymax = edges.total_bounds
    
    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    xx, yy = np.meshgrid(x, y)
    
    elevation = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            dist_from_west = (xx[i, j] - xmin) / (xmax - xmin)
            base_ele = 2 + 25 * dist_from_west
            variation = np.sin(xx[i, j] * 0.0001) * np.cos(yy[i, j] * 0.0001) * 4
            elevation[i, j] = max(0, base_ele + variation)
    
    points = [Point(xx[i, j], yy[i, j]) for i in range(xx.shape[0]) for j in range(xx.shape[1])]
    elev_vals = [elevation[i, j] for i in range(xx.shape[0]) for j in range(xx.shape[1])]
    elev_gdf = gpd.GeoDataFrame({'elevation': elev_vals}, geometry=points, crs=edges.crs)
    
    return elev_gdf

class DynamicFloodSimulator:
    def __init__(self, elev_gdf, edges, nodes, station, lat, lon, initial_people=50):
        self.elev_gdf = elev_gdf
        self.edges = edges
        self.nodes = nodes
        self.station = station
        self.lat = lat
        self.lon = lon
        self.road_lines = edges.geometry.tolist()

        # Elevation ranges
        self.min_elevation = elev_gdf['elevation'].min()
        self.max_elevation = elev_gdf['elevation'].max()

        # People
        self.people_gdf = self._generate_people(initial_people)
        self.current_people_count = initial_people

    def _generate_people(self, num_people):
        """Generate people randomly along road network with improved positioning"""
        people_points = []
        attempts = 0
        max_attempts = num_people * 50  # Increase attempts significantly

        # Get bounds from edges
        if hasattr(self.edges, 'total_bounds'):
            bounds = self.edges.total_bounds
            xmin, ymin, xmax, ymax = bounds
        else:
            buffer = 0.005  # Smaller buffer for better accuracy
            xmin, ymin = self.lon - buffer, self.lat - buffer
            xmax, ymax = self.lon + buffer, self.lat + buffer

        # Strategy 1: Place people directly on road segments
        road_segments = []
        for idx, row in self.edges.iterrows():
            if isinstance(row.geometry, LineString) and row.geometry.length > 0:
                road_segments.append(row.geometry)

        # Generate points along roads first
        target_on_roads = int(num_people * 0.8)  # 80% on roads
        while len(people_points) < target_on_roads and attempts < max_attempts:
            try:
                if road_segments:
                    road = random.choice(road_segments)
                    # Sample multiple points along the road
                    for fraction in [0.1, 0.3, 0.5, 0.7, 0.9]:
                        if len(people_points) >= target_on_roads:
                            break
                        point = road.interpolate(fraction, normalized=True)
                        if (xmin <= point.x <= xmax and ymin <= point.y <= ymax):
                            people_points.append(point)
                attempts += 1
            except Exception as e:
                attempts += 1
                continue

        # Strategy 2: Place remaining people near road intersections
        if len(people_points) < num_people and hasattr(self, 'nodes'):
            node_coords = [(data.get('x'), data.get('y')) for node, data in self.nodes.iterrows() 
                          if data.get('x') is not None and data.get('y') is not None]
            
            while len(people_points) < num_people and node_coords:
                try:
                    x, y = random.choice(node_coords)
                    # Add small random offset near intersection
                    offset = 0.0001  # Very small offset
                    x += random.uniform(-offset, offset)
                    y += random.uniform(-offset, offset)
                    point = Point(x, y)
                    if (xmin <= point.x <= xmax and ymin <= point.y <= ymax):
                        people_points.append(point)
                except:
                    continue

        # Strategy 3: Fill remaining with grid-based placement near roads
        while len(people_points) < num_people:
            try:
                # Create a grid and find points near roads
                x = random.uniform(xmin, xmax)
                y = random.uniform(ymin, ymax)
                candidate_point = Point(x, y)
                
                # Check if point is reasonably close to any road
                min_distance = float('inf')
                for road in road_segments[:10]:  # Check first 10 roads for efficiency
                    distance = candidate_point.distance(road)
                    min_distance = min(min_distance, distance)
                
                # Accept point if it's within reasonable distance of a road
                if min_distance < 0.001:  # Adjust threshold as needed
                    people_points.append(candidate_point)
                elif len(people_points) < num_people * 0.5:  # If we have very few points, be less strict
                    people_points.append(candidate_point)
                
                if len(people_points) >= num_people:
                    break
                    
            except Exception as e:
                # Last resort: just place randomly
                try:
                    x = random.uniform(xmin, xmax)
                    y = random.uniform(ymin, ymax)
                    people_points.append(Point(x, y))
                except:
                    break

        # Ensure we have the requested number of people
        while len(people_points) < num_people:
            try:
                x = random.uniform(xmin, xmax)
                y = random.uniform(ymin, ymax)
                people_points.append(Point(x, y))
            except:
                break

        print(f"Generated {len(people_points)} people (requested: {num_people})")
        
        return gpd.GeoDataFrame(
            {'person_id': list(range(1, len(people_points) + 1))},
            geometry=people_points,
            crs=self.edges.crs
        )

    def _calculate_flood_impact(self, flood_fraction):
        """Calculate flood impact based on fraction (0-1) - ADVANCED CONCENTRIC RING MODEL"""
        flood_gdf = gpd.GeoDataFrame(geometry=[], crs=self.edges.crs)

        # Origin point from station coordinates
        origin_point = Point(self.lon, self.lat)

        # Project to metric CRS for buffer calculations
        projected_crs = "EPSG:3857"
        origin_gdf = gpd.GeoDataFrame(geometry=[origin_point], crs="EPSG:4326").to_crs(projected_crs)

        # Concentric ring logic for gradient flood effect
        max_radius = 5000  # 5km
        num_rings = 10
        rings_to_fill = int(np.clip(flood_fraction, 0, 1) * num_rings)

        flood_geoms = []
        colors = []
        cmap = cm.get_cmap('Blues', num_rings + 2)

        for i in range(rings_to_fill):
            inner_radius = (i / num_rings) * max_radius
            outer_radius = ((i + 1) / num_rings) * max_radius

            outer = origin_gdf.buffer(outer_radius).geometry.iloc[0]
            inner = origin_gdf.buffer(inner_radius).geometry.iloc[0]
            ring = outer.difference(inner)

            ring_gdf = gpd.GeoDataFrame(geometry=[ring], crs=projected_crs).to_crs(self.edges.crs)
            flood_geoms.append(ring_gdf.geometry.iloc[0])
            colors.append(to_hex(cmap(num_rings - i)))  # darker near center

        if flood_geoms:
            flood_gdf = gpd.GeoDataFrame({'geometry': flood_geoms, 'color': colors}, crs=self.edges.crs)
            flood_poly = flood_gdf.unary_union
        else:
            flood_gdf = gpd.GeoDataFrame(geometry=[], crs=self.edges.crs)
            flood_poly = None

        # Calculate impacts
        if flood_poly is not None:
            flooded_people = self.people_gdf[self.people_gdf.geometry.within(flood_poly)]
            safe_people = self.people_gdf[~self.people_gdf.geometry.within(flood_poly)]
            blocked_edges = self.edges[self.edges.geometry.intersects(flood_poly)]
        else:
            flooded_people = gpd.GeoDataFrame(columns=self.people_gdf.columns, crs=self.people_gdf.crs)
            safe_people = self.people_gdf.copy()
            blocked_edges = gpd.GeoDataFrame(columns=self.edges.columns, crs=self.edges.crs)

        return {
            'flood_gdf': flood_gdf,
            'flood_poly': flood_poly,
            'flooded_people': flooded_people,
            'safe_people': safe_people,
            'blocked_edges': blocked_edges
        }

    def update_people_count(self, new_count):
        """Update the number of people in simulation"""
        if new_count != self.current_people_count:
            self.people_gdf = self._generate_people(new_count)
            self.current_people_count = new_count