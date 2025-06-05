import osmnx as ox
from osmnx.features import features_from_place, features_from_address

def get_osm_features(location, tags, label):
    """
    Try to load OSM features from a location.
    Falls back from place to address if needed.
    """
    try:
        gdf = features_from_place(location, tags=tags)
        if gdf.empty:
            raise ValueError("No features found using place query.")
    except Exception:
        try:
            print(f"⚠️ Falling back to address search for {label}...")
            gdf = features_from_address(f"{label} near {location}", tags=tags)
        except Exception as e:
            print(f"❌ Could not load {label} data for {location}: {e}")
            return None
    return gdf

def load_road_network_with_filtering(location_name, lat, lon, network_dist, filter_minor=True):
    """
    Load road network with option to filter minor roads.
    Falls back from place to point if needed.
    """
    try:
        # Try graph_from_place first
        G = ox.graph_from_place(location_name, network_type='drive')
    except Exception as e:
        print(f"⚠️ graph_from_place failed: {e}")
        print("📍 Falling back to graph_from_point() ...")
        try:
            # Use provided coordinates as fallback
            G = ox.graph_from_point((lat, lon), dist=network_dist, network_type='drive')
        except Exception as e2:
            print(f"❌ graph_from_point also failed: {e2}")
            return None
    
    if G is None:
        return None
    
    # Filter out minor roads if requested
    if filter_minor:
        minor_types = {'service', 'track', 'path', 'footway', 'bridleway'}
        edges_to_remove = []
        
        for u, v, k, d in G.edges(keys=True, data=True):
            highway = d.get('highway')
            if highway:
                # Handle both string and list highway values
                if isinstance(highway, str) and highway in minor_types:
                    edges_to_remove.append((u, v, k))
                elif isinstance(highway, list) and any(hw in minor_types for hw in highway):
                    edges_to_remove.append((u, v, k))
        
        G.remove_edges_from(edges_to_remove)
        print(f"🚧 Filtered out {len(edges_to_remove)} minor road segments")
    
    return G