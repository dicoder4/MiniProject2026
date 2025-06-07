import googlemaps
from datetime import datetime
import polyline
import geopandas as gpd
from shapely.geometry import LineString

class TrafficSimulator:
    def __init__(self, api_key):
        """Initialize Google Maps client"""
        self.gmaps = googlemaps.Client(key=api_key)
    
    def get_traffic_data(self, origin, destination, departure_time=None):
        """Get traffic data between two points"""
        if departure_time is None:
            departure_time = datetime.now()
            
        try:
            directions = self.gmaps.directions(
                origin,
                destination,
                mode="driving",
                departure_time=departure_time,
                traffic_model="best_guess"
            )
            
            if not directions:
                return None
                
            route = directions[0]
            traffic_info = {
                'duration': route['legs'][0]['duration']['value'],
                'duration_in_traffic': route['legs'][0]['duration_in_traffic']['value'],
                'distance': route['legs'][0]['distance']['value'],
                'geometry': self._decode_polyline(route)
            }
            
            return traffic_info
        except Exception as e:
            print(f"Error getting traffic data: {e}")
            return None
    
    def _decode_polyline(self, route):
        """Decode Google Maps polyline to LineString"""
        points = polyline.decode(route['overview_polyline']['points'])
        coords = [(point[1], point[0]) for point in points]
        return LineString(coords)

    def get_area_traffic(self, center_lat, center_lon, destinations):
        """Get traffic data for multiple destinations around a center point"""
        origin = f"{center_lat},{center_lon}"
        traffic_data = []
        
        for dest in destinations:
            dest_str = f"{dest[0]},{dest[1]}"
            traffic = self.get_traffic_data(origin, dest_str)
            if traffic:
                traffic_data.append(traffic)
        
        return traffic_data