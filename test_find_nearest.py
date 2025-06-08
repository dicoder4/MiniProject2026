import requests

def find_nearest_safe_center_gmaps(user_address, safe_centers, api_key):
    """
    user_address: str - Address of the user
    safe_centers: list of dicts with lat/lon
    api_key: str - Google Maps API key
    """
    # Geocode the user address
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={user_address}&key={api_key}"
    response = requests.get(geocode_url).json()

    if response["status"] != "OK" or not response["results"]:
        print("Failed to geocode user address.")
        return None

    user_location = response["results"][0]["geometry"]["location"]
    user_latlon = f"{user_location['lat']},{user_location['lng']}"

    # Build destinations string for all mock centers
    destinations = "|".join([f"{c['lat']},{c['lon']}" for c in safe_centers])

    # Call Distance Matrix API
    matrix_url = (
        f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={user_latlon}"
        f"&destinations={destinations}&key={api_key}"
    )
    matrix_response = requests.get(matrix_url).json()

    if matrix_response["status"] != "OK":
        print("Distance Matrix API error:", matrix_response.get("error_message"))
        return None

    distances = matrix_response["rows"][0]["elements"]

    # Find the closest center
    min_dist = float("inf")
    best_center = None
    for i, element in enumerate(distances):
        if element["status"] == "OK":
            dist_km = element["distance"]["value"] / 1000  # meters to km
            if dist_km < min_dist:
                min_dist = dist_km
                best_center = safe_centers[i]
                best_center["distance_km"] = dist_km
                best_center["gmaps_link"] = f"https://www.google.com/maps/search/?api=1&query={best_center['lat']},{best_center['lon']}"

    return best_center

safe_centers = [
    {"name": "Center A", "lat": 19.0760, "lon": 72.8777},  # Mumbai
    {"name": "Center B", "lat": 12.9716, "lon": 77.5946},  # Bangalore
    {"name": "Center C", "lat": 17.3850, "lon": 78.4867},  # Hyderabad
]

api_key = "AIzaSyAR43jUoPTiNpTyqj8jlJcupR2-g9OFHKo"
user_address = "Purva Heights, Bengaluru-560076"

result = find_nearest_safe_center_gmaps(user_address, safe_centers, api_key)
if result:
    print(f"Nearest center: {result['name']} ({result['distance_km']:.2f} km)")
    print("Google Maps link:", result["gmaps_link"])
else:
    print("Could not find nearest center.")

