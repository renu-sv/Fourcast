import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load Excel file (adjust sheet name if needed)
df = pd.read_excel("WQMN_list.xlsx")  # or use sheet name
print(df.columns.tolist())


# Keep only necessary columns (adjust if names differ)
df = df[['Water\nQuality Station Code', 'Latitude', 'Longitude']]  # use exact column names

# Create geometry column
geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# Set CRS (WGS 84)
gdf.set_crs(epsg=4326, inplace=True)

# Export to shapefile
gdf.to_file("stations.shp")
