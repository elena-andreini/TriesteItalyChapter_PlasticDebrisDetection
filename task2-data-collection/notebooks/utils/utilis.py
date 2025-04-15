import pandas as pd

def excel_file_to_image_coords(df=None):
    if df is None:
        path = r"LM_centroids.xlsx"
        df = pd.read_excel(path)

    d_lat = 0.011575
    d_lon = 0.01339
    for idx, centroid in df.iterrows():
        lat, lon = centroid[:2]
        bbox = (lat-d_lat, lat+d_lat, lon-d_lon, lon+d_lon)
        time = ("2018-12-30T09:24:01Z")
        t = centroid.CodeT
        time = str(centroid.Date) + "-".join([t[:3], t[3:5], t[5:7]]) + "Z"
        yield {"bbox": bbox, "time": time}
