# Import libraries
from cuml.manifold import UMAP
import pandas as pd
import plotly.graph_objects as go
import os
import rasterio
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sknetwork.clustering import Louvain

# Functions for Dimension Reduction and Unsupervised ML

# Function to preprocess dataframe
def pre_process(df):
    # Copy the dataframe
    df_proc = df.copy()
    #Remove unnecesary columnsbox_bounds1
    df_proc = df.drop(['Unnamed: 0.1','Unnamed: 0','tile_name', 'box_bounds0', 'box_bounds1', 'box_bounds2',
                       'box_bounds3','centroid_x', 'centroid_y'], axis=1)
    # get another dataframe with only the tile name
    df_meta = df[['tile_name', 'centroid_x', 'centroid_y']]
    
    return(df_proc, df_meta)

# Function to create the 3D umap function
def run_umap(df_proc, nn, md):
    
    dred = UMAP(n_neighbors = nn, min_dist = md, metric = 'cosine', n_components=3, random_state=0)
    df_2D = dred.fit_transform(df_proc)

    cols = ['Cp1','Cp2', 'Cp3']

    df_2D.columns = cols
    
    return (dred, df_2D)

# Function to create class labels
def get_classes(df_meta2, gas, nogas, n_gas, n_nogas):
    
    geometry = gpd.points_from_xy(df_meta2['centroid_x'], df_meta2['centroid_y'])
    gdf_meta2 = gpd.GeoDataFrame(df_meta2, geometry=geometry)
    
    # standardise the crs
    gdf_meta2.set_crs(gas.crs, inplace=True)

    # Check if each point is within the gas polygon
    df_meta2['gas'] = gdf_meta2['geometry'].within(gas.iloc[n_gas].geometry).astype(int)

    # Check if each point is within the nogas polygon
    df_meta2['nogas'] = gdf_meta2['geometry'].within(nogas.iloc[n_nogas].geometry).astype(int)

    # Assign values 1 and 2 based on the within_gas and within_nogas columns
    df_meta2['class'] = df_meta2['gas'] + (2 * df_meta2['nogas'])

    # drop redundant columns
    df_meta2 = df_meta2.drop(['gas', 'nogas', 'geometry'], axis=1)
    
    return df_meta2

# over-arching data processing function
def preprocess_pipeline(directory_path, var_exp, nn, md, gas, nogas):
    # Initialize a dictionary to store processed dataframes with filenames as keys
    umap_obj = {}
    proc_dfs = {}
    meta_dfs = {}
    
    # Load all of the dfs into the list
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)
            
            # Use the filename (without extension) as the key for the processed dataframe
            filename_without_extension = os.path.splitext(filename)[0]
            
            # Pre-process the dataframe
            df_proc, df_meta = pre_process(df)           
                       
            # Add a geomtry column
            df_meta['geometry'] = df_meta.apply(lambda row: Point(row['centroid_x'], 
                                                                  row['centroid_y']), axis=1)
            # add class labels
            df_meta2 = get_classes(df_meta, gas, nogas, 0, 0)
            
            # append metadata dfs
            meta_dfs[filename_without_extension] = df_meta
            
            # Initialize the StandardScaler
            scaler = StandardScaler()            
            # Scale the data 
            df_proc = pd.DataFrame(scaler.fit_transform(df_proc), columns=df_proc.columns)
            
            # Perform PCA
            pca = PCA(n_components=var_exp)
            PC = pca.fit_transform(df_proc)
            pca_df = pd.DataFrame(PC)
            
            # Perform UMAP
            reduced, df_2D = run_umap(pca_df, nn, md)  
            # append the umap object to a list
            umap_obj[filename_without_extension] = reduced          
           
            # Rename columns of df_2D
            df_2D.columns = [f"{filename_without_extension}_{i}" for i in df_2D.columns]
            
            #append the the dataframe to the list of processed dataframes
            proc_dfs[filename_without_extension] = df_2D

    return (proc_dfs, meta_dfs, umap_obj) 

# Function to visalise the labels in 3D
def Umap_vis(proc_dfs, meta_dfs, var, clr_list):
    # obtain the required dataframe of UMAP embeddings
    df = proc_dfs[var]
    # obtain the required metadatafile
    df['class'] = meta_dfs[var]['class']
    
    column1 = df[var + '_Cp1']
    column2 = df[var + '_Cp2']
    column3 = df[var + '_Cp3']

    # Define the custom color list for each class
    class_colors = clr_list  # Example colors, you can customize this list

    scatter = go.Scatter3d(
        x=column1,
        y=column2,
        z=column3,
        mode='markers',
        marker=dict(
            size=3,
            color=df['class'],
            colorscale=class_colors,
            opacity=0.6,  # Adjust the opacity value (0.0 to 1.0)
            colorbar=dict(
                title='Classes'
            )
        )
    )

    # Create the layout
    layout = go.Layout(
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis'
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    # Create the figure and add the scatter trace
    fig = go.Figure(data=[scatter], layout=layout)

    # Show the plot
    fig.show()

# Function for implementing Louvain clustering
def clusters(umap_obj, var, proc_dfs):
    # extract the graph from the umap object
    dred = umap_obj[var]    
    graph = dred.graph_.get()

    # run louvain clustering
    labels = Louvain().fit_predict(graph)
    print(np.unique(labels))
    
    # get the dataframe
    df = proc_dfs[var]    
    df['clusters'] = labels
    
    return df

# Function to re-order the clusters based on the UMAP embeddings
def re_order_clusters(df):
    # Filter columns that contain 'Cp'
    cp_columns = [col for col in df.columns if 'Cp' in col]

    # Calculate the product of the Cp columns
    df['Cp_Product'] = df[cp_columns].prod(axis=1)

    # Calculate the average 'Cp_Product' for each cluster
    average_cp = df.groupby('clusters')['Cp_Product'].mean()

    # Sort these averages
    sorted_clusters = average_cp.sort_values().index

    # Create a mapping from old cluster labels to new ones
    cluster_mapping = {old: new for new, old in enumerate(sorted_clusters)}

    # Map the old cluster values to the new ones
    df['reordered_cluster'] = df['clusters'].map(cluster_mapping)
    
    return df

# Function to visualise the clusters in 3D
def Umap_vis_alternative(df, var):
    
    column1 = df[var + '_Cp1']
    column2 = df[var + '_Cp2']
    column3 = df[var + '_Cp3']
    classes = df['reordered_cluster']  # Assuming 'classes' is the name of the column

    # Define the custom color list for each class
    colorscale = 'rainbow'  # Choose a desired colorscale

    scatter = go.Scatter3d(
        x=column1,
        y=column2,
        z=column3,
        mode='markers',
        marker=dict(
            size=3,
            opacity=0.8,  # Adjust the opacity value (0.0 to 1.0)
            color=df['reordered_cluster'],
            colorscale=colorscale,
            colorbar=dict(title='tiles')
        )
    )

    # Create the layout
    layout = go.Layout(
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis'
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    # Create the figure and add the scatter trace
    fig = go.Figure(data=[scatter], layout=layout)

    # Show the plot
    fig.show()