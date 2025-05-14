import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from kmodes.kmodes import KModes
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
import numpy as np
import pre_cluster

v_path = "./vehicle.csv"
a_path = "./accident.csv"
p_path = "./person.csv"
atmo_path = "./atmospheric_cond.csv"

everything_df = pre_cluster.everything_df(v_path, a_path, atmo_path, p_path)

# Using a sample to perform agglomerative clustering
aggcls_sample_df = everything_df.sample(n = 20000, random_state=20250523)

targetvars = ['VEHICLE_DAMAGE_LEVEL','AVERAGE_INJ_LEVEL']
featurevars1 = ['NO_OF_WHEELS','NO_OF_CYLINDERS','SEATING_CAPACITY','TARE_WEIGHT','TOTAL_NO_OCCUPANTS']
featurevars2 = ['LIGHT_LEVEL', 'SPEED_ZONE', 'MAIN_ATMOSPH_COND', 'ROAD_SURFACE_TYPE_DESC']

def normalize(df, features):
    # Takes a dataframe and an array of column names and returns a min max normalized dataframe of those columns
    normalized = (df[features] - df[features].min())/(df[features].max() - df[features].min())
    return normalized

def kmeans_elbow(df, features, elbowpoint):
    # Takes a dataframe with target columns and performs elbow method for KMeans, highlighting a point
    # Saves output as png

    df = normalize(df, features)

    # Code derived from workshop 6
    distortions = []
    k_range = range(1,11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=20250523).fit(df)
        distortions.append(kmeans.inertia_)
    # Setting random state for consistency

    plt.plot(k_range,distortions,'bo-')
    plt.title('Elbow Method')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.xticks(range(1,11))

    plt.plot(elbowpoint,distortions[elbowpoint - 1],'ro',markersize=10)
    plt.annotate("Elbow Point",(elbowpoint + 0.3,(distortions[elbowpoint - 1]*1.1)))

    plt.savefig("elbow_point_kmeans.png")
    plt.close()

def kmodes_elbow(df, features, elbowpoint):
    # Takes a dataframe with target columns and performs elbow method for KModes, highlighting a point
    # Saves output as png

    # Code derived from workshop 6
    distortions = []
    k_range = range(1,11)
    for k in k_range:
        kmode = KModes(n_clusters=k, random_state=20250523, n_init=5)
        kmode.fit(df[features])
        distortions.append(kmode.cost_)
    # Setting random state for consistency

    # this took 4m 25.4s to run at n_init=2

    plt.plot(k_range,distortions,'bo-')
    plt.title('Elbow Method')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.xticks(range(1,11))

    plt.plot(elbowpoint,distortions[elbowpoint-1],'ro',markersize=10)
    plt.annotate("Elbow Point",(elbowpoint+0.3,(distortions[elbowpoint-1]*1.1)))

    plt.savefig("elbow_point_kmodes.png")
    plt.close()

def aggcls_dendrogram(df, features):
    # Takes a dataframe with target columns and creates a dendrogram
    # Saves output as png
    normalized = normalize(df, features)

    # Code derived from SKLearn Documentation example
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage="ward").fit(normalized)

    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig("hierarchical_clustering_dendrogram.png")
    plt.close()

def kmeans_clustering(df, features, k):
    # KMeans Clustering analysis

    normalized = normalize(df, features)
    kmeans = KMeans(n_clusters=k, random_state=20250523).fit(normalized)
    kmeans_df = df.copy()
    kmeans_df['cluster'] = kmeans.labels_
    kmeans_df.to_csv("kmeans_df.csv")

    kmeans_agg = kmeans_df[['cluster','AVERAGE_INJ_LEVEL','VEHICLE_DAMAGE_LEVEL']].groupby(['cluster']).mean()
    kmeans_agg['count'] = kmeans_df[['cluster','ACCIDENT_NO']].groupby(['cluster']).count()
    kmeans_agg.to_csv("kmeans_aggregate_stats.csv")

    rowlabels = ["Aggregate"]
    kmeans_clusters = []
    for i in range(0,k):
        kmeans_clusters.append(kmeans_df[kmeans_df['cluster']==i])
        rowlabels.append("Cluster "+str(i))

    # Getting vehicle counts per cluster
    kmeans_vehiclecounts = pd.DataFrame()
    for cat in kmeans_df['VEHICLE_CATEGORY'].unique():
        vehiclecounts = [sum(kmeans_df['VEHICLE_CATEGORY'] == cat)]
        for cluster in kmeans_clusters:
            vehiclecounts.append(sum(cluster['VEHICLE_CATEGORY'] == cat))
        kmeans_vehiclecounts[cat] = vehiclecounts

    kmeans_vehiclecounts.index = rowlabels
    kmeans_vehiclecounts.to_csv("kmeans_vehiclecounts.csv")

    # Getting average injury and vehicle damage levels per cluster per environmental condition
    for i in targetvars:
        for j in featurevars2:
            df = pd.DataFrame()
            for cat in kmeans_df[j].unique():
                col = [kmeans_df[kmeans_df[j] == cat][i].mean()]
                for cluster in kmeans_clusters:
                    col.append(cluster[cluster[j] == cat][i].mean())
                df[cat] = col
            df.index = rowlabels
            df.to_csv("kmeans_"+i+"_by_"+j+".csv")

def aggcls_clustering(df, features, n):
    # Agglomerative Clustering analysis

    normalized = normalize(df, features)

    aggcls = AgglomerativeClustering(n_clusters=n, linkage="ward").fit(normalized)
    aggcls_df = df.copy()

    rowlabels = ["Aggregate"]
    aggcls_df['cluster'] = aggcls.labels_
    aggcls_df.to_csv("aggcls_df.csv")

    aggcls_agg = aggcls_df[['cluster','AVERAGE_INJ_LEVEL','VEHICLE_DAMAGE_LEVEL']].groupby(['cluster']).mean()
    aggcls_agg['count'] = aggcls_df[['cluster','ACCIDENT_NO']].groupby(['cluster']).count()
    aggcls_agg.to_csv("aggcls_aggregate_stats.csv")

    aggcls_clusters = []
    for i in range(0,n):
        aggcls_clusters.append(aggcls_df[aggcls_df['cluster']==i])
        rowlabels.append("Cluster "+str(i))

    # Getting vehicle counts per cluster
    aggcls_vehiclecounts = pd.DataFrame()
    for cat in aggcls_df['VEHICLE_CATEGORY'].unique():
        vehiclecounts = [sum(aggcls_df['VEHICLE_CATEGORY'] == cat)]
        for cluster in aggcls_clusters:
            vehiclecounts.append(sum(cluster['VEHICLE_CATEGORY'] == cat))
        aggcls_vehiclecounts[cat] = vehiclecounts

    aggcls_vehiclecounts.index = rowlabels
    aggcls_vehiclecounts.to_csv("aggcls_vehiclecounts.csv")

    # Getting average injury and vehicle damage levels per cluster per environmental condition
    for i in targetvars:
        for j in featurevars2:
            df = pd.DataFrame()
            for cat in aggcls_df[j].unique():
                col = [aggcls_df[aggcls_df[j] == cat][i].mean()]
                for cluster in aggcls_clusters:
                    col.append(cluster[cluster[j] == cat][i].mean())
                df[cat] = col
            df.index = rowlabels
            df.to_csv("aggcls_"+i+"_by_"+j+".csv")

def kmodes_clustering(df, features, k):
    # KModes Clustering analysis

    kmodes = KModes(n_clusters=k, random_state=20250523).fit(df[features])
    kmodes_df = df.copy()
    kmodes_df['cluster'] = kmodes.labels_
    kmodes_df.to_csv("kmodes_df.csv")

    kmodes_cluster_centroids = pd.DataFrame(kmodes.cluster_centroids_)
    kmodes_cluster_centroids.to_csv("kmodes_cluster_centroids.csv")

    kmodes_agg = kmodes_df[['cluster','AVERAGE_INJ_LEVEL','VEHICLE_DAMAGE_LEVEL']].groupby(['cluster']).mean()
    kmodes_agg['count'] = kmodes_df[['cluster','ACCIDENT_NO']].groupby(['cluster']).count()
    kmodes_agg.to_csv("kmodes_aggregate_stats.csv")

    rowlabels = ["Aggregate"]
    kmodes_clusters = []
    for i in range(0,k):
        kmodes_clusters.append(kmodes_df[kmodes_df['cluster']==i])
        rowlabels.append("Cluster "+str(i))

    # Getting vehicle percents per cluster
    # Using %s over counts since KModes clusters denote road/environment conditions and not vehicle clusters
    kmodes_categorypercents = pd.DataFrame()

    for cat in kmodes_df['VEHICLE_CATEGORY'].unique():
        percents = [sum(kmodes_df['VEHICLE_CATEGORY'] == cat)/kmodes_df.shape[0]]
        for cluster in kmodes_clusters:
            percents.append(sum(cluster['VEHICLE_CATEGORY'] == cat)/cluster.shape[0])
        kmodes_categorypercents[cat] = percents

    kmodes_categorypercents.index = rowlabels
    kmodes_categorypercents.to_csv("kmodes_vehiclepercents.csv")

    # Getting average injury and vehicle damage levels per cluster per vehicle category
    kmodes_averageinj = pd.DataFrame()
    kmodes_vehicledmg = pd.DataFrame()

    for cat in kmodes_df['VEHICLE_CATEGORY'].unique():
        averageinj = [kmodes_df[kmodes_df['VEHICLE_CATEGORY'] == cat]['AVERAGE_INJ_LEVEL'].mean()]
        vehicledmg = [kmodes_df[kmodes_df['VEHICLE_CATEGORY'] == cat]['VEHICLE_DAMAGE_LEVEL'].mean()]
        for cluster in kmodes_clusters:
            averageinj.append(cluster[cluster['VEHICLE_CATEGORY'] == cat]['AVERAGE_INJ_LEVEL'].mean())
            vehicledmg.append(cluster[cluster['VEHICLE_CATEGORY'] == cat]['VEHICLE_DAMAGE_LEVEL'].mean())
        kmodes_averageinj[cat] = averageinj
        kmodes_vehicledmg[cat] = vehicledmg

    kmodes_averageinj.index = rowlabels
    kmodes_averageinj.to_csv("kmodes_AVERAGE_INJ_LEVEL.csv")
    kmodes_vehicledmg.index = rowlabels
    kmodes_vehicledmg.to_csv("kmodes_VEHICLE_DAMAGE_LEVEL.csv")

# Function derived from SKLearn Documentation example
# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# kmeans_elbow(everything_df, featurevars1, 3)
# kmodes_elbow(everything_df, featurevars2, 5)

# aggcls_dendrogram(aggcls_sample_df, featurevars1)

# kmeans_clustering(everything_df, featurevars1, 3)
# aggcls_clustering(aggcls_sample_df, featurevars1, 5)
# kmodes_clustering(everything_df, featurevars2, 5)
