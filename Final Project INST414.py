import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.metrics import chi2_kernel, hellinger_kernel
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Loads neighborhood-level data
demographics = pd.read_csv("neighborhood_demographics.csv", index_col="neighborhood_id")
housing = pd.read_csv("neighborhood_housing.csv", index_col="neighborhood_id")
amenities = pd.read_csv("neighborhood_amenities.csv", index_col="neighborhood_id")
geo_boundaries = gpd.read_file("neighborhood_boundaries.shp")

# Customizes distance metrics for different data types
def chi_square_dist(x, y):
   return chi2_kernel(x.values.reshape(1, -1), y.values.reshape(1, -1))

def hellinger_dist(x, y):
   return hellinger_kernel(x.values.reshape(1, -1), y.values.reshape(1, -1))

def gowers_dist(x, y):
   cols = x.columns
   weighted_dists = []
   for col in cols:
       if x[col].dtype == 'object':
           weighted_dists.append(hellinger_dist(x[col], y[col]))
       else:
           weighted_dists.append(np.abs(x[col] - y[col]))
   return np.mean(weighted_dists)

# Calculates the component similarities  
demo_sim = cdist(demographics, demographics, metric=chi_square_dist)
race_sim = cdist(pd.get_dummies(demographics["race"]), pd.get_dummies(demographics["race"]), metric=hellinger_dist)    
housing_sim = cdist(MinMaxScaler().fit_transform(housing), MinMaxScaler().fit_transform(housing))
amenities_sim = cdist(amenities, amenities, metric=gowers_dist)

# Combines into holistic similarity score
pca = PCA()
combined_data = pd.concat([demographics, housing, amenities], axis=1)
pca.fit(combined_data)
weights = pca.explained_variance_ratio_
similarity_matrix = weights[0] * demo_sim + weights[1] * race_sim + weights[2] * housing_sim + weights[3] * amenities_sim   

# Clustering and Visualization
clustering = AgglomerativeClustering().fit(similarity_matrix)
plt.figure(figsize=(10, 7))
plt.title("Hierarchical Clustering Dendrogram")
dendrogram = plt.dendrogram(clustering.children_, labels=geo_boundaries.index, orientation='top')
plt.show()

silhouette_scores = []
k_range = range(3, 15)
for k in k_range:
   kmeans = KMeans(n_clusters=k).fit(similarity_matrix)
   silhouette_scores.append(silhouette_score(similarity_matrix, kmeans.labels_))

plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for K-Means Clustering')
plt.show()

# Cluster Profiling and Visualization
optimal_k = k_range[np.argmax(silhouette_scores)]
kmeans = KMeans(n_clusters=optimal_k).fit(similarity_matrix)
geo_boundaries['cluster'] = kmeans.labels_

# Visualizations
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
plt.subplots_adjust(hspace=0.4)

# Cluster distribution on map
geo_boundaries.plot(column='cluster', cmap='viridis', ax=axes[0, 0], legend=True)
axes[0, 0].set_title('Neighborhood Clusters')

# Radar chart of the cluster profiles
cluster_profiles = geo_boundaries.groupby('cluster')[['median_income', 'housing_affordability', 'amenity_access']].mean()
radar = pd.melt(cluster_profiles.reset_index(), id_vars='cluster', value_vars=['median_income', 'housing_affordability', 'amenity_access'])
sns.lineplot(x='variable', y='value', hue='cluster', data=radar, ax=axes[0, 1])
axes[0, 1].set_title('Cluster Profiles')

# Heatmap of the amenity access
amenity_access = geo_boundaries.groupby('cluster')['amenity_access'].mean().reset_index()
sns.heatmap(amenity_access.pivot(index='cluster', columns='amenity_access', values='amenity_access'), annot=True, cmap='YlGnBu', ax=axes[1, 0])
axes[1, 0].set_title('Amenity Access by Cluster')

# Housing affordability by the cluster
housing_affordability = geo_boundaries.groupby('cluster')['housing_affordability'].mean().reset_index()
sns.barplot(x='cluster', y='housing_affordability', data=housing_affordability, ax=axes[1, 1])
axes[1, 1].set_title('Housing Affordability by Cluster')

plt.tight_layout()
plt.show()
