# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 2: Load dataset from Downloads
csv_path = r"C:\Users\GANESH\Downloads\retail_customers.csv"
df = pd.read_csv(csv_path)

# Step 3: Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Step 4: Fit KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)  # You can try 2-6 clusters
kmeans.fit(scaled_data)
df['cluster'] = kmeans.labels_

# Step 5: Visualize clusters
sns.pairplot(df, hue='cluster', palette='Set1', diag_kind='kde')
plt.suptitle("Customer Segments (K-Means)", y=1.02)
plt.show()

# Optional: View cluster centers (scaled space)
print("Cluster Centers (scaled features):\n", kmeans.cluster_centers_)
