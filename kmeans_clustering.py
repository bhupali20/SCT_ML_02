import pandas as pd  # Importing pandas for data manipulation and analysis
import matplotlib.pyplot as plt  # Importing matplotlib for plotting graphs
import seaborn as sns  # Importing seaborn for advanced data visualization
from sklearn.cluster import KMeans  # Importing KMeans algorithm from sklearn for clustering

# Load the dataset
# Specify the file path for the Mall Customers dataset and load it into a pandas DataFrame
file_path = r'C:\Users\LENOVO\OneDrive\Desktop\SkillsCraft Internship\Mall_Customers.csv'
data = pd.read_csv(file_path)

# Select relevant features
# Extract the columns 'Annual Income (k$)' and 'Spending Score (1-100)' from the dataset
# These features will be used to cluster the customers
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Determine the optimal number of clusters using the Elbow Method
# The Elbow Method helps us decide the optimal number of clusters by plotting the within-cluster sum of squares (WCSS)
wcss = []  # Initialize an empty list to store the WCSS for each number of clusters

# Loop through different numbers of clusters (from 1 to 10)
for i in range(1, 11):
    # Apply KMeans with the current number of clusters (i)
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)  # Fit the KMeans model on the selected features (X)
    wcss.append(kmeans.inertia_)  # Append the WCSS (inertia) of the current model to the list

# Plot the Elbow Method graph
# The plot will show WCSS values for different numbers of clusters
# We look for an "elbow point" where the WCSS starts decreasing more slowly
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-means clustering
# Based on the Elbow Method, choose 5 clusters as the optimal number
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=42)
# Fit the KMeans model and predict the clusters for each customer
y_kmeans = kmeans.fit_predict(X)
