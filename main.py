import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('spotify_songs.csv')

print("Columnas que tiene el archivo")
print(list(df.columns))

print("Informacion General")
df.info()

print('-' * 75)
print('Cantidad de canciones por genero')
genre_counts = df['playlist_genre'].value_counts()
print(genre_counts.to_string(header=False))

print('-' * 75)
print("Media de popularidad por cancion:", df['track_popularity'].mean())

print('-' * 75)
print("Media de duracion por cancion:", df['duration_ms'].mean())

print('-' * 75)
print("Media de bailabilidad por cancion:", df['danceability'].mean())

# Create a 2x3 subplot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1
df['track_popularity'].plot(kind='hist', bins=20, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Distribución de la popularidad')
axes[0, 0].set_xlabel('Popularidad')
axes[0, 0].set_ylabel('Frecuencia')

# Plot 2
df['track_popularity'].plot(kind='box', ax=axes[0, 1], color='salmon')
axes[0, 1].set_title('Diagrama de caja de Popularidad')
axes[0, 1].set_xlabel('Popularidad')

# Plot 3
df['duration_ms'].plot(kind='hist', bins=20, ax=axes[0, 2], color='lightgreen')
axes[0, 2].set_title('Distribución de la duracion')
axes[0, 2].set_xlabel('Duracion (ms)')
axes[0, 2].set_ylabel('Frecuencia')

# Plot 4
df['duration_ms'].plot(kind='box', ax=axes[1, 0], color='gold')
axes[1, 0].set_title('Diagrama de caja de Duracion')
axes[1, 0].set_xlabel('Duracion (ms)')

# Plot 5
df['danceability'].plot(kind='hist', bins=20, ax=axes[1, 1], color='lightcoral')
axes[1, 1].set_title('Distribución de la bailabilidad')
axes[1, 1].set_ylabel('Frecuencia')

# Plot 6
df['danceability'].plot(kind='box', ax=axes[1, 2], color='mediumorchid')
axes[1, 2].set_title('Diagrama de caja de Bailabilidad')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plots
plt.show()

# Correlation Heatmap
# Identify numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Correlation Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='viridis')
plt.title('Mapa de Calor de Correlaciones')
plt.show()

# Pair Plot
# Creating a new categorical variable as hue
df['custom_hue'] = pd.cut(df['loudness'], bins=3, labels=['Low', 'Medium', 'High'])

# Pair Plot with Custom Hue
plt.figure()
sns.pairplot(df[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'custom_hue']],
             hue='custom_hue',
             height=6,
             palette='Set2',
             diag_kind='kde',
             markers=["o", "s", "D"],
             plot_kws={'alpha': 0.7}
            )
plt.title('Pair Plot de Features de Audio')
plt.show()



# Creating a new categorical variable as hue
df['custom_hue'] = pd.cut(df['danceability'], bins=1, labels=['Bailabilidad por genero'])

# Box Plot with Custom Hue
plt.figure(figsize=(12, 10))
sns.boxplot(x='playlist_genre', y='danceability', data=df, hue='custom_hue', palette='pastel', dodge=True)
plt.title('Box Plot de Bailabilidad por Genero')
plt.xticks(rotation=45)
plt.show()


# Count Plot of Genres
plt.figure(figsize=(8, 6))
sns.countplot(x='playlist_genre', data=df, color='magenta')
plt.title('Distribucion de Generos')
plt.xticks(rotation=45)
plt.yticks(np.arange(0, 6001, 500))
plt.xlabel('Genero')
plt.ylabel('Numero de Canciones')
plt.show()


# Scatter Plot of Popularity vs. Energy
plt.figure(figsize=(14, 12))
plt.scatter(x='track_popularity', y='energy', data=df, alpha=0.5, color='skyblue')
plt.title('Popularidad vs. Energia')
plt.xlabel('Popularidad')
plt.ylabel('Energia')
plt.show()

# Histogram of Release Dates
# df['track_album_release_date'] = pd.to_datetime(df['track_album_release_date'])
# df['release_year'] = df['track_album_release_date'].dt.year
#
# axes[1, 2].hist(df['release_year'], bins=20, edgecolor='black')
# axes[1, 2].set_title('Distribution of Track Releases Over Time')
# axes[1, 2].set_xlabel('Release Year')
# axes[1, 2].set_ylabel('Number of Tracks')
