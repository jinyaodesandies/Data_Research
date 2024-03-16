import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
file_path = "/Users/ISASC_ST/Documents/Data Project/americans_by_descent.csv"
df = pd.read_csv(file_path)
df['ethnicity_count'] = df.iloc[:, 2:].sum(axis=1)
ethnicity_counts = df.iloc[:, 2:].sum().sort_values(ascending=False)
top_ethnicities = ethnicity_counts.head(10).index
corr_matrix = df[top_ethnicities].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Top Ethnicities')
plt.show()