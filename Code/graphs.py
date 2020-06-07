import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Visualise results
x_cats = ['Weekly', 'Daily', 'Hourly', 'Half Hourly']

mlp_cust_best = [38.31, 61.85, 77.01, 86.66]
mlp_cust_worst = [2.28, 1.93, 1.78, 1.95]
mlp_post_best = [23.74, 32.8, 45.46, 34.55]
mlp_post_worst = [11.48, 11.31, 10.2, 9.85]

plot_mlp = pd.DataFrame({
    'cats': x_cats,
    'MLP Customer best case': mlp_cust_best,
    'MLP Customer worst case': mlp_cust_worst,
    'MLP Postcode best case': mlp_post_best,
    'MLP Postcode worst case': mlp_post_worst
})
sns.lineplot(x='cats', y='value', hue='variable', sort=False,
             data=pd.melt(plot_mlp, ['cats']))
plt.title("Multilayer Perceptron Results")
plt.xlabel('Transaction Frequency')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

rf_cust_best = [98.02, 98.75, 100.00, 100.00]
rf_cust_worst = [0.37, 0.51, 0.71, 0.58]
rf_post_best = [97.87, 98.70, 100.00, 100.00]
rf_post_worst = [0.47, 0.65, 0.87, 0.75]

plot_rf = pd.DataFrame({
    'cats': x_cats,
    'RF Customer best case': rf_cust_best,
    'RF Customer worst case': rf_cust_worst,
    'RF Postcode best case': rf_post_best,
    'RF Postcde worst case': rf_post_worst
})
sns.lineplot(x='cats', y='value', hue='variable', sort=False,
             data=pd.melt(plot_rf, ['cats']))
plt.title("Random Forest Results")
plt.xlabel('Transaction Frequency')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

knn_cust_best = [10.68, 12.75, 14.08, 8.55]
knn_cust_worst = [0.95, 1.86, 2.22, 1.21]
knn_post_best = [13.84, 15.75, 17.56, 15.78]
knn_post_worst = [10.30, 10.75, 9.22, 8.37]

plot_knn = pd.DataFrame({
    'cats': x_cats,
    'KNN Customer best case': knn_cust_best,
    'KNN Customer worst case': knn_cust_worst,
    'KNN Postcode best case': knn_post_best,
    'KNN Postcode worst case': knn_post_worst
})
sns.lineplot(x='cats', y='value', hue='variable', sort=False,
             data=pd.melt(plot_knn, ['cats']))
plt.title("K-Nearest Neighbours Results")
plt.xlabel('Transaction Frequency')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

mlp_cust_best = [29.97, 31.94, 27.94, 27.87]
mlp_cust_worst = [5.4, 3.65, 4.95, 5.66]
mlp_post_best = [24.64, 32.71, 35.37, 35.11]
mlp_post_worst = [5.23, 10.4, 10.05, 8.03]

plot_mlp_spread = pd.DataFrame({
    'cats': x_cats,
    'MLP Customer best case': mlp_cust_best,
    'MLP Customer worst case': mlp_cust_worst,
    'MLP Postcode best case': mlp_post_best,
    'MLP Postcode worst case': mlp_post_worst
})
sns.lineplot(x='cats', y='value', hue='variable', sort=False,
             data=pd.melt(plot_mlp_spread, ['cats']))
plt.title("Multilayer Perceptron Spread")
plt.xlabel('Transaction Frequency')
plt.ylabel('Spread (STDEV %)')
plt.legend()
plt.show()
