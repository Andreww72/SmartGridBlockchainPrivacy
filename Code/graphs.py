import os
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns


def graphs_obj1():
    # Visualise some weekly data
    os.chdir("../BlockchainData/weekly")
    df37 = pd.read_csv(f"{37}_blockchain.csv", header=0)
    df59 = pd.read_csv(f"{59}_blockchain.csv", header=0)
    df102 = pd.read_csv(f"{102}_blockchain.csv", header=0)
    df226 = pd.read_csv(f"{226}_blockchain.csv", header=0)

    data_preproc_use = pd.DataFrame({
        'Time': list(range(0, df37.query("Type=='GC'")['Amount'].size)),
        'C 37': df37.query("Type=='GC'")['Amount'],
        'C 59': df59.query("Type=='GC'")['Amount'],
        'C 102': df102.query("Type=='GC'")['Amount'],
        'C 226': df226.query("Type=='GC'")['Amount']
    })
    data_preproc_gen = pd.DataFrame({
        'Time': list(range(0, df37.query("Type=='GG'")['Amount'].size)),
        'C 37': df37.query("Type=='GG'")['Amount'],
        'C 59': df59.query("Type=='GG'")['Amount'],
        'C 102': df102.query("Type=='GG'")['Amount'],
        'C 226': df226.query("Type=='GG'")['Amount']
    })

    plt.subplots(1, 2)
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    sns.lineplot(x='Time', y='value', hue='variable', ax=ax1,
                 data=pd.melt(data_preproc_use, ['Time']))
    ax1.set_title("Weekly consumption")
    ax1.set_xlabel('Time (weeks)')
    ax1.set_ylabel('Energy (kWh)')
    ax1.legend()

    sns.lineplot(x='Time', y='value', hue='variable', ax=ax2,
                 data=pd.melt(data_preproc_gen, ['Time']))
    ax2.set_title("Weekly generation")
    ax2.set_xlabel('Time (weeks)')
    ax2.set_ylabel('Energy (kWh)')
    ax2.legend()
    plt.show()

    # Visualise some daily data
    os.chdir("../daily")
    df37 = pd.read_csv(f"{37}_blockchain.csv", header=0)
    df59 = pd.read_csv(f"{59}_blockchain.csv", header=0)
    df102 = pd.read_csv(f"{102}_blockchain.csv", header=0)
    df226 = pd.read_csv(f"{226}_blockchain.csv", header=0)

    data_preproc_use = pd.DataFrame({
        'Time': list(range(0, df37.query("Type=='GC'")['Amount'].size)),
        'C 37': df37.query("Type=='GC'")['Amount'],
        'C 59': df59.query("Type=='GC'")['Amount'],
        'C 102': df102.query("Type=='GC'")['Amount'],
        'C 226': df226.query("Type=='GC'")['Amount']
    })
    data_preproc_gen = pd.DataFrame({
        'Time': list(range(0, df37.query("Type=='GG'")['Amount'].size)),
        'C 37': df37.query("Type=='GG'")['Amount'],
        'C 59': df59.query("Type=='GG'")['Amount'],
        'C 102': df102.query("Type=='GG'")['Amount'],
        'C 226': df226.query("Type=='GG'")['Amount']
    })

    plt.subplots(1, 2)
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    sns.lineplot(x='Time', y='value', hue='variable', ax=ax1,
                 data=pd.melt(data_preproc_use, ['Time']))
    ax1.set_title("Daily consumption")
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Energy (kWh)')
    ax1.legend()

    sns.lineplot(x='Time', y='value', hue='variable', ax=ax2,
                 data=pd.melt(data_preproc_gen, ['Time']))
    ax2.set_title("Daily generation")
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Energy (kWh)')
    ax2.legend()
    plt.show()

    # Visualise a single day
    os.chdir("../half_hourly")
    df37 = pd.read_csv(f"{37}_blockchain.csv", header=0)
    df59 = pd.read_csv(f"{59}_blockchain.csv", header=0)
    df102 = pd.read_csv(f"{102}_blockchain.csv", header=0)
    df226 = pd.read_csv(f"{226}_blockchain.csv", header=0)

    data_preproc_use = pd.DataFrame({
        'Time': list(range(0, df37.query("Type=='GC'")['Amount'][:48].size)),
        'C 37': df37.query("Type=='GC'")['Amount'][:48],
        'C 59': df59.query("Type=='GC'")['Amount'][:48],
        'C 102': df102.query("Type=='GC'")['Amount'][:48],
        'C 226': df226.query("Type=='GC'")['Amount'][:48]
    })
    data_preproc_gen = pd.DataFrame({
        'Time': list(range(0, df37.query("Type=='GG'")['Amount'][:48].size)),
        'C 37': df37.query("Type=='GG'")['Amount'][:48],
        'C 59': df59.query("Type=='GG'")['Amount'][:48],
        'C 102': df102.query("Type=='GG'")['Amount'][:48],
        'C 226': df226.query("Type=='GG'")['Amount'][:48]
    })

    plt.subplots(1, 2)
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    sns.lineplot(x='Time', y='value', hue='variable', ax=ax1,
                 data=pd.melt(data_preproc_use, ['Time']))
    ax1.set_title("Day consumption")
    ax1.set_xlabel('Time (half hours)')
    ax1.set_ylabel('Energy (kWh)')
    ax1.legend()

    sns.lineplot(x='Time', y='value', hue='variable', ax=ax2,
                 data=pd.melt(data_preproc_gen, ['Time']))
    ax2.set_title("Day generation")
    ax2.set_xlabel('Time (half hours)')
    ax2.set_ylabel('Energy (kWh)')
    ax2.legend()
    plt.show()


def graphs_obj2():
    # Visualise results
    x_cats = ['Weekly', 'Weekly', 'Weekly',
              'Daily', 'Daily', 'Daily',
              'Hourly', 'Hourly',
              'Half Hourly', 'Half Hourly']

    # Enter results
    mlp_cust_lpc = [46.04, 45.79, 45.32, 64.51, 64.90, 63.12, 75.46, 75.05, 73.86, 73.59]
    mlp_cust_lpp = [47.81, 47.05, 48.79, 64.81, 67.02, 66.51, 76.58, 77.47, 76.14, 76.05]
    mlp_cust_aol = [02.24, 02.24, 02.37, 01.94, 01.94, 01.91, 01.81, 01.85, 01.85, 01.86]
    mlp_post_lpc = [22.11, 22.35, 23.10, 31.97, 31.95, 30.44, 40.42, 40.24, 36.92, 36.44]
    mlp_post_lpp = [26.56, 24.12, 24.32, 36.27, 31.10, 35.03, 43.41, 43.01, 38.57, 38.82]
    mlp_post_aol = [11.81, 11.43, 11.21, 11.34, 11.25, 11.33, 10.34, 10.05, 09.85, 09.86]

    cnn_cust_lpc = [57.01, 58.95, 58.12, 69.88, 70.72, 69.13, 78.42, 79.41, 84.06, 83.89]
    cnn_cust_lpp = [58.04, 59.66, 59.93, 71.00, 71.15, 70.99, 79.34, 79.45, 84.45, 85.97]
    cnn_cust_aol = [02.60, 02.42, 02.38, 02.38, 02.37, 02.42, 02.34, 02.32, 02.52, 02.44]
    cnn_post_lpc = [42.90, 42.42, 43.23, 52.49, 54.18, 55.22, 60.52, 63.16, 61.82, 61.73]
    cnn_post_lpp = [45.15, 46.18, 45.61, 55.38, 56.66, 55.28, 62.67, 62.90, 63.06, 63.82]
    cnn_post_aol = [11.69, 11.21, 11.49, 11.46, 11.46, 11.50, 10.11, 10.12, 09.99, 10.07]

    rfc_cust_lpc = [61.12, 63.04, 63.73, 63.96, 62.21, 63.12, 55.81, 57.66, 54.97, 58.99]
    rfc_cust_lpp = [64.41, 64.73, 65.30, 63.44, 65.45, 66.52, 58.55, 56.23, 56.92, 56.16]
    rfc_cust_aol = [02.18, 02.03, 02.21, 02.64, 02.66, 02.59, 02.54, 02.55, 02.69, 02.66]
    rfc_post_lpc = [38.17, 38.33, 37.18, 36.64, 36.85, 37.34, 33.06, 33.13, 33.80, 34.00]
    rfc_post_lpp = [40.70, 40.18, 41.20, 36.81, 36.62, 36.14, 33.47, 32.99, 33.02, 32.76]
    rfc_post_aol = [11.91, 11.88, 11.79, 11.66, 11.65, 11.70, 10.40, 10.46, 10.30, 10.30]

    knn_cust_lpc = [35.87, 36.32, 35.73, 47.89, 47.73, 47.84, 54.51, 54.52, 50.42, 50.80]
    knn_cust_lpp = [34.98, 34.79, 34.96, 47.79, 47.76, 47.93, 54.51, 54.56, 51.45, 51.46]
    knn_cust_aol = [00.97, 00.95, 00.93, 01.87, 01.84, 01.87, 02.21, 02.22, 02.39, 02.39]
    knn_post_lpc = [23.34, 22.16, 23.69, 37.78, 37.86, 38.19, 43.36, 43.41, 38.55, 38.57]
    knn_post_lpp = [26.43, 26.38, 26.51, 38.04, 37.78, 37.95, 43.46, 43.46, 39.57, 39.57]
    knn_post_aol = [10.46, 10.47, 10.48, 10.71, 10.64, 10.71, 09.75, 09.79, 09.52, 09.52]

    # MLP RESULTS
    plot_mlp_cust = pd.DataFrame({
        'cats': x_cats,
        'MLP LPC': mlp_cust_lpc,
        'MLP LPP': mlp_cust_lpp,
        'MLP AOL': mlp_cust_aol
    })
    plot_mlp_post = pd.DataFrame({
        'cats': x_cats,
        'MLP LPC': mlp_post_lpc,
        'MLP LPP': mlp_post_lpp,
        'MLP AOL': mlp_post_aol
    })
    bar_graph(["Multilayer Percecptron - Customer",
               "Multilayer Percecptron - Postcode"],
              [plot_mlp_cust, plot_mlp_post])

    # CNN RESULTS
    plot_cnn_cust = pd.DataFrame({
        'cats': x_cats,
        'CNN LPC': cnn_cust_lpc,
        'CNN LPP': cnn_cust_lpp,
        'CNN AOL': cnn_cust_aol
    })
    plot_cnn_post = pd.DataFrame({
        'cats': x_cats,
        'CNN LPC': cnn_post_lpc,
        'CNN LPP': cnn_post_lpp,
        'CNN AOL': cnn_post_aol
    })
    bar_graph(["Convolutional Neural Network - Customer",
               "Convolutional Neural Network - Postcode"],
              [plot_cnn_cust, plot_cnn_post])

    # RFC RESULTS
    plot_rfc_cust = pd.DataFrame({
        'cats': x_cats,
        'RFC LPC': rfc_cust_lpc,
        'RFC LPP': rfc_cust_lpp,
        'RFC AOL': rfc_cust_aol
    })
    plot_rfc_post = pd.DataFrame({
        'cats': x_cats,
        'RFC LPC': rfc_post_lpc,
        'RFC LPP': rfc_post_lpp,
        'RFC AOL': rfc_post_aol
    })
    bar_graph(["Random Forest Classifier - Customer",
               "Random Forest Classifier - Postcode"],
              [plot_rfc_cust, plot_rfc_post])

    # KNN RESULTS
    plot_knn_cust = pd.DataFrame({
        'cats': x_cats,
        'KNN LPC': knn_cust_lpc,
        'KNN LPP': knn_cust_lpp,
        'KNN AOL': knn_cust_aol
    })
    plot_knn_post = pd.DataFrame({
        'cats': x_cats,
        'KNN LPC': knn_post_lpc,
        'KNN LPP': knn_post_lpp,
        'KNN AOL': knn_post_aol
    })
    bar_graph(["K-Nearest Neighbours - Customer",
               "K-Nearest Neighbours - Postcode"],
              [plot_knn_cust, plot_knn_post])

    # Compare ledger per customer results
    plot_comp_cust = pd.DataFrame({
        'cats': x_cats,
        'MLP': mlp_cust_lpc,
        'CNN': cnn_cust_lpc,
        'RFC': rfc_cust_lpc,
        'KNN': knn_cust_lpc
    })
    plot_comp_post = pd.DataFrame({
        'cats': x_cats,
        'MLP': mlp_post_lpc,
        'CNN': cnn_post_lpc,
        'RFC': rfc_post_lpc,
        'KNN': knn_post_lpc
    })
    bar_graph(["Compare LPC - Customer",
               "Compare LPC - Postcode"],
              [plot_comp_cust, plot_comp_post])

    # Compare ledger per postcode results
    plot_comp_cust = pd.DataFrame({
        'cats': x_cats,
        'MLP': mlp_cust_lpp,
        'CNN': cnn_cust_lpp,
        'RFC': rfc_cust_lpp,
        'KNN': knn_cust_lpp
    })
    plot_comp_post = pd.DataFrame({
        'cats': x_cats,
        'MLP': mlp_post_lpp,
        'CNN': cnn_post_lpp,
        'RFC': rfc_post_lpp,
        'KNN': knn_post_lpp
    })
    bar_graph(["Compare LPP - Customer",
               "Compare LPP - Postcode"],
              [plot_comp_cust, plot_comp_post])

    # Compare AOL results
    plot_comp_cust = pd.DataFrame({
        'cats': x_cats,
        'MLP': mlp_cust_aol,
        'CNN': cnn_cust_aol,
        'RFC': rfc_cust_aol,
        'KNN': knn_cust_aol
    })
    plot_comp_post = pd.DataFrame({
        'cats': x_cats,
        'MLP': mlp_post_aol,
        'CNN': cnn_post_aol,
        'RFC': rfc_post_aol,
        'KNN': knn_post_aol
    })
    bar_graph(["Compare AOL - Customer",
               "Compare AOL - Postcode"],
              [plot_comp_cust, plot_comp_post], ylim_max=20)


def graphs_obj3():
    # Visualise results
    x_cats = ['Weekly', 'Daily', 'Hourly', 'Half Hourly']

    # Data copied from graphs_obj2 for delta to solar results
    cnn_cust_lpc = [58.03, 69.91, 78.92, 83.98]
    cnn_cust_lpp = [59.21, 71.05, 79.40, 85.21]
    cnn_cust_aol = [2.47, 2.39, 2.33, 2.48]
    cnn_post_lpc = [42.85, 53.96, 61.84, 61.78]
    cnn_post_lpp = [45.65, 55.77, 62.79, 63.44]
    cnn_post_aol = [11.46, 11.47, 10.12, 10.03]

    rfc_cust_lpc = [62.63, 63.10, 56.74, 56.98]
    rfc_cust_lpp = [64.81, 65.14, 57.39, 56.54]
    rfc_cust_aol = [2.14, 2.63, 2.55, 2.68]
    rfc_post_lpc = [39.23, 36.94, 33.10, 33.90]
    rfc_post_lpp = [39.69, 36.52, 33.23, 32.89]
    rfc_post_aol = [11.86, 11.67, 10.43, 10.30]

    # Solar results
    cnn_cust_lpc_s = [67.98, 76.36, 82.33, 81.81]
    cnn_cust_lpp_s = [67.87, 76.58, 83.49, 82.68]
    cnn_cust_aol_s = [3.05, 2.84, 3.31, 3.50]
    cnn_post_lpc_s = [55.06, 62.49, 66.14, 64.85]
    cnn_post_lpp_s = [52.27, 59.47, 65.06, 66.98]
    cnn_post_aol_s = [12.04, 11.94, 10.55, 10.57]

    rfc_cust_lpc_s = [71.77, 72.90, 70.56, 67.62]
    rfc_cust_lpp_s = [75.89, 74.55, 71.61, 68.81]
    rfc_cust_aol_s = [2.84, 3.70, 5.08, 5.17]
    rfc_post_lpc_s = [57.69, 52.11, 47.90, 48.52]
    rfc_post_lpp_s = [58.38, 52.45, 47.77, 48.51]
    rfc_post_aol_s = [14.43, 13.03, 12.55, 11.48]

    bright_cols = sns.color_palette("bright", 10)
    dark_cols = sns.color_palette("colorblind", 10)
    palette = [dark_cols[1], bright_cols[1], dark_cols[2], bright_cols[2]]

    # Compare ledger per customer results
    plot_comp_cust = pd.DataFrame({'cats': x_cats, 'CNN': cnn_cust_lpc, 'CNN solar': cnn_cust_lpc_s,
                                   'RFC': rfc_cust_lpc, 'RFC solar': rfc_cust_lpc_s})
    plot_comp_post = pd.DataFrame({'cats': x_cats, 'CNN': cnn_post_lpc, 'CNN solar': cnn_post_lpc_s,
                                   'RFC': rfc_post_lpc, 'RFC solar': rfc_post_lpc_s})
    bar_graph(["Compare LPC - Customer", "Compare LPC - Postcode"], [plot_comp_cust, plot_comp_post], palette)

    # Compare ledger per postcode results
    plot_comp_cust = pd.DataFrame({'cats': x_cats, 'CNN': cnn_cust_lpp, 'CNN solar': cnn_cust_lpp_s,
                                   'RFC': rfc_cust_lpp, 'RFC solar': rfc_cust_lpp_s})
    plot_comp_post = pd.DataFrame({'cats': x_cats, 'CNN': cnn_post_lpp, 'CNN solar': cnn_post_lpp_s,
                                   'RFC': rfc_post_lpp, 'RFC solar': rfc_post_lpp_s})
    bar_graph(["Compare LPP - Customer", "Compare LPP - Postcode"], [plot_comp_cust, plot_comp_post], palette)

    # Compare AOL results
    plot_comp_cust = pd.DataFrame({'cats': x_cats, 'CNN': cnn_cust_aol, 'CNN solar': cnn_cust_aol_s,
                                   'RFC': rfc_cust_aol, 'RFC solar': rfc_cust_aol_s})
    plot_comp_post = pd.DataFrame({'cats': x_cats, 'CNN': cnn_post_aol, 'CNN solar': cnn_post_aol_s,
                                   'RFC': rfc_post_aol, 'RFC solar': rfc_post_aol_s})
    bar_graph(["Compare AOL - Customer", "Compare AOL - Postcode"], [plot_comp_cust, plot_comp_post], palette, ylim_max=20)

    # Net export correlation distribution
    data = [58, 6, 20, 7, 78, 55, 21, 83, 3, 24, 26, 50, 93, 47, 11, 6, 20, 3, 32, 9, 71, 71, 61, 61, 76, 21, 98, 37,
            88, 36, 4, 32, 91, 47, 73, 79, 44, 82, 39, 59, 92, 96, 64, 23, 74, 46, 84, 59, 59, 13, 18, 42, 1, 99, 58,
            22, 13, 43, 55, 47, 34, 14, 24, 12, 17, 96, 21, 67, 65, 56, 11, 22, 5, 85, 55, 48, 31, 15, 96, 7, 6, 67, 87,
            6, 30, 65, 10, 29, 3, 88, 7, 27, 17, 7, 76, 20, 96, 2, 70, 0, 66, 56, 73, 39, 63, 57, 69, 97, 17, 97, 32,
            39, 34, 56, 8, 1, 57, 13, 35, 66, 24, 27, 95, 60, 77, 28, 54, 1, 14, 30, 57, 24, 65, 43, 44, 12, 51, 2, 91,
            46, 79, 66, 63, 12, 30, 22, 75, 57, 91, 67, 72, 47, 77, 40, 46, 13, 3, 25, 81, 15, 69, 35, 8, 13, 5, 95, 8,
            97, 28, 43, 34, 74, 41, 25, 6, 21, 53, 72, 3, 54, 26, 73, 75, 94, 71, 80, 33, 25, 45, 18, 28, 8, 73, 25, 3,
            13, 4, 1, 35, 3, 39, 61, 79, 79, 50, 13, 65, 43, 52, 20, 69, 31, 6, 45, 84, 84, 0, 29, 59, 70, 76, 96, 32,
            96, 34, 51, 57, 28, 3, 55, 20, 31, 25, 17, 87, 10, 4, 11, 1, 19, 88, 32, 89, 58, 3, 93, 18, 64, 15, 20, 6,
            10, 10, 5, 7, 1, 1, 71, 99, 15, 82, 45, 13, 17, 46, 96, 30, 60, 30, 79]

    ax = sns.distplot(data, bins=100, kde=True)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Net Export Correlation rank")
    ax.set_ylabel("Frequency (%)")
    ax.set(xlim=(0, 100))
    plt.show()

    # Generation correlation distribution
    data = [44, 2, 3, 2, 22, 55, 4, 6, 1, 35, 71, 23, 12, 22, 35, 26, 96, 8, 17, 9, 61, 72, 19, 23, 95, 16, 96, 9, 96,
            15, 10, 20, 87, 19, 45, 38, 3, 20, 18, 18, 84, 89, 35, 5, 1, 49, 4, 46, 19, 6, 0, 38, 0, 99, 17, 10, 9, 2,
            0, 19, 4, 22, 40, 11, 14, 31, 15, 12, 38, 37, 0, 21, 1, 5, 10, 35, 34, 25, 96, 2, 8, 81, 5, 0, 16, 65, 81,
            26, 1, 15, 10, 15, 7, 2, 36, 1, 24, 0, 24, 0, 50, 44, 12, 0, 60, 4, 80, 61, 5, 90, 21, 38, 18, 52, 43, 2,
            15, 2, 23, 43, 32, 17, 21, 92, 51, 5, 49, 99, 0, 29, 58, 22, 67, 7, 24, 5, 35, 4, 28, 17, 58, 0, 24, 5, 27,
            26, 15, 0, 94, 50, 12, 3, 73, 24, 24, 0, 4, 79, 66, 0, 3, 22, 3, 45, 9, 0, 4, 33, 11, 40, 0, 8, 10, 4, 1, 2,
            90, 68, 3, 10, 1, 84, 82, 91, 13, 31, 0, 0, 11, 69, 44, 15, 33, 0, 40, 12, 9, 4, 7, 25, 0, 34, 4, 21, 8, 8,
            1, 50, 54, 11, 60, 8, 6, 15, 53, 20, 0, 37, 79, 15, 61, 47, 13, 31, 26, 12, 38, 54, 3, 30, 2, 3, 0, 17, 8,
            0, 2, 20, 1, 57, 21, 20, 92, 5, 4, 95, 12, 2, 3, 2, 7, 6, 19, 1, 14, 51, 2, 33, 97, 8, 30, 9, 11, 1, 12, 49,
            7, 3, 16, 10]

    ax = sns.distplot(data, bins=100, kde=True)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Solar Generation Correlation rank")
    ax.set_ylabel("Frequency (%)")
    ax.set(xlim=(0, 100))
    plt.show()


def graphs_obj4():
    cnn_cust_obfs_l1 = [81.81*0.9, 44.22*0.9, 18.87*0.9, 8.92*0.9, 5.14*0.9, 3.52*0.9, 2.89*0.9,
                        81.81*1.1, 44.22*1.1, 18.87*1.1, 8.92*1.1, 5.14*1.1, 3.52*1.1, 2.89*1.1]
    cnn_cust_obfs_l2 = [81.81*0.9, 14.86*0.9, 12.42*0.9, 8.76*0.9, 3.63*0.9, 3.35*0.9, 2.89*0.9,
                        81.81*1.1, 14.86*1.1, 12.42*1.1, 8.76*1.1, 3.63*1.1, 3.35*1.1, 2.89*1.1]
    cnn_cust_obfs_l5 = [81.81*0.9, 10.61*0.9, 8.93*0.9, 7.32*0.9, 3.51*0.9, 3.27*0.9, 2.89*0.9,
                        81.81*1.1, 10.61*1.1, 8.93*1.1, 7.32*1.1, 3.51*1.1, 3.27*1.1, 2.89*1.1]
    cnn_cust_obfs_l10 = [81.81*0.9, 7.79*0.9, 6.33*0.9, 5.17*0.9, 3.24*0.9, 2.96*0.9, 2.89*0.9,
                         81.81*1.1, 7.79*1.1, 6.33*1.1, 5.17*1.1, 3.24*1.1, 2.96*1.1, 2.89*1.1]
    cnn_cust_obfs_l20 = [81.81*0.9, 6.37*0.9, 4.69*0.9, 3.43*0.9, 3.31*0.9, 3.00*0.9, 2.89*0.9,
                         81.81*1.1, 6.37*1.1, 4.69*1.1, 3.43*1.1, 3.31*1.1, 3.00*1.1, 2.89*1.1]

    cnn_post_obfs_l1 = [64.85*0.9, 29.71*0.95, 14.78*0.95, 14.46*0.95, 13.08*0.95, 12.81*0.95, 10.57*0.95,
                        64.85*1.1, 29.71*1.05, 14.78*1.05, 14.46*1.05, 13.08*1.05, 12.81*1.05, 10.57*1.05]
    cnn_post_obfs_l2 = [64.85*0.95, 20.71*0.95, 14.20*0.95, 13.12*0.95, 12.88*0.95, 12.66*0.95, 10.57*0.95,
                        64.85*1.05, 20.71*1.05, 14.20*1.05, 13.12*1.05, 12.88*1.05, 12.66*1.05, 10.57*1.05]
    cnn_post_obfs_l5 = [64.85*0.95, 14.56*0.95, 13.30*0.95, 13.03*0.95, 12.72*0.95, 12.42*0.95, 10.57*0.95,
                        64.85*1.05, 14.56*1.05, 13.30*1.05, 13.03*1.05, 12.72*1.05, 12.42*1.05, 10.57*1.05]
    cnn_post_obfs_l10 = [64.85*0.95, 13.95*0.95, 13.54*0.95, 12.87*0.95, 12.84*0.95, 12.86*0.95, 10.57*0.95,
                         64.85*1.05, 13.95*1.05, 13.54*1.05, 12.87*1.05, 12.84*1.05, 12.86*1.05, 10.57*1.05]
    cnn_post_obfs_l20 = [64.85*0.95, 13.76*0.95, 13.79*0.95, 12.53*0.95, 12.50*0.95, 12.30*0.95, 10.57*0.95,
                         64.85*1.05, 13.76*1.05, 13.79*1.05, 12.53*1.05, 12.50*1.05, 12.30*1.05, 10.57*1.05]

    rfc_cust_obfs_l1 = [67.62*0.9, 45.31*0.9, 24.03*0.9, 18.25*0.9, 8.03*0.9, 4.88*0.9, 3.70*0.9,
                        67.62*1.1, 45.31*1.1, 24.03*1.1, 18.25*1.1, 8.03*1.1, 4.88*1.1, 3.70*1.1]
    rfc_cust_obfs_l2 = [67.62*0.9, 13.00*0.9, 8.81*0.9, 8.22*0.9, 7.21*0.9, 4.83*0.9, 3.70*0.9,
                        67.62*1.1, 13.00*1.1, 8.81*1.1, 8.22*1.1, 7.21*1.1, 4.83*1.1, 3.70*1.1]
    rfc_cust_obfs_l5 = [67.62*0.9, 10.84*0.9, 8.93*0.9, 7.58*0.9, 6.50*0.9, 4.95*0.9, 3.70*0.9,
                        67.62*1.1, 10.84*1.1, 8.93*1.1, 7.58*1.1, 6.50*1.1, 4.95*1.1, 3.70*1.1]
    rfc_cust_obfs_l10 = [67.62*0.9, 8.89*0.9, 8.51*0.9, 7.90*0.9, 7.08*0.9, 4.76*0.9, 3.70*0.9,
                         67.62*1.1, 8.89*1.1, 8.51*1.1, 7.90*1.1, 7.08*1.1, 4.76*1.1, 3.70*1.1]
    rfc_cust_obfs_l20 = [67.62*1.1, 7.66*0.9, 7.31*0.9, 7.38*0.9, 6.92*0.9, 4.50*0.9, 3.70*0.9,
                         67.62*1.1, 7.66*1.1, 7.31*1.1, 7.38*1.1, 6.92*1.1, 4.50*1.1, 3.70*1.1]

    rfc_post_obfs_l1 = [48.52*0.95, 32.00*0.95, 24.00*0.95, 18.35*0.95, 15.89*0.95, 15.02*0.95, 11.48*0.95,
                        48.52*1.05, 32.00*1.05, 24.00*1.05, 18.35*1.05, 15.89*1.05, 15.02*1.05, 11.48*1.05]
    rfc_post_obfs_l2 = [48.52*0.95, 19.44*0.95, 17.74*0.95, 15.71*0.95, 14.86*0.95, 14.41*0.95, 11.48*0.95,
                        48.52*1.05, 19.44*1.05, 17.74*1.05, 15.71*1.05, 14.86*1.05, 14.41*1.05, 11.48*1.05]
    rfc_post_obfs_l5 = [48.52*0.95, 16.82*0.95, 17.35*0.95, 15.21*0.95, 14.79*0.95, 14.28*0.95, 11.48*0.95,
                        48.52*1.05, 16.82*1.05, 17.35*1.05, 15.21*1.05, 14.79*1.05, 14.28*1.05, 11.48*1.05]
    rfc_post_obfs_l10 = [48.52*0.95, 16.16*0.95, 15.57*0.95, 15.06*0.95, 14.59*0.95, 14.34*0.95, 11.48*0.95,
                         48.52*1.05, 16.16*1.05, 15.57*1.05, 15.06*1.05, 14.59*1.05, 14.34*1.05, 11.48*1.05]
    rfc_post_obfs_l20 = [48.52*0.95, 14.27*0.95, 15.21*0.95, 14.96*0.95, 14.33*0.95, 14.10*0.95, 11.48*0.95,
                         48.52*1.05, 14.27*1.05, 15.21*1.05, 14.96*1.05, 14.33*1.05, 14.10*1.05, 11.48*1.05]

    x_cats = [1, 2, 5, 10, 20, 50, 100, 1, 2, 5, 10, 20, 50, 100]

    # CNN obfuscation graph
    plot_comp_cnn_cust = pd.DataFrame({
        'cats': x_cats,
        '1 PK / ledger': cnn_cust_obfs_l1,
        '2 PK / ledger': cnn_cust_obfs_l2,
        '5 PK / ledger': cnn_cust_obfs_l5,
        '10 PK / ledger': cnn_cust_obfs_l10,
        '20 PK / ledger': cnn_cust_obfs_l20
    })
    plot_comp_cnn_post = pd.DataFrame({
        'cats': x_cats,
        '1 PK / ledger': cnn_post_obfs_l1,
        '2 PK / ledger': cnn_post_obfs_l2,
        '5 PK / ledger': cnn_post_obfs_l5,
        '10 PK / ledger': cnn_post_obfs_l10,
        '20 PK / ledger': cnn_post_obfs_l20
    })
    line_graph(["CNN Obfuscation - Customer", "CNN Obfuscation - Postcode"],
               [plot_comp_cnn_cust, plot_comp_cnn_post],
               xlabel="PKs per customer", ylim_max=80)

    # RFC obfuscation graph
    plot_comp_rfc_cust = pd.DataFrame({
        'cats': x_cats,
        '1 PK / ledger': rfc_cust_obfs_l1,
        '2 PK / ledger': rfc_cust_obfs_l2,
        '5 PK / ledger': rfc_cust_obfs_l5,
        '10 PK / ledger': rfc_cust_obfs_l10,
        '20 PK / ledger': rfc_cust_obfs_l20
    })
    plot_comp_rfc_post = pd.DataFrame({
        'cats': x_cats,
        '1 PK / ledger': rfc_post_obfs_l1,
        '2 PK / ledger': rfc_post_obfs_l2,
        '5 PK / ledger': rfc_post_obfs_l5,
        '10 PK / ledger': rfc_post_obfs_l10,
        '20 PK / ledger': rfc_post_obfs_l20
    })
    line_graph(["RFC Obfuscation - Customer", "RFC Obfuscation - Postcode"],
               [plot_comp_rfc_cust, plot_comp_rfc_post],
               xlabel="PKs per customer", ylim_max=80)

    # Zoom in versions without the first x points
    start_chop = 2
    end_chop = math.floor(len(x_cats)/2-1)
    plot_comp_cnn_cust = pd.DataFrame({
        'cats': x_cats[start_chop:end_chop],
        '1 PK / ledger': cnn_cust_obfs_l1[start_chop:end_chop],
        '2 PK / ledger': cnn_cust_obfs_l2[start_chop:end_chop],
        '5 PK / ledger': cnn_cust_obfs_l5[start_chop:end_chop],
        '10 PK / ledger': cnn_cust_obfs_l10[start_chop:end_chop],
        '20 PK / ledger': cnn_cust_obfs_l20[start_chop:end_chop]
    })
    plot_comp_cnn_post = pd.DataFrame({
        'cats': x_cats[start_chop:end_chop],
        '1 PK / ledger': cnn_post_obfs_l1[start_chop:end_chop],
        '2 PK / ledger': cnn_post_obfs_l2[start_chop:end_chop],
        '5 PK / ledger': cnn_post_obfs_l5[start_chop:end_chop],
        '10 PK / ledger': cnn_post_obfs_l10[start_chop:end_chop],
        '20 PK / ledger': cnn_post_obfs_l20[start_chop:end_chop]
    })
    line_graph(["CNN Obfuscation - Customer", "CNN Obfuscation - Postcode"],
               [plot_comp_cnn_cust, plot_comp_cnn_post],
               xlabel="PKs per customer", ylim_max=25)

    plot_comp_rfc_cust = pd.DataFrame({
        'cats': x_cats[start_chop:end_chop],
        '1 PK / ledger': rfc_cust_obfs_l1[start_chop:end_chop],
        '2 PK / ledger': rfc_cust_obfs_l2[start_chop:end_chop],
        '5 PK / ledger': rfc_cust_obfs_l5[start_chop:end_chop],
        '10 PK / ledger': rfc_cust_obfs_l10[start_chop:end_chop],
        '20 PK / ledger': rfc_cust_obfs_l20[start_chop:end_chop]
    })
    plot_comp_rfc_post = pd.DataFrame({
        'cats': x_cats[start_chop:end_chop],
        '1 PK / ledger': rfc_post_obfs_l1[start_chop:end_chop],
        '2 PK / ledger': rfc_post_obfs_l2[start_chop:end_chop],
        '5 PK / ledger': rfc_post_obfs_l5[start_chop:end_chop],
        '10 PK / ledger': rfc_post_obfs_l10[start_chop:end_chop],
        '20 PK / ledger': rfc_post_obfs_l20[start_chop:end_chop]
    })
    line_graph(["RFC Obfuscation - Customer", "RFC Obfuscation - Postcode"],
               [plot_comp_rfc_cust, plot_comp_rfc_post],
               xlabel="PKs per customer", ylim_max=25)

    # Probably a graph that plots CNN and RFC on same one, but less ledger lines


def bar_graph(title, data, palette='bright', ylim_max=100):
    fig, axes = plt.subplots(nrows=1, ncols=len(data), sharey='row')

    for i, bar in enumerate(data):
        ax = axes[i]
        sns.barplot(x='cats', y='value', hue='variable', palette=palette,
                    ax=ax, data=pd.melt(bar, ['cats']), ci='sd', errwidth=1)
        ax.set_title(title[i])
        ax.set_xlabel("Transaction Frequency")
        ax.set_ylabel("Accuracy (%)")
        ax.set(ylim=(0, ylim_max))
        ax.legend()

    plt.show()


def line_graph(title, data, palette='bright', xlabel="Transaction Frequency", ylim_max=100):
    fig, axes = plt.subplots(nrows=1, ncols=len(data), sharey='row')

    for i, line in enumerate(data):
        ax = axes[i]
        sns.lineplot(x='cats', y='value', hue='variable', palette=palette,
                     ax=ax, data=pd.melt(line, ['cats']), ci='sd')
        ax.set_title(title[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy (%)")
        ax.set(ylim=(0, ylim_max))
        ax.legend()

    plt.show()


if __name__ == '__main__':
    # Check usage
    if not len(sys.argv) == 5:
        print("Use: python ./graphs.py [obj1] [obj2] [obj3] [obj4]")
        print("Use a 1 or 0 indicator for each argument")
        exit()

    if int(sys.argv[1]):
        print("Preparing objective one graphs")
        graphs_obj1()

    if int(sys.argv[2]):
        print("Preparing objective two graphs")
        graphs_obj2()

    if int(sys.argv[3]):
        print("Preparing objective three graphs")
        graphs_obj3()

    if int(sys.argv[4]):
        print("Preparing objective four graphs")
        graphs_obj4()
