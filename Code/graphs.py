import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from  matplotlib.ticker import PercentFormatter
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
    x_cats = ['Weekly', 'Daily', 'Hourly', 'Half Hourly']

    # Enter results
    mlp_cust_lpc = [45.72, 64.18, 75.26, 73.73]
    mlp_cust_lpp = [47.88, 66.11, 77.03, 76.10]
    mlp_cust_aol = [2.28, 1.93, 1.83, 1.86]
    mlp_post_lpc = [22.52, 31.45, 40.33, 36.68]
    mlp_post_lpp = [25.00, 34.43, 43.21, 38.70]
    mlp_post_aol = [11.48, 11.31, 10.20, 9.86]

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

    knn_cust_lpc = [35.97, 47.82, 54.52, 50.40]
    knn_cust_lpp = [34.91, 47.83, 54.54, 51.46]
    knn_cust_aol = [0.95, 1.86, 2.22, 2.39]
    knn_post_lpc = [23.06, 37.94, 43.39, 38.56]
    knn_post_lpp = [26.44, 37.92, 43.46, 39.57]
    knn_post_aol = [10.30, 10.75, 9.22, 9.52]

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

    # Correlation distribution
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
    ax.set_xlabel("Correlation rank")
    ax.set_ylabel("Frequency (%)")
    ax.set(xlim=(0, 100))
    plt.show()


def graphs_obj4():
    cnn_cust_obfs_l1 = [67.98, 44.22, 1, 8.92, 5.14, 3.52, 3.05]
    cnn_cust_obfs_l2 = [67.98, 44.22, 1, 8.92, 5.14, 3.52, 3.05]
    cnn_cust_obfs_l5 = [67.98, 44.22, 1, 8.92, 5.14, 3.52, 3.05]
    cnn_cust_obfs_l10 = [67.98, 44.22, 1, 8.92, 5.14, 3.52, 3.05]
    cnn_cust_obfs_l20 = [67.98, 44.22, 1, 8.92, 5.14, 3.52, 3.05]
    cnn_post_obfs_l1 = [55.06, 29.71, 1, 12.46, 12.08, 11.61, 12.04]
    cnn_post_obfs_l2 = [55.06, 29.71, 1, 12.46, 12.08, 11.61, 12.04]
    cnn_post_obfs_l5 = [55.06, 29.71, 1, 12.46, 12.08, 11.61, 12.04]
    cnn_post_obfs_l10 = [55.06, 29.71, 1, 12.46, 12.08, 11.61, 12.04]
    cnn_post_obfs_l20 = [55.06, 29.71, 1, 12.46, 12.08, 11.61, 12.04]

    rfc_cust_obfs_l1 = [71.77, 45.31, 24.03, 18.25, 8.03, 4.88, 2.84]
    rfc_cust_obfs_l2 = [71.77, 13.00, 8.81, 8.22, 7.21, 4.83, 2.84]
    rfc_cust_obfs_l5 = [71.77, 10.84, 8.93, 7.58, 6.50, 4.95, 2.84]
    rfc_cust_obfs_l10 = [71.77, 8.89, 8.51, 7.90, 7.08, 4.76, 2.84]
    rfc_cust_obfs_l20 = [71.77, 7.66, 7.31, 7.38, 6.92, 4.50, 2.84]
    rfc_post_obfs_l1 = [57.69, 32.00, 24.00, 18.35, 15.89, 15.02, 14.43]
    rfc_post_obfs_l2 = [57.69, 19.44, 17.74, 15.71, 14.86, 14.41, 14.43]
    rfc_post_obfs_l5 = [57.69, 16.82, 17.35, 15.21, 14.79, 14.28, 14.43]
    rfc_post_obfs_l10 = [57.69, 16.16, 15.57, 15.06, 14.59, 14.34, 14.43]
    rfc_post_obfs_l20 = [57.69, 14.27, 15.21, 14.96, 14.33, 14.10, 14.43]

    x_cats = [1, 2, 5, 10, 20, 50, 100]

    # CNN obfuscation graph
    plot_comp_cnn_cust = pd.DataFrame({
        'cats': x_cats,
        'CNN 1 PK / ledger': cnn_cust_obfs_l1,
        'CNN 2 PK / ledger': cnn_cust_obfs_l2,
        'CNN 5 PK / ledger': cnn_cust_obfs_l5,
        'CNN 10 PK / ledger': cnn_cust_obfs_l10,
        'CNN 20 PK / ledger': cnn_cust_obfs_l20
    })
    plot_comp_cnn_post = pd.DataFrame({
        'cats': x_cats,
        'CNN 1 PK / ledger': cnn_post_obfs_l1,
        'CNN 2 PK / ledger': cnn_post_obfs_l2,
        'CNN 5 PK / ledger': cnn_post_obfs_l5,
        'CNN 10 PK / ledger': cnn_post_obfs_l10,
        'CNN 20 PK / ledger': cnn_post_obfs_l20
    })
    line_graph(["CNN Obfuscation - Customer", "CNN Obfuscation - Postcode"],
               [plot_comp_cnn_cust, plot_comp_cnn_post],
               xlabel="PKs per customer", ylim_max=80)

    # RFC obfuscation graph
    plot_comp_rfc_cust = pd.DataFrame({
        'cats': x_cats,
        'RFC 1 PK / ledger': rfc_cust_obfs_l1,
        'RFC 2 PK / ledger': rfc_cust_obfs_l2,
        'RFC 5 PK / ledger': rfc_cust_obfs_l5,
        'RFC 10 PK / ledger': rfc_cust_obfs_l10,
        'RFC 20 PK / ledger': rfc_cust_obfs_l20
    })
    plot_comp_rfc_post = pd.DataFrame({
        'cats': x_cats,
        'RFC 1 PK / ledger': rfc_post_obfs_l1,
        'RFC 2 PK / ledger': rfc_post_obfs_l2,
        'RFC 5 PK / ledger': rfc_post_obfs_l5,
        'RFC 10 PK / ledger': rfc_post_obfs_l10,
        'RFC 20 PK / ledger': rfc_post_obfs_l20
    })
    line_graph(["RFC Obfuscation - Customer", "RFC Obfuscation - Postcode"],
               [plot_comp_rfc_cust, plot_comp_rfc_post],
               xlabel="PKs per customer", ylim_max=80)

    # Zoom in versions without the first x points
    chop = 2
    plot_comp_cnn_cust = pd.DataFrame({
        'cats': x_cats[chop:-1],
        'CNN 1 PK / ledger': cnn_cust_obfs_l1[chop:-1],
        'CNN 2 PK / ledger': cnn_cust_obfs_l2[chop:-1],
        'CNN 5 PK / ledger': cnn_cust_obfs_l5[chop:-1],
        'CNN 10 PK / ledger': cnn_cust_obfs_l10[chop:-1],
        'CNN 20 PK / ledger': cnn_cust_obfs_l20[chop:-1]
    })
    plot_comp_cnn_post = pd.DataFrame({
        'cats': x_cats[chop:-1],
        'CNN 1 PK / ledger': cnn_post_obfs_l1[chop:-1],
        'CNN 2 PK / ledger': cnn_post_obfs_l2[chop:-1],
        'CNN 5 PK / ledger': cnn_post_obfs_l5[chop:-1],
        'CNN 10 PK / ledger': cnn_post_obfs_l10[chop:-1],
        'CNN 20 PK / ledger': cnn_post_obfs_l20[chop:-1]
    })
    line_graph(["CNN Obfuscation - Customer", "CNN Obfuscation - Postcode"],
               [plot_comp_cnn_cust, plot_comp_cnn_post],
               xlabel="PKs per customer", ylim_max=25)

    plot_comp_rfc_cust = pd.DataFrame({
        'cats': x_cats[chop:-1],
        'RFC 1 PK / ledger': rfc_cust_obfs_l1[chop:-1],
        'RFC 2 PK / ledger': rfc_cust_obfs_l2[chop:-1],
        'RFC 5 PK / ledger': rfc_cust_obfs_l5[chop:-1],
        'RFC 10 PK / ledger': rfc_cust_obfs_l10[chop:-1],
        'RFC 20 PK / ledger': rfc_cust_obfs_l20[chop:-1]
    })
    plot_comp_rfc_post = pd.DataFrame({
        'cats': x_cats[chop:-1],
        'RFC 1 PK / ledger': rfc_post_obfs_l1[chop:-1],
        'RFC 2 PK / ledger': rfc_post_obfs_l2[chop:-1],
        'RFC 5 PK / ledger': rfc_post_obfs_l5[chop:-1],
        'RFC 10 PK / ledger': rfc_post_obfs_l10[chop:-1],
        'RFC 20 PK / ledger': rfc_post_obfs_l20[chop:-1]
    })
    line_graph(["RFC Obfuscation - Customer", "RFC Obfuscation - Postcode"],
               [plot_comp_rfc_cust, plot_comp_rfc_post],
               xlabel="PKs per customer", ylim_max=25)

    # Probably a graph that plots CNN and RFC on same one, but less ledger lines


def bar_graph(title, data, palette='bright', ylim_max=100):
    fig, axes = plt.subplots(nrows=1, ncols=len(data), sharey='row')

    for i, bar in enumerate(data):
        ax = axes[i]
        sns.barplot(x='cats', y='value', hue='variable', palette=palette, ax=ax, data=pd.melt(bar, ['cats']))
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
        sns.lineplot(x='cats', y='value', hue='variable', palette=palette, ax=ax, data=pd.melt(line, ['cats']))
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
