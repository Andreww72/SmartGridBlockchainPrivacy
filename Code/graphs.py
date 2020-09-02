import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
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
    rfc_cust_lpp = [64.81, 65.14, 57.39, 53.04]
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
    grapher(["Multilayer Percecptron - Customer",
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
    grapher(["Convolutional Neural Network - Customer",
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
    grapher(["Random Forest Classifier - Customer",
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
    grapher(["K-Nearest Neighbours - Customer",
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
    grapher(["Compare LPC - Customer",
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
    grapher(["Compare LPP - Customer",
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
    grapher(["Compare AOL - Customer",
             "Compare AOL - Postcode"],
            [plot_comp_cust, plot_comp_post])


def graphs_obj3():
    # Visualise results
    x_cats = ['Weekly', 'Daily']

    # Data copied from graphs_obj2 for delta to solar results
    cnn_cust_lpc = [58.03, 69.91]
    cnn_cust_lpp = [59.21, 71.05]
    cnn_cust_aol = [2.47, 2.39]
    cnn_post_lpc = [42.85, 53.96]
    cnn_post_lpp = [45.65, 55.77]
    cnn_post_aol = [11.46, 11.47]

    rfc_cust_lpc = [62.63, 63.10]
    rfc_cust_lpp = [64.81, 65.14]
    rfc_cust_aol = [2.14, 2.63]
    rfc_post_lpc = [39.23, 36.94]
    rfc_post_lpp = [39.69, 36.52]
    rfc_post_aol = [11.86, 11.67]

    # Solar results
    cnn_cust_lpc_s = [67.98, 76.36]
    cnn_cust_lpp_s = [67.87, 76.58]
    cnn_cust_aol_s = [3.05, 2.84]
    cnn_post_lpc_s = [55.06, 62.49]
    cnn_post_lpp_s = [52.27, 59.47]
    cnn_post_aol_s = [12.04, 11.94]

    rfc_cust_lpc_s = [71.77, 72.90]
    rfc_cust_lpp_s = [75.89, 74.55]
    rfc_cust_aol_s = [2.84, 3.70]
    rfc_post_lpc_s = [57.69, 52.11]
    rfc_post_lpp_s = [58.38, 52.45]
    rfc_post_aol_s = [14.43, 13.03]

    bright_cols = sns.color_palette("bright", 10)
    dark_cols = sns.color_palette("colorblind", 10)
    palette = [dark_cols[1], bright_cols[1], dark_cols[2], bright_cols[2]]

    # Compare ledger per customer results
    plot_comp_cust = pd.DataFrame({'cats': x_cats, 'CNN': cnn_cust_lpc, 'CNN solar': cnn_cust_lpc_s,
                                   'RFC': rfc_cust_lpc, 'RFC solar': rfc_cust_lpc_s})
    plot_comp_post = pd.DataFrame({'cats': x_cats, 'CNN': cnn_post_lpc, 'CNN solar': cnn_post_lpc_s,
                                   'RFC': rfc_post_lpc, 'RFC solar': rfc_post_lpc_s})
    grapher(["Compare LPC - Customer", "Compare LPC - Postcode"], [plot_comp_cust, plot_comp_post], palette)

    # Compare ledger per postcode results
    plot_comp_cust = pd.DataFrame({'cats': x_cats, 'CNN': cnn_cust_lpp, 'CNN solar': cnn_cust_lpp_s,
                                   'RFC': rfc_cust_lpp, 'RFC solar': rfc_cust_lpp_s})
    plot_comp_post = pd.DataFrame({'cats': x_cats, 'CNN': cnn_post_lpp, 'CNN solar': cnn_post_lpp_s,
                                   'RFC': rfc_post_lpp, 'RFC solar': rfc_post_lpp_s})
    grapher(["Compare LPP - Customer", "Compare LPP - Postcode"], [plot_comp_cust, plot_comp_post], palette)

    # Compare AOL results
    plot_comp_cust = pd.DataFrame({'cats': x_cats, 'CNN': cnn_cust_aol, 'CNN solar': cnn_cust_aol_s,
                                   'RFC': rfc_cust_aol, 'RFC solar': rfc_cust_aol_s})
    plot_comp_post = pd.DataFrame({'cats': x_cats, 'CNN': cnn_post_aol, 'CNN solar': cnn_post_aol_s,
                                   'RFC': rfc_post_aol, 'RFC solar': rfc_post_aol_s})
    grapher(["Compare AOL - Customer", "Compare AOL - Postcode"], [plot_comp_cust, plot_comp_post], palette)


def graphs_obj4():
    pass


def grapher(title, data, palette='bright'):
    fig, axes = plt.subplots(nrows=1, ncols=len(data), sharey='row')

    for i, line in enumerate(data):
        ax = axes[i]
        sns.barplot(x='cats', y='value', hue='variable', palette=palette, ax=ax, data=pd.melt(line, ['cats']))
        ax.set_title(title[i])
        ax.set_xlabel("Transaction Frequency")
        ax.set_ylabel("Accuracy (%)")
        ax.set(ylim=(0, 100))
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
