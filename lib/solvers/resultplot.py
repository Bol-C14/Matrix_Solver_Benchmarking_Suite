import matplotlib.colors as mcolors
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def average_ratio(df):
    """Calculate the average of nnz/(rows^2) for each row in the dataframe"""
    ratios = df.apply(lambda row: row['nnz'] / (row['rows']**2), axis=1)
    return ratios.mean()

def adjust_lightness(color, amount=0.5):
    """
    Adjusts the lightness of a given color.
    
    :param color: Input color to adjust
    :param amount: 1 is original color, < 1 is darker and > 1 is lighter
    :return: adjusted color
    """
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = mcolors.to_rgb(c)
    c = tuple(map(lambda x: max(min(x * amount, 1), 0), c))
    return mcolors.to_hex(c)

def get_shade_multipliers(n):
    """
    Generates a list of shade multipliers based on the number of threads.
    :param n: The number of threads
    :return: A list of multipliers to adjust lightness
    """
    base = 0.5
    end = 1
    step = (end - base) / (n - 1) if n > 1 else 0
    return [base + step * i for i in range(n)]

def plot_data(df, base_color):
    df = df[df["Factorization"] != -1]
    
    # Check if there are NaN values in "threads" column
    if df["threads"].isna().any():
        nan_data = df[df["threads"].isna()]
        nan_data = nan_data.sort_values(by="rows")
        # plt.plot(nan_data["rows"], nan_data["Factorization"], label=f'Algorithm={df["algorithmname"].iloc[0]}', color=base_color, linewidth=1)
        plt.loglog(nan_data["rows"], nan_data["Factorization"], label=f'Algorithm={df["algorithmname"].iloc[0]}', color=base_color, linewidth=1)
        # plt.loglog(nan_data["rows"], nan_data["Factorization_1"], label=f'Algorithm={df["algorithmname"].iloc[0]}', color=base_color, linewidth=1)

        
        # Remove rows with NaN in threads for the next steps
        df = df.dropna(subset=["threads"])

    
    thread_counts = df["threads"].unique()
    shade_multipliers = get_shade_multipliers(len(thread_counts))
    print (shade_multipliers)

    for i, thread in enumerate(thread_counts):
        filtered_data = df[df["threads"] == thread]
        filtered_data = filtered_data.sort_values(by="rows")
        thread_color = adjust_lightness(base_color, shade_multipliers[i])
        # plt.plot(filtered_data["rows"], filtered_data["Factorization"], label=f'Algorithm={df["algorithmname"].iloc[0]}, Threads={thread}', color=thread_color, linewidth=1)
        plt.loglog(filtered_data["rows"], filtered_data["Factorization_1"], label=f'Algorithm={df["algorithmname"].iloc[0]}, Threads={thread}', color=thread_color, linewidth=1)

# Define base colors for each algorithm
algorithm_colors = {
    "NICSLU": "#33FF57",  # A shade of red33FF57
    "GLU": "#FF5733",    # A shade of greenFF5733
    "KLU": "#5733FF",    # A shade of blue
    "PARDISO": "#FF33F6" # A shade of pink
}

# Load data from both CSV files
df1 = pd.read_csv("Results/results_nicslu_kernel_d.csv")
df2 = pd.read_csv("Results/results_glu_kernel_d.csv")
df3 = pd.read_csv("Results/results_klu_kernel_d.csv")
df4 = pd.read_csv("Results/results_pardiso_kernel_d.csv")

df2_2 = pd.read_csv("Results/results_glu_kernel_d_double.csv")

colors = ['blue', 'red', 'green', 'purple', 'black']

# Use the base color from the dictionary for each algorithm.
# plot_data(df1, algorithm_colors[df1['algorithmname'].iloc[0]])
plot_data(df2, algorithm_colors[df2['algorithmname'].iloc[0]])
# plot_data(df3, algorithm_colors[df3['algorithmname'].iloc[0]])
# plot_data(df4, algorithm_colors[df4['algorithmname'].iloc[0]])

# plot_data(df2_2, algorithm_colors[df2_2['algorithmname'].iloc[0]])


plt.xlabel('Matrix Size')
plt.ylabel('Time for Factorization')
plt.title('Factorization Time vs Number of non-zeros')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('resultplot.png', dpi=600)
plt.show()


avg_ratio_df1 = average_ratio(df1)
print(f"nnz/rows df1: {avg_ratio_df1*100}"+"%")







