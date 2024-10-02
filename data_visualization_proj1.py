import matplotlib.pyplot as plt
import seaborn as sns

###############################################################
#
# Get the dataframe from PCA.py
#
###############################################################

from PCA import df

###############################################################
#
# Representations of data
#
###############################################################

def save_correlation_heatmap(df, filename):

    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Set up the matplotlib figure with default size
    plt.figure(figsize=(8, 6))

    # Draw the heatmap
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 10})

    # Set the title
    plt.title('Correlation Matrix Heatmap')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Adjust the layout to prevent clipping of labels
    plt.tight_layout()

    # Save the figure to a file
    plt.savefig(filename, dpi=300, bbox_inches='tight')

def save_density_histograms(df, filename):

    # Check if the DataFrame has at least 4 columns
    if df.shape[1] < 4:
        raise ValueError("DataFrame must contain at least four columns.")
    
    # Setting the style of seaborn
    sns.set(style="whitegrid")

    # Create a figure and axis to plot
    plt.figure(figsize=(12, 8))

    # List of attributes to plot (first four columns)
    attributes = df.columns[:4]

    # Loop through each attribute and plot
    for i, attribute in enumerate(attributes):
        plt.subplot(2, 2, i + 1)  # 2x2 grid for 4 plots
        sns.kdeplot(df[attribute], fill=True, alpha=0.5, linewidth=1.5)
        plt.title(f'Density Histogram of {attribute}')
        plt.xlabel(attribute)
        plt.ylabel('Density')

    # Adjust layout
    plt.tight_layout()

    # Save the plot to the specified filename
    plt.savefig(filename)
    plt.close()  # Close the plot to free memory

def plot_scatter(df, x_attr, y_attr):

    ax = df.plot.scatter(x=x_attr, y=y_attr, alpha=0.5, color='blue')
    ax.set_title(f'Scatter plot of {y_attr} vs {x_attr}')
    return ax

def plot_multiple_scatter(plots, ncols=2):

    n = len(plots)
    nrows = (n + ncols - 1) // ncols  # Calculate number of rows needed

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    axs = axs.flatten()  # Flatten the array of axes for easy iteration

    for i, (df, x_attr, y_attr) in enumerate(plots):
        ax = axs[i]
        df.plot.scatter(x=x_attr, y=y_attr, alpha=0.5, color='blue', ax=ax)
        ax.set_title(f'Scatter plot of {y_attr} vs {x_attr}')
    
    # Hide any unused axes
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    plt.savefig('Intro to machine learning/scatterplots.PNG')

plots = [(df, 'rarity', 'price'), (df, 'power', 'toughness')]

plot_multiple_scatter(plots, ncols=2)
save_correlation_heatmap(df, 'Intro to machine learning/correlation_heatmap.png')
save_density_histograms(df, 'Intro to machine learning/density_histograms.png')