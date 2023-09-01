import matplotlib.pyplot as plt
import seaborn as sns

palette = 'ch:.25'

def histplot(df, cols, figsize):
    fig, axes = plt.subplots(nrows=len(cols), figsize=figsize)

    for i,c in enumerate(cols):
        sns.histplot(df[c], ax=axes[i], kde=True, color='orange')
    
        mean = df[c].mean()
        median = df[c].median()
    
        axes[i].axvline(mean, color='red', linestyle='--', label=f"Mean: {mean:.2f}")
        axes[i].axvline(median, color='blue', linestyle='--', label=f"Median: {median:.2f}")
    
        axes[i].set_title(f'Histrogram for {c}')
        axes[i].set_xlabel(c)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()

    plt.tight_layout()
    plt.show()
     
def countplot(df, cols):
    fig, axes = plt.subplots(len(cols), 1, figsize=(8, 6 * len(cols)))

    for i, (c, ax) in enumerate(zip(cols, axes)):
        sns.countplot(x=c, data=df, ax=ax, palette=palette, hue='drafted')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title(c)
    
    plt.tight_layout()
    plt.show()