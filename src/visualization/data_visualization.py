import matplotlib.pyplot as plt
import seaborn as sns

palette = 'ch:.25'

class DataVisualization:
    def __init__(self, palette='ch:.25'):
        self.palette = palette

    def histplot(self, df, num_cols, target_col):
        fig, axes = plt.subplots(9, 6, figsize=(24, 18))
        axes = [ax for axes_row in axes for ax in axes_row]

        for i,c in enumerate(num_cols):
            if c == target_col:
                continue

            sns.histplot(data=df, x=c, ax=axes[i], kde=True, color='orange', hue=target_col)
    
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

    def boxplot(self, df, num_cols, target_col):
        fig, axes = plt.subplots(9, 6, figsize=(30, 20))
        axes = [ax for axes_row in axes for ax in axes_row]

        for i, c in enumerate(num_cols):
            if c == target_col:
                continue

            sns.boxplot(data=df, x=target_col, y=c, ax=axes[i], palette=self.palette)
            axes[i].set_title(f'Boxplot for {c}')
            axes[i].set_xlabel(target_col)
            axes[i].set_ylabel(c)

    def corr_plot(self, df, cols):
        fig, axes = plt.subplots(8, 7, figsize=(26, 28))
        axes = [ax for axes_row in axes for ax in axes_row]

        for i, c in enumerate(cols[:-1]):
            sns.boxplot(x="drafted", y=c, data=df, ax=axes[i], palette=palette)
            axes[i].set_title(f'Correlation with drafted: {c}', fontsize=12)

        plt.tight_layout()
        plt.show()

    def heatmap(self, df, cols):
        plt.figure(figsize=(50, 30))
        ax = plt.subplot()
        sns.heatmap(df[cols].corr(), annot=True)

        plt.show()
     
    def countplot(self, df, cols):
        fig, axes = plt.subplots(5, 1, figsize=(8, 6 * len(cols)))

        for i, (c, ax) in enumerate(zip(cols, axes)):
            if c == 'player_id' or c == 'team':
                continue
            sns.countplot(x=c, data=df, ax=ax, palette=palette, hue='drafted')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_title(c)
    
        plt.tight_layout()
        plt.show()

    def target_cols(self, df, target_cols):
        ax = sns.countplot(x=target_cols, data=df, palette=palette)

        total = len(df)
        for p in ax.patches:
            percentage = '{:.1f}%'.format(100 * p.get_height() / total)
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax.annotate(percentage, (x, y), ha='center')
        
        plt.show()