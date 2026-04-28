import matplotlib.pyplot as plt
import numpy as np

def plot_model_comparison():

    # =========================
    # Models
    # =========================
    models = ['CT', 'Clinical', 'Genomic', 'Fusion']

    # =========================
    # YOUR clinical values added
    # Replace others with your real results
    # =========================
    accuracy  = [0.91, 0.5133, 0.87, 0.95]
    precision = [0.89, 0.52,   0.86, 0.94]
    recall    = [0.93, 0.52,   0.89, 0.96]
    f1_score  = [0.91, 0.5409, 0.87, 0.95]

    x = np.arange(len(models))
    width = 0.2

    plt.figure()

    b1 = plt.bar(x - 1.5*width, accuracy,  width, label='Accuracy')
    b2 = plt.bar(x - 0.5*width, precision, width, label='Precision')
    b3 = plt.bar(x + 0.5*width, recall,    width, label='Recall')
    b4 = plt.bar(x + 1.5*width, f1_score,  width, label='F1-score')

    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.title("Multi-Model Performance Comparison")
    plt.xticks(x, models)
    plt.legend()
    plt.grid()

    # =========================
    # Add values on bars
    # =========================
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                     f'{height:.2f}', ha='center')

    for bars in [b1, b2, b3, b4]:
        add_labels(bars)

    plt.show()
    plt.savefig("model_comparison.png")
plt.close()