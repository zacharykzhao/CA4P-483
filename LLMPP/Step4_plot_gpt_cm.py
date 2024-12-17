import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(cm, labels, cmap="Blues", output_file="confusion_matrix.pdf"):
    """
    Plot a confusion matrix using matplotlib and save it as a PDF.

    Parameters:
    - cm: array-like of shape (n_classes, n_classes)
      Confusion matrix values.
    - labels: array-like of shape (n_classes,)
      List of labels to index the matrix.
    - cmap: str or matplotlib Colormap, optional
      Colormap used to plot the confusion matrix.
    - output_file: str, optional
      File path to save the figure as a PDF.

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(11, 11))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap=cmap, values_format="d", colorbar=True)

    # Adjust bar and labels
    ax.images[0].set_clim(0, cm.max())  # Brighten blue color
    ax.set_xlabel("Predicted Label", fontsize=16, weight="bold")
    ax.set_ylabel("True Label", fontsize=16, weight="bold")
    ax.tick_params(axis='x', labelrotation=45, labelsize=14, width=2)
    ax.tick_params(axis='y', labelsize=14, width=2)

    # Increase font size for confusion matrix values
    for text in disp.text_.ravel():
        text.set_fontsize(14)
        text.set_weight("bold")

    # Make grid lines thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_aspect('auto')
    plt.savefig(output_file, format="pdf")
    plt.show()
    # plt.close(fig)

# Example usage
if __name__ == "__main__":
    # Confusion matrix data
    cm = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [7565, 5568, 0, 0, 0, 0, 0, 0],
        [4300, 0, 4097, 0, 0, 0, 0, 0],
        [3675, 0, 0, 1450, 0, 0, 0, 0],
        [3282, 0, 0, 0, 1603, 0, 0, 0],
        [4875, 0, 0, 0, 0, 117, 0, 0],
        [4475, 0, 0, 0, 0, 0, 244, 0],
        [1992, 0, 0, 0, 0, 0, 0, 1101]
    ])
    labels = ["O", "Data", "Controller", "Collection", "Sharing", "Condition", "Purpose", "Receiver"]

    # Save confusion matrix as PDF
    plot_confusion_matrix(cm, labels, output_file="confusion_matrix.pdf")
