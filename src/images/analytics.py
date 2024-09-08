"""

"""

import os
from typing import List, Tuple, Counter, Dict
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from PIL import Image
import tensorflow as tf
import logging
import numpy as np



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Function to get the dimensions of an image
def get_image_size(image_path: str) -> Tuple[int, int]:
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        logging.error(f"Error opening image {image_path}: {e}")
        return (0, 0)


# Verify and collect all image sizes
def get_all_image_sizes(folders: List[str]) -> List[Tuple[int, int]]:
    image_sizes = []
    for folder in folders:
        if not os.path.isdir(folder):
            logging.warning(f"Folder {folder} does not exist or is not a directory.")
            continue
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                image_path = os.path.join(folder, filename)
                image_size = get_image_size(image_path)
                if image_size != (0, 0):
                    image_sizes.append(image_size)
    return image_sizes


# Calculate the average size of the images
def calculate_average_size(image_sizes: List[Tuple[int, int]]) -> Tuple[float, float]:
    if not image_sizes:
        raise ValueError("The list of image sizes is empty.")
    
    total_width = sum(width for width, height in image_sizes)
    total_height = sum(height for width, height in image_sizes)
    num_images = len(image_sizes)
    
    average_width = total_width / num_images
    average_height = total_height / num_images
    
    return average_width, average_height


# Function to count images per class
def count_images_per_class(dataset: tf.data.Dataset) -> Dict[int, int]:
    class_counts = Counter()
    for images, labels in dataset:
        class_counts.update(labels.numpy())
    return dict(class_counts)


def plot_image(ax: plt.Axes, img: tf.Tensor) -> None:
    ax.imshow(img.numpy().astype("uint8"))
    ax.axis("off")
    ax.set_facecolor('none')

def plot_images_by_category(
    images: tf.Tensor,
    labels: tf.Tensor,
    class_names: List[str],
    dataset_type: str,
    axes: plt.Axes,
    start_col: int
) -> None:
    sns.set(style="whitegrid")

    unique_labels = [0, 1, 2, 3]  # Categories: shine, cloudy, sunrise, rain
    images_per_category = len(axes)

    # Create a dictionary of category images
    category_images_dict = {label: images[labels == label][:images_per_category] for label in unique_labels}

    # Plot images using map and lambda functions
    list(map(lambda i_label: list(map(lambda j_img: plot_image(axes[j_img[0], start_col + i_label[0]], j_img[1]), enumerate(category_images_dict[i_label[1]]))), enumerate(unique_labels)))

    # Add a title for each column with black font color, bold, and smaller font size
    for i, label in enumerate(unique_labels):
        axes[0, start_col + i].set_title(f'{dataset_type} - {class_names[label]}', fontsize=8, fontweight='bold', pad=20, color='black')

def plot_training_and_validation(
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    class_names: List[str],
    num_images: int = 5
) -> None:
    # Create a figure with a size of 14x7 inches
    fig, axes = plt.subplots(num_images, 8, figsize=(14, 7), subplot_kw={'xticks': [], 'yticks': []})
    fig.patch.set_alpha(0.0)  # Set the figure background to transparent

    # Visualize images per category from the training dataset on the left
    for images, labels in train_dataset.take(1):
        plot_images_by_category(images, labels, class_names, 'Training', axes, start_col=0)

    # Visualize images per category from the validation dataset on the right
    for images, labels in validation_dataset.take(1):
        plot_images_by_category(images, labels, class_names, 'Validation', axes, start_col=4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the titles
    plt.show()
    
    
# One-hot encode the labels
def one_hot_encode(dataset: tf.data.Dataset, num_classes: int) -> tf.data.Dataset:
    return dataset.map(lambda x, y: (x, tf.one_hot(y, num_classes)))


def plot_training_validation_curves(
    epochs: List[int],
    metrics: Dict[str, List[float]],
    title: str = 'Training and Validation Curves',
    height: int = 600,
    width: int = 1200
) -> None:
    # Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Model Accuracy', 'Model Loss'))

    # Helper function to add traces
    def add_trace(fig, x, y, name, row, col, color):
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=name,
                                 line=dict(color=color), marker=dict(size=6)), row=row, col=col)

    # Add accuracy traces
    add_trace(fig, epochs, metrics.get('Train Accuracy', []), 'Train Accuracy', 1, 1, 'blue')
    add_trace(fig, epochs, metrics.get('Validation Accuracy', []), 'Validation Accuracy', 1, 1, 'red')

    # Add loss traces
    add_trace(fig, epochs, metrics.get('Train Loss', []), 'Train Loss', 1, 2, 'blue')
    add_trace(fig, epochs, metrics.get('Validation Loss', []), 'Validation Loss', 1, 2, 'red')

    # Helper function to update axes
    def update_axes(fig, title, row, col):
        fig.update_xaxes(title_text='Epoch', showgrid=True, gridwidth=1, gridcolor='LightGray', row=row, col=col)
        fig.update_yaxes(title_text=title, showgrid=True, gridwidth=1, gridcolor='LightGray', row=row, col=col)

    # Update layout and axes
    fig.update_layout(title_text=title, height=height, width=width, title_x=0.5, showlegend=True)
    update_axes(fig, 'Accuracy', 1, 1)
    update_axes(fig, 'Loss', 1, 2)

    # Show plot
    fig.show()
    

def visualize_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: List[str],
    num_images: int = 10,
    figsize: Tuple[int, int] = (14,7 )
) -> None:
    """
    Visualize predictions for a batch of images from the validation dataset.

    Parameters:
    - model: The trained model to use for predictions.
    - dataset: The validation dataset.
    - class_names: List of class names.
    - num_images: Number of images to visualize.
    - figsize: Size of the figure for visualization.
    """
    # Get a batch of images from the validation dataset
    validation_images, validation_labels = next(iter(dataset.take(1)))

    # Get model predictions
    y_pred_probs = model.predict(validation_images)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Convert true labels to class format
    y_true = np.argmax(validation_labels.numpy(), axis=1)

    # Visualize images with their predictions
    plt.figure(figsize=figsize)
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(validation_images[i].numpy().astype("uint8"))
        plt.title(f"Pred: {class_names[y_pred[i]]}\nTrue: {class_names[y_true[i]]}")
        plt.axis('off')
    plt.show()
