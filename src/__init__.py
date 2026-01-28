"""
Plant Variety Classification Package

Модуль для классификации сортов растений по изображениям семян
с использованием глубокого обучения.
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from .data_preprocessing import SeedDataPreprocessor, load_and_preprocess_image
from .model import SeedClassificationModel
from .utils import (
    create_directories,
    plot_training_history,
    plot_confusion_matrix,
    plot_sample_predictions
)

__all__ = [
    'SeedDataPreprocessor',
    'load_and_preprocess_image',
    'SeedClassificationModel',
    'create_directories',
    'plot_training_history',
    'plot_confusion_matrix',
    'plot_sample_predictions'
]
