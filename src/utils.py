"""
Вспомогательные функции для визуализации и утилит
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json


def create_directories():
    """Создание необходимых директорий для проекта"""
    directories = [
        'data/raw',
        'data/processed',
        'data/augmented',
        'models',
        'reports',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Директории созданы")


def plot_training_history(history, save_path='reports/training_history.png'):
    """
    Визуализация истории обучения модели
    
    Args:
        history: объект History из Keras
        save_path: путь для сохранения графика
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top-3 Accuracy (если есть)
    if 'top_3_accuracy' in history.history:
        axes[1, 0].plot(history.history['top_3_accuracy'], 
                       label='Train Top-3 Accuracy', linewidth=2)
        axes[1, 0].plot(history.history['val_top_3_accuracy'], 
                       label='Val Top-3 Accuracy', linewidth=2)
        axes[1, 0].set_title('Top-3 Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-3 Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Precision/Recall (если есть)
    if 'precision' in history.history:
        axes[1, 1].plot(history.history['precision'], 
                       label='Train Precision', linewidth=2)
        axes[1, 1].plot(history.history['val_precision'], 
                       label='Val Precision', linewidth=2)
        if 'recall' in history.history:
            axes[1, 1].plot(history.history['recall'], 
                           label='Train Recall', linewidth=2)
            axes[1, 1].plot(history.history['val_recall'], 
                           label='Val Recall', linewidth=2)
        axes[1, 1].set_title('Precision & Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ История обучения сохранена: {save_path}")
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path='reports/confusion_matrix.png',
                         figsize=(12, 10)):
    """
    Визуализация матрицы ошибок
    
    Args:
        cm: матрица ошибок (numpy array или list)
        class_names: список имён классов
        save_path: путь для сохранения
        figsize: размер фигуры
    """
    if isinstance(cm, list):
        cm = np.array(cm)
    
    # Нормализация
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Абсолютные значения
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    
    # Нормализованные значения
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Percentage'})
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Confusion matrix сохранена: {save_path}")
    plt.close()


def plot_sample_predictions(model, test_generator, class_names, 
                           num_samples=16, save_path='reports/predictions_sample.png'):
    """
    Визуализация примеров предсказаний
    
    Args:
        model: обученная модель
        test_generator: генератор тестовых данных
        class_names: список имён классов
        num_samples: количество примеров
        save_path: путь для сохранения
    """
    # Получаем батч изображений
    test_generator.reset()
    images, true_labels = next(test_generator)
    
    # Ограничиваем количество
    images = images[:num_samples]
    true_labels = true_labels[:num_samples]
    
    # Предсказания
    predictions = model.predict(images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels_idx = np.argmax(true_labels, axis=1)
    
    # Визуализация
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Изображение
        ax.imshow(images[i])
        ax.axis('off')
        
        # Метки
        true_class = class_names[true_labels_idx[i]]
        pred_class = class_names[predicted_labels[i]]
        confidence = predictions[i][predicted_labels[i]]
        
        # Цвет в зависимости от правильности
        color = 'green' if true_labels_idx[i] == predicted_labels[i] else 'red'
        
        title = f"True: {true_class}\nPred: {pred_class}\n({confidence:.2%})"
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
    
    # Скрываем пустые subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Примеры предсказаний сохранены: {save_path}")
    plt.close()


def plot_class_distribution(df, save_path='reports/class_distribution.png'):
    """
    Визуализация распределения классов
    
    Args:
        df: DataFrame с колонкой 'class'
        save_path: путь для сохранения
    """
    plt.figure(figsize=(12, 6))
    
    class_counts = df['class'].value_counts().sort_values(ascending=False)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_counts)))
    bars = plt.bar(range(len(class_counts)), class_counts.values, color=colors)
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(range(len(class_counts)), class_counts.index, rotation=45, ha='right')
    
    # Добавляем значения на бары
    for bar, count in zip(bars, class_counts.values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Распределение классов сохранено: {save_path}")
    plt.close()


def save_model_summary(model, save_path='reports/model_summary.txt'):
    """
    Сохранение архитектуры модели в текстовый файл
    
    Args:
        model: модель Keras
        save_path: путь для сохранения
    """
    with open(save_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print(f"✅ Архитектура модели сохранена: {save_path}")


def calculate_metrics_summary(results):
    """
    Расчёт сводных метрик из результатов оценки
    
    Args:
        results: словарь с результатами из evaluate.py
        
    Returns:
        summary: словарь со сводными метриками
    """
    report = results['classification_report']
    
    summary = {
        'overall_accuracy': report['accuracy'],
        'macro_precision': report['macro avg']['precision'],
        'macro_recall': report['macro avg']['recall'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_precision': report['weighted avg']['precision'],
        'weighted_recall': report['weighted avg']['recall'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'total_samples': int(report['macro avg']['support'])
    }
    
    return summary


if __name__ == "__main__":
    # Создание директорий
    create_directories()
    print("✅ Утилиты готовы к использованию")
