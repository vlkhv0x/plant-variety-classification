"""
Скрипт для оценки обученной модели на тестовой выборке
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

from data_preprocessing import SeedDataPreprocessor
from utils import plot_confusion_matrix, plot_sample_predictions


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Оценка модели классификации семян')
    
    parser.add_argument('--model_path', type=str, default='models/best_model.h5',
                        help='Путь к обученной модели')
    parser.add_argument('--data_dir', type=str, default='data/raw/seeds',
                        help='Путь к директории с данными')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Размер батча')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Размер изображения')
    
    return parser.parse_args()


def evaluate_model(model, test_generator, class_names):
    """
    Оценка модели на тестовой выборке
    
    Args:
        model: обученная модель
        test_generator: генератор тестовых данных
        class_names: список имён классов
        
    Returns:
        results: словарь с результатами
    """
    print("\n" + "=" * 70)
    print("📊 ОЦЕНКА МОДЕЛИ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 70)
    
    # Предсказания
    print("\n🔮 Получение предсказаний...")
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    
    # Истинные метки
    y_true = test_generator.classes
    
    # Базовые метрики
    print("\n📈 Вычисление метрик...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    
    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Вывод результатов
    print("\n" + "=" * 70)
    print("📊 РЕЗУЛЬТАТЫ")
    print("=" * 70)
    
    print(f"\n🎯 Общие метрики:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    
    print(f"\n📊 Усреднённые метрики:")
    print(f"   Precision (macro): {report['macro avg']['precision']:.4f}")
    print(f"   Recall (macro): {report['macro avg']['recall']:.4f}")
    print(f"   F1-Score (macro): {report['macro avg']['f1-score']:.4f}")
    
    print(f"\n📋 Per-class метрики:")
    print("-" * 70)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    for class_name in class_names:
        metrics = report[class_name]
        print(f"{class_name:<20} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1-score']:<12.4f} "
              f"{int(metrics['support']):<10}")
    
    print("-" * 70)
    
    # Сохранение результатов
    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': predictions.tolist(),
        'true_labels': y_true.tolist(),
        'predicted_labels': y_pred.tolist()
    }
    
    return results


def save_classification_report(report, class_names, save_path='reports/classification_report.txt'):
    """Сохранение classification report в текстовый файл"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ОТЧЁТ ПО КЛАССИФИКАЦИИ СОРТОВ РАСТЕНИЙ\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Per-class метрики:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 80 + "\n")
        
        for class_name in class_names:
            metrics = report[class_name]
            f.write(f"{class_name:<25} {metrics['precision']:<12.4f} "
                   f"{metrics['recall']:<12.4f} {metrics['f1-score']:<12.4f} "
                   f"{int(metrics['support']):<10}\n")
        
        f.write("-" * 80 + "\n\n")
        
        f.write("Усреднённые метрики:\n")
        f.write(f"  Accuracy: {report['accuracy']:.4f}\n")
        f.write(f"  Macro avg - Precision: {report['macro avg']['precision']:.4f}\n")
        f.write(f"  Macro avg - Recall: {report['macro avg']['recall']:.4f}\n")
        f.write(f"  Macro avg - F1-Score: {report['macro avg']['f1-score']:.4f}\n")
        f.write(f"  Weighted avg - Precision: {report['weighted avg']['precision']:.4f}\n")
        f.write(f"  Weighted avg - Recall: {report['weighted avg']['recall']:.4f}\n")
        f.write(f"  Weighted avg - F1-Score: {report['weighted avg']['f1-score']:.4f}\n")
    
    print(f"✅ Classification report сохранён: {save_path}")


def main():
    """Основная функция оценки"""
    
    args = parse_args()
    
    print("=" * 70)
    print("🌱 ОЦЕНКА МОДЕЛИ КЛАССИФИКАЦИИ СОРТОВ РАСТЕНИЙ")
    print("=" * 70)
    
    # Загрузка модели
    print(f"\n📥 Загрузка модели: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Модель не найдена: {args.model_path}")
    
    model = keras.models.load_model(args.model_path)
    print("✅ Модель загружена")
    
    # Загрузка конфигурации
    config_path = 'models/config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        class_names = config['class_names']
        print(f"✅ Загружена конфигурация: {len(class_names)} классов")
    else:
        print("⚠️  config.json не найден, используем данные из генератора")
        class_names = None
    
    # Подготовка данных
    print("\n📊 Подготовка тестовых данных...")
    preprocessor = SeedDataPreprocessor(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    
    # Получаем только тестовый генератор
    image_paths, labels = preprocessor.load_data_paths()
    df = preprocessor.create_dataframe(image_paths, labels)
    train_df, val_df, test_df = preprocessor.split_data(df)
    _, _, test_gen = preprocessor.create_data_generators(train_df, val_df, test_df)
    
    # Если class_names не загружены из конфига, берём из генератора
    if class_names is None:
        class_indices = test_gen.class_indices
        class_names = list(class_indices.keys())
    
    # Оценка модели
    results = evaluate_model(model, test_gen, class_names)
    
    # Сохранение результатов
    print("\n💾 Сохранение результатов...")
    
    # JSON с результатами
    with open('reports/evaluation_results.json', 'w') as f:
        # Убираем predictions для уменьшения размера файла
        results_to_save = results.copy()
        results_to_save.pop('predictions', None)
        json.dump(results_to_save, f, indent=4)
    
    print("✅ Результаты сохранены: reports/evaluation_results.json")
    
    # Classification report
    save_classification_report(
        results['classification_report'],
        class_names,
        'reports/classification_report.txt'
    )
    
    # Confusion matrix
    plot_confusion_matrix(
        results['confusion_matrix'],
        class_names,
        save_path='reports/confusion_matrix.png'
    )
    
    # Примеры предсказаний
    plot_sample_predictions(
        model,
        test_gen,
        class_names,
        num_samples=16,
        save_path='reports/predictions_sample.png'
    )
    
    print("\n" + "=" * 70)
    print("✅ ОЦЕНКА ЗАВЕРШЕНА")
    print("=" * 70)
    print(f"\n📁 Сохранённые файлы:")
    print(f"   - reports/evaluation_results.json")
    print(f"   - reports/classification_report.txt")
    print(f"   - reports/confusion_matrix.png")
    print(f"   - reports/predictions_sample.png")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
