"""
Скрипт для предсказания класса семян на новых изображениях
"""

import os
import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from data_preprocessing import load_and_preprocess_image


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Предсказание класса семян')
    
    parser.add_argument('--image_path', type=str, required=True,
                        help='Путь к изображению')
    parser.add_argument('--model_path', type=str, default='models/best_model.h5',
                        help='Путь к обученной модели')
    parser.add_argument('--config_path', type=str, default='models/config.json',
                        help='Путь к конфигурации модели')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Количество топ предсказаний')
    parser.add_argument('--show_image', action='store_true',
                        help='Показать изображение с предсказанием')
    
    return parser.parse_args()


def load_config(config_path):
    """Загрузка конфигурации модели"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Конфигурация не найдена: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def predict_image(model, image_path, class_names, img_size=(224, 224), top_k=3):
    """
    Предсказание класса для одного изображения
    
    Args:
        model: обученная модель
        image_path: путь к изображению
        class_names: список имён классов
        img_size: размер изображения
        top_k: количество топ предсказаний
        
    Returns:
        predictions: словарь с результатами
    """
    # Загрузка и предобработка изображения
    img = load_and_preprocess_image(image_path, img_size)
    img_batch = np.expand_dims(img, axis=0)  # Добавляем batch dimension
    
    # Предсказание
    predictions = model.predict(img_batch, verbose=0)[0]
    
    # Топ-k предсказаний
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_classes = [class_names[i] for i in top_indices]
    top_probs = [predictions[i] for i in top_indices]
    
    results = {
        'top_classes': top_classes,
        'top_probabilities': [float(p) for p in top_probs],
        'all_predictions': {class_names[i]: float(predictions[i]) 
                           for i in range(len(class_names))}
    }
    
    return results, img


def visualize_prediction(image, predictions, save_path=None):
    """
    Визуализация изображения с предсказаниями
    
    Args:
        image: изображение (numpy array)
        predictions: результаты предсказания
        save_path: путь для сохранения (если None - показать)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Изображение
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Входное изображение', fontsize=14, fontweight='bold')
    
    # Топ предсказаний
    top_classes = predictions['top_classes']
    top_probs = predictions['top_probabilities']
    
    y_pos = np.arange(len(top_classes))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_classes)))
    
    bars = ax2.barh(y_pos, top_probs, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_classes)
    ax2.invert_yaxis()
    ax2.set_xlabel('Вероятность', fontsize=12)
    ax2.set_title('Топ предсказаний', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    
    # Добавляем значения на бары
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        ax2.text(prob + 0.02, i, f'{prob:.2%}', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Визуализация сохранена: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Основная функция предсказания"""
    
    args = parse_args()
    
    print("=" * 70)
    print("🔮 ПРЕДСКАЗАНИЕ КЛАССА СЕМЯН")
    print("=" * 70)
    
    # Проверка существования изображения
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Изображение не найдено: {args.image_path}")
    
    print(f"\n📷 Изображение: {args.image_path}")
    
    # Загрузка конфигурации
    print(f"\n📥 Загрузка конфигурации: {args.config_path}")
    config = load_config(args.config_path)
    class_names = config['class_names']
    img_size = config.get('img_size', 224)
    
    print(f"✅ Конфигурация загружена: {len(class_names)} классов")
    
    # Загрузка модели
    print(f"\n📥 Загрузка модели: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Модель не найдена: {args.model_path}")
    
    model = keras.models.load_model(args.model_path)
    print("✅ Модель загружена")
    
    # Предсказание
    print(f"\n🔮 Выполнение предсказания (top-{args.top_k})...")
    predictions, image = predict_image(
        model, 
        args.image_path, 
        class_names,
        img_size=(img_size, img_size),
        top_k=args.top_k
    )
    
    # Вывод результатов
    print("\n" + "=" * 70)
    print("📊 РЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЯ")
    print("=" * 70)
    
    print(f"\n🥇 Топ-{args.top_k} предсказаний:")
    for i, (class_name, prob) in enumerate(zip(predictions['top_classes'], 
                                               predictions['top_probabilities']), 1):
        print(f"   {i}. {class_name:<20} - {prob:.2%}")
    
    print(f"\n🎯 Лучшее предсказание:")
    best_class = predictions['top_classes'][0]
    best_prob = predictions['top_probabilities'][0]
    print(f"   Класс: {best_class}")
    print(f"   Уверенность: {best_prob:.2%}")
    
    # Визуализация
    if args.show_image:
        print("\n📊 Визуализация результатов...")
        save_path = f"reports/prediction_{Path(args.image_path).stem}.png"
        visualize_prediction(image, predictions, save_path=save_path)
    
    # Сохранение результатов в JSON
    output_path = f"reports/prediction_{Path(args.image_path).stem}.json"
    with open(output_path, 'w') as f:
        json.dump({
            'image_path': args.image_path,
            'predictions': predictions
        }, f, indent=4)
    
    print(f"\n✅ Результаты сохранены: {output_path}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
