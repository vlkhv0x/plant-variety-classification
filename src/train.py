"""
Скрипт для обучения модели классификации семян
"""

import os
import argparse
import json
from pathlib import Path
import tensorflow as tf

from data_preprocessing import SeedDataPreprocessor
from model import SeedClassificationModel
from utils import plot_training_history, create_directories


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Обучение модели классификации семян')
    
    parser.add_argument('--data_dir', type=str, default='data/raw/seeds',
                        help='Путь к директории с данными')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Размер батча')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Размер изображения')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--model_type', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet', 'vgg16'],
                        help='Тип базовой модели')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Выполнить fine-tuning после обучения')
    parser.add_argument('--fine_tune_epochs', type=int, default=20,
                        help='Количество эпох для fine-tuning')
    
    return parser.parse_args()


def main():
    """Основная функция обучения"""
    
    # Парсинг аргументов
    args = parse_args()
    
    # Создание необходимых директорий
    create_directories()
    
    print("=" * 70)
    print("🌱 ОБУЧЕНИЕ МОДЕЛИ КЛАССИФИКАЦИИ СОРТОВ РАСТЕНИЙ")
    print("=" * 70)
    
    # Проверка GPU
    print("\n🖥️  Проверка доступности GPU:")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Найдено GPU: {len(gpus)}")
        for gpu in gpus:
            print(f"   - {gpu}")
    else:
        print("⚠️  GPU не найдено, используется CPU")
    
    # ========== 1. ПОДГОТОВКА ДАННЫХ ==========
    print("\n" + "=" * 70)
    print("📊 ШАГ 1: ПОДГОТОВКА ДАННЫХ")
    print("=" * 70)
    
    preprocessor = SeedDataPreprocessor(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    
    train_gen, val_gen, test_gen = preprocessor.prepare_data_pipeline()
    
    num_classes = len(train_gen.class_indices)
    print(f"\n✅ Данные подготовлены: {num_classes} классов")
    
    # ========== 2. СОЗДАНИЕ МОДЕЛИ ==========
    print("\n" + "=" * 70)
    print("🏗️  ШАГ 2: СОЗДАНИЕ МОДЕЛИ")
    print("=" * 70)
    
    model_builder = SeedClassificationModel(
        num_classes=num_classes,
        img_size=(args.img_size, args.img_size),
        model_type=args.model_type
    )
    
    model = model_builder.build_model(trainable_base=False)
    model_builder.compile_model(learning_rate=args.learning_rate)
    
    print(f"\n✅ Модель создана: {args.model_type}")
    print(f"Параметры модели: {model.count_params():,}")
    
    # ========== 3. ОБУЧЕНИЕ МОДЕЛИ ==========
    print("\n" + "=" * 70)
    print("🚀 ШАГ 3: ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 70)
    
    history = model_builder.train(
        train_generator=train_gen,
        val_generator=val_gen,
        epochs=args.epochs
    )
    
    # Сохранение истории обучения
    history_dict = history.history
    with open('reports/training_history.json', 'w') as f:
        # Конвертируем numpy в list для JSON
        history_json = {k: [float(v) for v in vals] for k, vals in history_dict.items()}
        json.dump(history_json, f, indent=4)
    
    print("\n✅ История обучения сохранена в reports/training_history.json")
    
    # Визуализация истории обучения
    plot_training_history(history, save_path='reports/training_history.png')
    
    # ========== 4. FINE-TUNING (опционально) ==========
    if args.fine_tune:
        print("\n" + "=" * 70)
        print("🔧 ШАГ 4: FINE-TUNING МОДЕЛИ")
        print("=" * 70)
        
        fine_tune_history = model_builder.fine_tune(
            train_generator=train_gen,
            val_generator=val_gen,
            epochs=args.fine_tune_epochs,
            unfreeze_layers=50,
            learning_rate=args.learning_rate / 10
        )
        
        # Сохранение истории fine-tuning
        with open('reports/fine_tune_history.json', 'w') as f:
            history_json = {k: [float(v) for v in vals] 
                          for k, vals in fine_tune_history.history.items()}
            json.dump(history_json, f, indent=4)
        
        plot_training_history(fine_tune_history, 
                            save_path='reports/fine_tune_history.png')
    
    # ========== 5. СОХРАНЕНИЕ МОДЕЛИ ==========
    print("\n" + "=" * 70)
    print("💾 ШАГ 5: СОХРАНЕНИЕ МОДЕЛИ")
    print("=" * 70)
    
    model_builder.save_model('models/final_model.h5')
    
    # Сохранение конфигурации
    config = {
        'model_type': args.model_type,
        'num_classes': num_classes,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'class_names': list(train_gen.class_indices.keys())
    }
    
    with open('models/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("✅ Конфигурация сохранена в models/config.json")
    
    # ========== ФИНАЛЬНЫЙ ОТЧЁТ ==========
    print("\n" + "=" * 70)
    print("📊 ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 70)
    
    # Лучшие результаты
    best_val_acc = max(history.history['val_accuracy'])
    best_val_loss = min(history.history['val_loss'])
    
    print(f"\n📈 Лучшие результаты на валидации:")
    print(f"   - Accuracy: {best_val_acc:.4f}")
    print(f"   - Loss: {best_val_loss:.4f}")
    
    print(f"\n📁 Сохранённые файлы:")
    print(f"   - models/best_model.h5")
    print(f"   - models/final_model.h5")
    print(f"   - models/config.json")
    print(f"   - reports/training_history.json")
    print(f"   - reports/training_history.png")
    
    print("\n🎯 Следующий шаг: запустите evaluate.py для оценки на тестовой выборке")
    print("=" * 70)


if __name__ == "__main__":
    main()
