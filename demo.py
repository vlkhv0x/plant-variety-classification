"""
Демонстрационный скрипт для проверки работы проекта
с синтетическими данными (без реального датасета)
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json

# Добавляем src в путь
sys.path.append('src')

from data_preprocessing import SeedDataPreprocessor
from model import SeedClassificationModel
from utils import create_directories, plot_training_history


def create_synthetic_dataset(output_dir='data/raw/seeds_demo', 
                             num_classes=5, 
                             samples_per_class=50,
                             img_size=(224, 224)):
    """
    Создание синтетического датасета для демонстрации
    
    Args:
        output_dir: директория для сохранения
        num_classes: количество классов
        samples_per_class: количество образцов на класс
        img_size: размер изображений
    """
    print("🎨 Создание синтетического датасета...")
    
    class_names = [
        'wheat', 'corn', 'barley', 'oats', 'rye',
        'soybean', 'sunflower', 'pea', 'lentil', 'chickpea'
    ][:num_classes]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for i in range(samples_per_class):
            # Создаём случайное изображение
            # Каждый класс имеет свой базовый цвет
            base_color = (
                50 + class_idx * 40,
                100 + class_idx * 30,
                150 + class_idx * 20
            )
            
            # Создаём изображение с шумом
            img_array = np.random.randint(0, 50, (*img_size, 3), dtype=np.uint8)
            img_array = img_array + np.array(base_color)
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
            img = Image.fromarray(img_array)
            
            # Добавляем случайные "семена" (круги)
            draw = ImageDraw.Draw(img)
            num_seeds = np.random.randint(5, 15)
            
            for _ in range(num_seeds):
                x = np.random.randint(20, img_size[0] - 20)
                y = np.random.randint(20, img_size[1] - 20)
                r = np.random.randint(5, 15)
                
                # Цвет круга зависит от класса
                seed_color = tuple([
                    int(c + np.random.randint(-30, 30)) 
                    for c in base_color
                ])
                
                draw.ellipse([x-r, y-r, x+r, y+r], fill=seed_color)
            
            # Сохраняем
            img_path = os.path.join(class_dir, f'{class_name}_{i:03d}.jpg')
            img.save(img_path, quality=95)
        
        print(f"  ✓ Класс '{class_name}': {samples_per_class} изображений")
    
    print(f"\n✅ Синтетический датасет создан: {output_dir}")
    print(f"   Классов: {num_classes}")
    print(f"   Всего изображений: {num_classes * samples_per_class}")
    
    return output_dir, class_names


def demo_training():
    """Демонстрация процесса обучения на синтетических данных"""
    
    print("=" * 70)
    print("🌱 ДЕМОНСТРАЦИЯ ОБУЧЕНИЯ МОДЕЛИ (СИНТЕТИЧЕСКИЕ ДАННЫЕ)")
    print("=" * 70)
    
    # Создание директорий
    create_directories()
    
    # Создание синтетического датасета
    data_dir, class_names = create_synthetic_dataset(
        num_classes=5,
        samples_per_class=30
    )
    
    # Параметры
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 8
    EPOCHS = 3  # Мало эпох для демо
    
    # ========== ПОДГОТОВКА ДАННЫХ ==========
    print("\n" + "=" * 70)
    print("📊 ПОДГОТОВКА ДАННЫХ")
    print("=" * 70)
    
    preprocessor = SeedDataPreprocessor(
        data_dir=data_dir,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    train_gen, val_gen, test_gen = preprocessor.prepare_data_pipeline()
    num_classes = len(train_gen.class_indices)
    
    # ========== СОЗДАНИЕ МОДЕЛИ ==========
    print("\n" + "=" * 70)
    print("🏗️  СОЗДАНИЕ МОДЕЛИ")
    print("=" * 70)
    
    model_builder = SeedClassificationModel(
        num_classes=num_classes,
        img_size=IMG_SIZE,
        model_type='resnet50'
    )
    
    model = model_builder.build_model(trainable_base=False)
    model_builder.compile_model(learning_rate=0.001)
    
    print(f"\n✅ Модель создана")
    print(f"   Классов: {num_classes}")
    print(f"   Параметров: {model.count_params():,}")
    
    # ========== ОБУЧЕНИЕ ==========
    print("\n" + "=" * 70)
    print("🚀 ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 70)
    print("⚠️  Это демо с синтетическими данными - всего 3 эпохи")
    
    history = model_builder.train(
        train_generator=train_gen,
        val_generator=val_gen,
        epochs=EPOCHS
    )
    
    # Сохранение истории
    with open('reports/demo_training_history.json', 'w') as f:
        history_json = {k: [float(v) for v in vals] 
                       for k, vals in history.history.items()}
        json.dump(history_json, f, indent=4)
    
    # Визуализация
    plot_training_history(history, save_path='reports/demo_training_history.png')
    
    # Сохранение модели
    model_builder.save_model('models/demo_model.h5')
    
    # Конфигурация
    config = {
        'model_type': 'resnet50',
        'num_classes': num_classes,
        'img_size': IMG_SIZE[0],
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'class_names': class_names,
        'note': 'Demo model trained on synthetic data'
    }
    
    with open('models/demo_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # ========== РЕЗУЛЬТАТЫ ==========
    print("\n" + "=" * 70)
    print("✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 70)
    
    best_val_acc = max(history.history['val_accuracy'])
    best_val_loss = min(history.history['val_loss'])
    
    print(f"\n📈 Результаты (синтетические данные):")
    print(f"   Лучшая Val Accuracy: {best_val_acc:.4f}")
    print(f"   Лучший Val Loss: {best_val_loss:.4f}")
    
    print(f"\n📁 Сохранённые файлы:")
    print(f"   ✓ models/demo_model.h5")
    print(f"   ✓ models/demo_config.json")
    print(f"   ✓ reports/demo_training_history.json")
    print(f"   ✓ reports/demo_training_history.png")
    print(f"   ✓ data/processed/class_mapping.json")
    
    print("\n" + "=" * 70)
    print("📝 ПРИМЕЧАНИЕ")
    print("=" * 70)
    print("Это демонстрация с СИНТЕТИЧЕСКИМИ данными.")
    print("Для реального проекта:")
    print("  1. Скачайте датасет с Kaggle")
    print("  2. Поместите в data/raw/seeds/")
    print("  3. Запустите: python src/train.py --epochs 50")
    print("=" * 70)


if __name__ == "__main__":
    demo_training()
