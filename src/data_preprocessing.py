"""
Модуль для предобработки данных изображений семян
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import json


class SeedDataPreprocessor:
    """
    Класс для предобработки данных изображений семян
    """
    
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        """
        Инициализация препроцессора
        
        Args:
            data_dir: путь к директории с данными
            img_size: размер изображений (высота, ширина)
            batch_size: размер батча
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = []
        
    def load_data_paths(self):
        """
        Загрузка путей к изображениям и меток классов
        
        Returns:
            image_paths: список путей к изображениям
            labels: список меток классов
        """
        image_paths = []
        labels = []
        
        # Предполагаем структуру: data_dir/class_name/image.jpg
        for class_dir in sorted(self.data_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name not in self.class_names:
                    self.class_names.append(class_name)
                
                for img_file in class_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        image_paths.append(str(img_file))
                        labels.append(class_name)
        
        print(f"Найдено {len(image_paths)} изображений в {len(self.class_names)} классах")
        print(f"Классы: {self.class_names}")
        
        return image_paths, labels
    
    def create_dataframe(self, image_paths, labels):
        """
        Создание DataFrame с путями и метками
        
        Args:
            image_paths: список путей к изображениям
            labels: список меток
            
        Returns:
            df: pandas DataFrame
        """
        df = pd.DataFrame({
            'filepath': image_paths,
            'class': labels
        })
        
        # Статистика по классам
        print("\nРаспределение классов:")
        print(df['class'].value_counts())
        
        return df
    
    def split_data(self, df, test_size=0.2, val_size=0.1, random_state=42):
        """
        Разделение данных на train/val/test
        
        Args:
            df: DataFrame с данными
            test_size: доля тестовой выборки
            val_size: доля валидационной выборки
            random_state: seed для воспроизводимости
            
        Returns:
            train_df, val_df, test_df
        """
        # Сначала отделяем test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['class']
        )
        
        # Затем из train_val отделяем validation
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val_df['class']
        )
        
        print(f"\nРазделение данных:")
        print(f"Train: {len(train_df)} изображений")
        print(f"Validation: {len(val_df)} изображений")
        print(f"Test: {len(test_df)} изображений")
        
        return train_df, val_df, test_df
    
    def create_data_generators(self, train_df, val_df, test_df, augmentation=True):
        """
        Создание генераторов данных с аугментацией
        
        Args:
            train_df, val_df, test_df: DataFrames с данными
            augmentation: применять ли аугментацию к train
            
        Returns:
            train_generator, val_generator, test_generator
        """
        # Аугментация для обучающей выборки
        if augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.2,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        # Только нормализация для val/test
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Создание генераторов
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='filepath',
            y_col='class',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        val_generator = val_test_datagen.flow_from_dataframe(
            val_df,
            x_col='filepath',
            y_col='class',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_dataframe(
            test_df,
            x_col='filepath',
            y_col='class',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Сохраняем маппинг классов
        self.class_indices = train_generator.class_indices
        self.save_class_mapping()
        
        return train_generator, val_generator, test_generator
    
    def save_class_mapping(self, save_path='data/processed/class_mapping.json'):
        """
        Сохранение маппинга классов
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.class_indices, f, indent=4)
        print(f"\nМаппинг классов сохранён в {save_path}")
    
    def prepare_data_pipeline(self):
        """
        Полный пайплайн подготовки данных
        
        Returns:
            train_generator, val_generator, test_generator
        """
        # 1. Загрузка путей
        image_paths, labels = self.load_data_paths()
        
        # 2. Создание DataFrame
        df = self.create_dataframe(image_paths, labels)
        
        # 3. Разделение данных
        train_df, val_df, test_df = self.split_data(df)
        
        # 4. Создание генераторов
        train_gen, val_gen, test_gen = self.create_data_generators(
            train_df, val_df, test_df
        )
        
        return train_gen, val_gen, test_gen


def load_and_preprocess_image(image_path, img_size=(224, 224)):
    """
    Загрузка и предобработка одного изображения
    
    Args:
        image_path: путь к изображению
        img_size: размер выходного изображения
        
    Returns:
        preprocessed_image: обработанное изображение
    """
    img = Image.open(image_path)
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    
    # Проверка каналов
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    
    return img_array


if __name__ == "__main__":
    # Пример использования
    preprocessor = SeedDataPreprocessor(
        data_dir='data/raw/seeds',
        img_size=(224, 224),
        batch_size=32
    )
    
    train_gen, val_gen, test_gen = preprocessor.prepare_data_pipeline()
    
    print("\n✅ Данные успешно подготовлены!")
