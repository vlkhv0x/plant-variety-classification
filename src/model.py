"""
Модуль с архитектурой модели для классификации семян
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0, VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class SeedClassificationModel:
    """
    Класс для создания и обучения модели классификации семян
    """
    
    def __init__(self, num_classes, img_size=(224, 224), model_type='resnet50'):
        """
        Инициализация модели
        
        Args:
            num_classes: количество классов
            img_size: размер входных изображений
            model_type: тип базовой модели ('resnet50', 'efficientnet', 'vgg16')
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.model_type = model_type
        self.model = None
        
    def build_model(self, trainable_base=False):
        """
        Построение модели с transfer learning
        
        Args:
            trainable_base: делать ли базовую модель trainable
            
        Returns:
            model: скомпилированная модель Keras
        """
        input_shape = (*self.img_size, 3)
        
        # Выбор базовой модели
        if self.model_type == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif self.model_type == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif self.model_type == 'vgg16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")
        
        # Заморозка базовой модели
        base_model.trainable = trainable_base
        
        # Построение полной модели
        inputs = keras.Input(shape=input_shape)
        
        # Базовая модель
        x = base_model(inputs, training=False)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Создание модели
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.0001):
        """
        Компиляция модели
        
        Args:
            learning_rate: learning rate для оптимизатора
        """
        if self.model is None:
            raise ValueError("Сначала создайте модель с помощью build_model()")
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        print("✅ Модель скомпилирована")
        
    def get_callbacks(self, checkpoint_path='models/best_model.h5'):
        """
        Создание callbacks для обучения
        
        Args:
            checkpoint_path: путь для сохранения лучшей модели
            
        Returns:
            список callbacks
        """
        callbacks = [
            # Early Stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model Checkpoint
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce Learning Rate
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, train_generator, val_generator, epochs=50, callbacks=None):
        """
        Обучение модели
        
        Args:
            train_generator: генератор обучающих данных
            val_generator: генератор валидационных данных
            epochs: количество эпох
            callbacks: список callbacks
            
        Returns:
            history: история обучения
        """
        if self.model is None:
            raise ValueError("Сначала создайте и скомпилируйте модель")
        
        if callbacks is None:
            callbacks = self.get_callbacks()
        
        print(f"\n🚀 Начало обучения модели {self.model_type}...")
        print(f"Эпохи: {epochs}")
        print(f"Train batches: {len(train_generator)}")
        print(f"Val batches: {len(val_generator)}")
        
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Обучение завершено!")
        
        return history
    
    def fine_tune(self, train_generator, val_generator, epochs=20, 
                  unfreeze_layers=50, learning_rate=1e-5):
        """
        Fine-tuning модели (разморозка части базовой модели)
        
        Args:
            train_generator: генератор обучающих данных
            val_generator: генератор валидационных данных
            epochs: количество эпох fine-tuning
            unfreeze_layers: сколько последних слоёв разморозить
            learning_rate: learning rate для fine-tuning
            
        Returns:
            history: история обучения
        """
        print(f"\n🔧 Fine-tuning: размораживаем последние {unfreeze_layers} слоёв")
        
        # Размораживаем последние слои базовой модели
        base_model = self.model.layers[1]  # Базовая модель - второй слой
        base_model.trainable = True
        
        # Замораживаем все кроме последних unfreeze_layers
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
        
        # Перекомпиляция с меньшим learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Обучение
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=self.get_callbacks(checkpoint_path='models/finetuned_model.h5'),
            verbose=1
        )
        
        print("\n✅ Fine-tuning завершён!")
        
        return history
    
    def summary(self):
        """Вывод архитектуры модели"""
        if self.model is None:
            raise ValueError("Сначала создайте модель")
        return self.model.summary()
    
    def save_model(self, filepath='models/final_model.h5'):
        """Сохранение модели"""
        if self.model is None:
            raise ValueError("Модель не создана")
        self.model.save(filepath)
        print(f"✅ Модель сохранена: {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """Загрузка сохранённой модели"""
        return keras.models.load_model(filepath)


if __name__ == "__main__":
    # Пример использования
    num_classes = 10
    
    model_builder = SeedClassificationModel(
        num_classes=num_classes,
        img_size=(224, 224),
        model_type='resnet50'
    )
    
    model = model_builder.build_model()
    model_builder.compile_model()
    
    print("\n📊 Архитектура модели:")
    model_builder.summary()
