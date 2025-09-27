import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import ResNet50 # **CORRECTED LINE** (removed asterisks)
import os
from PIL import ImageFile
import matplotlib.pyplot as plt

# This line tells the image library to ignore the "truncated file" error
ImageFile.LOAD_TRUNCATED_IMAGES = True 

# ----------------- 1. Configuration -----------------
# !!! RENAME this to the folder containing your 5 pose subfolders !!!
DATA_DIRECTORY = r"C:\Users\varsh\Downloads\archive\DATASET\TRAIN" 
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 15 # For initial training with frozen layers
FINE_TUNE_EPOCHS = 10 # For final training with unfrozen layers
MODEL_PATH = 'yoga_pose_detector_resnet50.h5'

# Check if the data directory exists
if not os.path.isdir(DATA_DIRECTORY):
    print(f"Error: Data directory '{DATA_DIRECTORY}' not found.")
    print("Please create the folder and put your 5 pose folders inside it.")
    exit()

# ----------------- 2. Data Preparation and Augmentation -----------------

# ImageDataGenerator handles loading, rescaling, and data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, # Reserve 20% of data for validation
    rotation_range=10, # Slightly reduced from previous version for stability
    width_shift_range=0.1, # Slightly reduced
    height_shift_range=0.1, # Slightly reduced
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load training data
train_generator = datagen.flow_from_directory(
    DATA_DIRECTORY,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Load validation data
validation_generator = datagen.flow_from_directory(
    DATA_DIRECTORY,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False 
)

num_classes = train_generator.num_classes
class_names = list(train_generator.class_indices.keys())

print("-" * 50)
print(f"Found {num_classes} classes: {class_names}")
print("-" * 50)


# ----------------- 3. Model Definition (Transfer Learning) -----------------

# Load ResNet50 base model, pre-trained on ImageNet
base_model = ResNet50(
    input_shape=(224, 224, 3),
    include_top=False, 
    weights='imagenet' 
)

# **Phase 1: Freeze the base layers for initial training**
base_model.trainable = False

# Build the new classification head
model = Sequential([
    base_model, 
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax') 
])

# ----------------- 4. Compile and Train (Phase 1: Frozen) -----------------
print("\n--- Phase 1: Training new layers (Frozen Base) ---")
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

model.summary()

history_1 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=INITIAL_EPOCHS
)

# ----------------- 5. Fine-Tuning (Phase 2: Unfrozen) -----------------
print("\n--- Phase 2: Fine-tuning base model ---")

# **Unfreeze the entire base model**
base_model.trainable = True

# **Freeze all but the last 50 layers for fine-tuning**
# This is done to preserve general features learned by the early layers
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Re-compile the model with a **very low learning rate**
# Use 1e-5 (0.00001) for stable fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history_2 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history_1.epoch[-1]
)

# ----------------- 6. Save Model -----------------
model.save(MODEL_PATH)
print(f"\nModel saved to **{MODEL_PATH}**")

# **Optional: Plotting the training history (requires matplotlib)**
try:
    # Combine history data
    acc = history_1.history['accuracy'] + history_2.history['accuracy']
    val_acc = history_1.history['val_accuracy'] + history_2.history['val_accuracy']
    loss = history_1.history['loss'] + history_2.history['loss']
    val_loss = history_1.history['val_loss'] + history_2.history['val_loss']
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

except Exception as e:
    # This block handles the case if matplotlib is not installed
    # or if the history object is unexpected, ensuring the main code finishes.
    print(f"\nNote: Could not plot history (Error: {e}). Ensure matplotlib is installed.")