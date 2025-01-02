import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths
data_dir = "C:/Users/karav_867vu4n/OneDrive/Desktop/images"  # Root folder containing subfolders for each food category
image_size = (224, 224)  # ResNet/EfficientNet default size
batch_size = 32

# Data Augmentation and Data Generators
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,  # 20% for validation
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Model - Using a Pretrained Model
base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*image_size, 3))
base_model.trainable = False  # Freeze base model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_gen.class_indices), activation='softmax')  # Number of classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    steps_per_epoch=train_gen.samples // batch_size,
    validation_steps=val_gen.samples // batch_size
)

# Evaluate the model
test_gen = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

y_pred = model.predict(test_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_gen.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=test_gen.class_indices.keys(), yticklabels=test_gen.class_indices.keys())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_true, y_pred_classes, target_names=test_gen.class_indices.keys()))

# Fine-Tune the Model
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
history_fine_tune = model.fit(train_gen, validation_data=val_gen, epochs=5)

# Calorie Mapping
calorie_dict = {
    "apple_pie": 237,
    "baby_back_ribs": 292,
    "baklava": 428,
    "beef_carpaccio": 175,
    "beef_tartare": 217,
    "beet_salad": 43,
    "beignets": 452,
    "bibimbap": 199,
    "bread_pudding": 153,
    "breakfast_burrito": 232,
    "bruschetta": 150,
    "caesar_salad": 190,
    "cannoli": 333,
    "caprese_salad": 250,
    "carrot_cake": 326,
    "ceviche": 120,
    "cheesecake": 321,
    "cheese_plate": 402,
    "chicken_curry": 165,
    "chicken_quesadilla": 270,
    "chicken_wings": 290,
    "chocolate_cake": 371,
    "chocolate_mousse": 267,
    "churros": 450,
    "clam_chowder": 174,
    "club_sandwich": 260,
    "crab_cakes": 220,
    "creme_brulee": 330,
    "croque_madame": 335,
    "cup_cakes": 356,
    "deviled_eggs": 150,
    "donuts": 452,
    "dumplings": 120,
    "edamame": 121,
    "eggs_benedict": 252,
    "escargots": 160,
    "falafel": 333,
    "filet_mignon": 250,
    "fish_and_chips": 290,
    "foie_gras": 462,
    "french_fries": 312,
    "french_onion_soup": 73,
    "french_toast": 222,
    "fried_calamari": 175,
    "fried_rice": 163,
    "frozen_yogurt": 127,
    "garlic_bread": 330,
    "gnocchi": 122,
    "greek_salad": 95,
    "grilled_cheese_sandwich": 330,
    "grilled_salmon": 206,
    "guacamole": 160,
    "gyoza": 134,
    "hamburger": 250,
    "hot_and_sour_soup": 50,
    "hot_dog": 290,
    "huevos_rancheros": 175,
    "hummus": 177,
    "ice_cream": 207,
    "lasagna": 135,
    "lobster_bisque": 93,
    "lobster_roll_sandwich": 198,
    "macaroni_and_cheese": 164,
    "macarons": 450,
    "miso_soup": 40,
    "mussels": 172,
    "nachos": 346,
    "omelette": 154,
    "onion_rings": 411,
    "oysters": 68,
    "pad_thai": 320,
    "paella": 158,
    "pancakes": 227,
    "panna_cotta": 288,
    "peking_duck": 337,
    "pho": 61,
    "pizza": 266,
    "pork_chop": 231,
    "poutine": 262,
    "prime_rib": 312,
    "pulled_pork_sandwich": 279,
    "ramen": 436,
    "ravioli": 245,
    "red_velvet_cake": 367,
    "risotto": 160,
    "samosa": 262,
    "sashimi": 150,
    "scallops": 111,
    "seaweed_salad": 70,
    "shrimp_and_grits": 202,
    "spaghetti_bolognese": 129,
    "spaghetti_carbonara": 131,
    "spring_rolls": 87,
    "steak": 271,
    "strawberry_shortcake": 270,
    "sushi": 150,
    "tacos": 226,
    "takoyaki": 175,
    "tiramisu": 240,
    "tuna_tartare": 144,
    "waffles": 291
}

# Inference
def predict_and_calorie(image_path, model, class_indices, calorie_dict):
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    class_name = list(class_indices.keys())[list(class_indices.values()).index(predicted_class)]
    calories = calorie_dict.get(class_name, "Unknown")

    return class_name, calories

# Test the model
image_path = "path/to/single_food_image.jpg"
class_name, calories = predict_and_calorie(image_path, model, train_gen.class_indices, calorie_dict)
print(f"Predicted Class: {class_name}, Estimated Calories: {calories}")
