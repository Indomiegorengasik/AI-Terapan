from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
from tensorflow import keras
from flask_cors import CORS
import os
import time
import tensorflow as tf
from tensorflow.keras.layers import StringLookup
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
import json 

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
model = load_model('handwriting_recognition.h5', compile=False)

# Database connection string
db_connection_string = "mysql+mysqlconnector://gree5372_handwriting:d06!9lsA_&&k@193.168.194.230:3306/gree5372_handwriting"

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = db_connection_string
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 10,
    'pool_recycle': 3600,
    'pool_pre_ping': True
}

app.config['JWT_SECRET_KEY'] = '2J4ia7lING6hcszUNjrSZLVxSAqFVRe2HkdixfejcsO/FJzz4PzA9AFi60IcER8Vd43Kl6mHOyo3UB12FsJ0qEfqG9umiO4nZmFSWIGymYNglpmPK5gkFsv0vO1b/W3yQZcEGwOoII+dlpelfEtvWSreZUnTNAf/sho8qhTHN62yq0FpOZAQ6BY8+xDxckO5cyEp5zJbkpzabvSyFNxbWDxZIY+3AGZ35xzvQVJm9XqddUTTSWvixk5g5Po6OqMGLZt4wmTuxa9rnB+8exMHq8fJCYQm1c/dL9ShVKGMIiXR1gi3c+SdMTChIkifdVQh92BpBBa3g8F/l4U8XqNg4Q=='  # Change this!
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)

db = SQLAlchemy(app)
jwt = JWTManager(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    gender = db.Column(db.Boolean(), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

base_path = "."
batch_size = 64
padding_token = 99
image_width = 128
image_height = 32
max_len = 21
characters = set()
max_len = 0
words_list = []
train_labels_cleaned = []

words = open(f"{base_path}/words.txt", "r").readlines()
for line in words:
    if line[0] == "#":
        continue
    if line.split(" ")[1] != "err":  # We don't need to deal with errored entries.
        words_list.append(line)

np.random.shuffle(words_list)

split_idx = int(0.9 * len(words_list))
train_samples = words_list[:split_idx]
base_image_path = os.path.join(base_path, "words")

def get_image_paths_and_labels(samples):
    paths = []
    corrected_samples = []
    for (i, file_line) in enumerate(samples):
        line_split = file_line.strip()
        line_split = line_split.split(" ")

        # Each line split will have this format for the corresponding image:
        # part1/part1-part2/part1-part2-part3.png
        image_name = line_split[0]
        partI = image_name.split("-")[0]
        partII = image_name.split("-")[1]
        img_path = os.path.join(
            base_image_path, partI, partI + "-" + partII, image_name + ".png"
        )
        if os.path.getsize(img_path):
            paths.append(img_path)
            corrected_samples.append(file_line.split("\n")[0])

    return paths, corrected_samples

train_img_paths, train_labels = get_image_paths_and_labels(train_samples)

for label in train_labels:
    label = label.split(" ")[-1].strip()
    for char in label:
        characters.add(char)

    max_len = max(max_len, len(label))
    train_labels_cleaned.append(label)

characters = sorted(list(characters))

AUTOTUNE = tf.data.AUTOTUNE

# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def prepare_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_images_labels, num_parallel_calls=AUTOTUNE
    )
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

@app.route('/register', methods=['POST'])
def register():
    try:
        with app.app_context():
            first_name = request.json.get('first_name', None)
            last_name = request.json.get('last_name', None)
            gender = request.json.get('gender', None)
            email = request.json.get('email', None)
            password = request.json.get('password', None)
            
            if not first_name or not last_name or not gender:
                return jsonify({"msg": "Missing first name, last name or gender"}), 400
            
            if not email or not password:
                return jsonify({"msg": "Missing email or password"}), 400
            
            if len(password) < 8:
                return jsonify({"msg": "Password should more than 8 characters"}), 400
            
            if User.query.filter_by(email=email).first():
                return jsonify({"msg": "Email already exists"}), 400
            
            hashed_password = generate_password_hash(password)
            new_user = User(email=email, password=hashed_password, first_name=first_name, last_name=last_name, gender=gender)
            db.session.add(new_user)
            db.session.commit()
            
            return jsonify({"msg": "User created successfully"}), 201
    except Exception as e:
        print(e)
        return str(e), 500
    
@app.route('/login', methods=['POST'])
def login():
    try:
        with app.app_context():
            email = request.json.get('email', None)
            password = request.json.get('password', None)
            
            user = User.query.filter_by(email=email).first()
            if not user or not check_password_hash(user.password, password):
                return jsonify({"msg": "Bad email or password"}), 401
            
            access_token = create_access_token(identity=email)
            return jsonify(user={"first_name": user.first_name, "last_name": user.last_name, "gender": user.gender, "email": user.email}, access_token=access_token), 200
    except Exception as e:
        print(e)
        return str(e), 500

@app.route('/predict', methods=['POST'])
@jwt_required()
def predict():
    try:
        current_user = get_jwt_identity()
        print(request.files)
        file_image = request.files['file']
        filename = str(time.time()) + file_image.filename
        file_path = os.path.join('./uploads', filename)
        file_image.save(file_path)
        
        preprocessed_image = preprocess_image(file_path)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0) 
        result = model.predict(preprocessed_image)
        decoded_result = decode_batch_predictions(result)
        
        os.unlink(file_path)
        return ' '.join(decoded_result)
    except Exception as e:
        print(e)
        return str(e), 500
if __name__ == "__main__":
    app.run(port=8000, debug=True)