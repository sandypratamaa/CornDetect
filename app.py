from flask import Flask,render_template,request,jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import gym
import pickle
import os
import tensorflow as tf
from PIL import Image
from flask_ngrok import run_with_ngrok

app = Flask(__name__)

@app.route('/', methods=["GET"])
def hello_world():
    return render_template("index.html")

# =[Variabel Global]=============================

app = Flask(__name__, static_url_path='/static')

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS']  = ['.jpg','.JPG']
app.config['UPLOAD_PATH']        = './static/img/uploads/'

# load model
model = tf.keras.models.load_model("modelcorn.h5")

# define classes
corndiseases_classes = [ "Corn Common Rust", "Corn Gray Leaf Spot","Corn Healthy", "Corn Northern Leaf Blight"]

# define image size
IMG_SIZE = (299, 299)

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]
@app.route("/")
def beranda():
	return render_template('index.html')

# ROUTING untuk chatbot
@app.route("/get")
def get_bot_response():
    user_input = str(request.args.get('msg'))
    result = generate_response(user_input)
    return result

# [Routing untuk API]	
@app.route("/api/deteksi",methods=['POST'])
def apiDeteksi():
	# Set nilai default untuk hasil prediksi dan gambar yang diprediksi
	hasil_prediksi  = '(none)'
	gambar_prediksi = '(none)'

	# Get File Gambar yg telah diupload pengguna
	uploaded_file = request.files['file']
	filename      = secure_filename(uploaded_file.filename)
	
	# Periksa apakah ada file yg dipilih untuk diupload
	if filename != '':
	
		# Set/mendapatkan extension dan path dari file yg diupload
		file_ext        = os.path.splitext(filename)[1]
		gambar_prediksi = '/static/img/uploads/' + filename
		
		# Periksa apakah extension file yg diupload sesuai (jpg)
		if file_ext in app.config['UPLOAD_EXTENSIONS']:
			
			# Simpan Gambar
			uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
			
			# Memuat Gambar
			test_image         = Image.open('.' + gambar_prediksi).resize(IMG_SIZE)

			img_array = np.expand_dims(test_image, 0)
			

			predictions = model.predict(img_array)
			hasil_prediksi = corndiseases_classes[np.argmax(predictions[0])]

			print(hasil_prediksi)
			
			# Return hasil prediksi dengan format JSON
			return jsonify({
				"prediksi": hasil_prediksi,
				"gambar_prediksi" : gambar_prediksi
			})
		else:
			# Return hasil prediksi dengan format JSON
			gambar_prediksi = '(none)'
			return jsonify({
				"prediksi": hasil_prediksi,
				"gambar_prediksi" : gambar_prediksi
			})


# =[Main]========================================		

if __name__ == '__main__':
    	

	# Run Flask di localhost 
	run_with_ngrok(app)
	app.run()