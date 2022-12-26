# Challenge-Platinum-API-for-Sentiment-Analyst

Melakukan analisis sentimen dengan pembuatan model menggunakan Naural Network dan LSTM dengan mendeploynya dengan API Flask dan Swagger UI. Pada endpoint dapat menginput teks dan mengupload file untuk menganalisis sentimen dan mengcleansingnya.

# 1 Prepare Dataset
Menganalisa data dan melihat label jumlah positif, negatif, dan netral.

# 2 Cleansing Text
Membersihkan teks dengan menggunakan cleansing function.

# 3 Feature Extraction
Mengubah setiap kata menjadi vector menggunakan CountVectorizer(NN) dan Tokenizer(LSTM).

# 4 Prepare and Validation Data
Menentukan jumlah data yang akan di train dan di test

# 5 Model Selection
Untuk NN menggunakan MLPClassifier from sklearn dan LSTM menggunakan deep learning seperti TensorFlow

# 6 Training Model dan Save Model
Melakukan training dan validation data setelah itu melakukan saving data
