from flask import Flask, request, render_template, send_file
import pandas as pd
from model import categorize_penghasilan, svm_model, preprocessor
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil file dari form
    file = request.files['file']
    
    # Baca file Excel
    input_data = pd.read_excel(file)
    
    # Kategorikan penghasilan
    input_data['Penghasilan'] = input_data['Penghasilan'].apply(categorize_penghasilan)

    # Terapkan aturan bisnis
    input_data['Status'] = 'Layak'  # Set default status
    input_data.loc[input_data['Penghasilan'] == 'tinggi', 'Status'] = 'Tidak Layak'
    #input_data.loc[input_data['Pekerjaan'] == 'Pegawai Negeri', 'Status'] = 'Tidak Layak'

    # Pisahkan data yang layak untuk prediksi
    df_to_predict = input_data[input_data['Status'] == 'Layak'].drop(columns=['Status'])

    if not df_to_predict.empty:
        # Transformasi data
        input_data_processed = preprocessor.transform(df_to_predict)

        # Prediksi
        predictions = svm_model.named_steps['classifier'].predict(input_data_processed)

        # Tambahkan hasil prediksi ke data asli
        input_data.loc[input_data['Status'] == 'Layak', 'Status'] = predictions

    # Simpan hasil ke file Excel baru
    output_file = 'output.xlsx'
    input_data.to_excel(output_file, index=False)

    # Kirim file hasil ke user
    return send_file(output_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
