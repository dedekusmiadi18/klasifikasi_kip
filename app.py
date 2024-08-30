from flask import Flask, request, render_template, send_file
import pandas as pd
from model import  svm_model, preprocessor

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        # Ambil file dari form
        file = request.files['file']
        
        # Baca file Excel
        input_data = pd.read_excel(file)
        
        # Kategorikan penghasilan
        # input_data['Penghasilan Orang Tua'] = input_data['Penghasilan Orang Tua'].apply(categorize_penghasilan)
        
        # Terapkan aturan tambahan sebelum prediksi
        # Default status adalah 'Layak'
        input_data['Status'] = 'Layak'
        
        # Kondisi untuk tidak layak
        # aturan = (input_data['Penghasilan Orang Tua'] == 'tinggi') | (input_data['Pekerjaan Orang Tua'] == 'PNS')
        # input_data.loc[aturan, 'Status'] = 'Tidak Layak'
        
        # Transformasi data
        input_data_processed = preprocessor.transform(input_data)
        
        # Prediksi untuk data yang tidak memenuhi aturan
        data_to_predict = input_data.loc[input_data['Status'] == 'Layak']
        if not data_to_predict.empty:
            predictions = svm_model.named_steps['classifier'].predict(preprocessor.transform(data_to_predict))
            input_data.loc[data_to_predict.index, 'Status'] = predictions
        
        # Simpan hasil ke file Excel baru
        output_file = 'output.xlsx'
        input_data.to_excel(output_file, index=False)

        # Kirim file hasil ke user
        return send_file(output_file, as_attachment=True)
    else:
        # Ambil data dari input manual
        data = {
            'Nama': [request.form['nama']],
            'Usia': [request.form['usia']],
            'Penghasilan Orang Tua': [request.form['penghasilan_orang_tua']],
            'Pekerjaan Orang Tua': [request.form['pekerjaan_orang_tua']],
            'Jumlah Keluarga': [request.form['jumlah_keluarga']],
            'Jumlah Tanggungan Anak': [request.form['jumlah_tanggungan_anak']],
            'Status Rumah': [request.form['status_rumah']],
            'Jenis Lantai': [request.form['jenis_lantai']],
            'Jenis Dinding': [request.form['jenis_dinding']]
        }
        
        input_data = pd.DataFrame(data)
        
        # Kategorikan penghasilan
        # input_data['Penghasilan Orang Tua'] = input_data['Penghasilan Orang Tua'].apply(categorize_penghasilan)

        # Terapkan aturan tambahan sebelum prediksi
        if input_data['Penghasilan Orang Tua'][0] == 'tinggi' or input_data['Pekerjaan Orang Tua'][0] == 'PNS':
            status = 'Tidak Layak'
        else:
            # Transformasi data
            input_data_processed = preprocessor.transform(input_data)
            
            # Prediksi
            predictions = svm_model.named_steps['classifier'].predict(input_data_processed)
            
            # Periksa nilai predictions dan ambil string dari array
            status = predictions[0] if predictions else 'Status Tidak Dikenal'
        
        # Tampilkan hasil di halaman web
        result_message = f"Atas Nama {request.form['nama']}: {status}"
        return render_template('index.html', result=result_message)

if __name__ == '__main__':
    app.run(debug=True)
