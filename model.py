import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Fungsi untuk mengkategorikan penghasilan
# def categorize_penghasilan(value):
#     if value <= 2000000:
#         return 'rendah'
#     elif value <= 3000000:
#         return 'sedang'
#     elif value <= 4000000:
#         return 'tinggi'
#     else:
#         return 'sangat tinggi' 
     # Handle values greater than 4000000



# Muat data
df = pd.read_csv('dataset_kip2.csv', index_col=0)

# Menghapus kolom yang tidak perlu
df_new = df.drop(columns=['Nama', 'Usia', 'Tanggal Lahir', 'Alamat'])

# Kategorikan penghasilan
# df['Penghasilan Orang Tua'] = df['Penghasilan Orang Tua'].apply(categorize_penghasilan)

# Identifikasi kolom yang berisi data string
categorical_columns = ['Pekerjaan Orang Tua', 'Status Rumah', 'Jenis Lantai', 'Jenis Dinding']
numerical_columns = ['Usia', 'Jumlah Keluarga', 'Jumlah Tanggungan Anak', 'Penghasilan Orang Tua']

# Pisahkan fitur dan label
x = df.drop(columns=['Status'])
y = df['Status']

# Split data menjadi train dan test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Gunakan OneHotEncoder untuk kolom kategorikal dan StandardScaler untuk kolom numerikal
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns),
        ('num', StandardScaler(), numerical_columns)
    ])

# Buat model SVM dengan pipeline
svm_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced'))
])

# Latih model
svm_model.fit(x_train, y_train)
