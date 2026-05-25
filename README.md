# Markov Clustering (MCL) dengan CUDA C++ dan cuSPARSE

Repositori ini berisi implementasi algoritma Markov Clustering (MCL) menggunakan komputasi paralel GPU (CUDA C++) dan format Sparse Matrix (cuSPARSE). Proyek ini dirancang untuk menangani dataset graf berskala besar, memangkas memori secara drastis dengan mengeliminasi elemen nol, dan mempercepat konvergensi matriks.

## Visualisasi Konsep Matriks: Dense vs Sparse (CSC)

Algoritma ini menggunakan format **Compressed Sparse Column (CSC)**. Format ini mengubah representasi graf dua dimensi menjadi tiga array linear agar dapat diproses secepat kilat oleh GPU.

**Contoh Matriks Padat (Dense Matrix):**
| Produk (Kolom) | 0 | 1 | 2 |
| :--- | :---: | :---: | :---: |
| **Baris 0** | 0.0 | 0.4 | 0.0 |
| **Baris 1** | 0.5 | 0.0 | 0.0 |
| **Baris 2** | 0.9 | 0.0 | 0.0 |

**Transformasi ke Format CSC di Memori:**
Untuk matriks di atas, program tidak menyimpan angka `0.0`. Memori direpresentasikan dalam tiga vektor (*array*):
* `col_ptr = {0, 2, 3, 3}` (Peta batas awal dan akhir setiap kolom)
* `row_idx = {1, 2, 0}` (Posisi baris dari elemen yang memiliki nilai)
* `val = {0.5, 0.9, 0.4}` (Bobot/nilai probabilitasnya)

---

## Alur Kerja Program (Penjelasan Baris Kode)

### TAHAP 1: Pembacaan Data dan Konversi CSC
Tahap ini berjalan di CPU untuk membaca file CSV dan mengubah *Edgelist* menjadi format matriks *Sparse*.

* **Pembacaan CSV (`ifstream`):** Program membuka file *edgelist*, membuang baris judul kolom (*header*), lalu membaca id produk A, id produk B, dan nilai bobot (*weight*). Data ini dimasukkan ke dalam struktur `temp_edges`.
* **Pengurutan Data (`sort`):**
  Fungsi lambda digunakan untuk mengurutkan `temp_edges` secara ketat. Aturan prioritas pertama adalah mengurutkan berdasarkan **Kolom** dari kecil ke besar. Jika kolomnya sama, diurutkan berdasarkan **Baris**. Pengurutan ini adalah syarat mutlak dari NVIDIA agar format CSC terbentuk sempurna.
* **Pembentukan Array CSC (Histogram & Prefix Sum):**
  Program melakukan *looping* untuk memisahkan baris (`h_row_idx`) dan bobot (`h_val`). Secara bersamaan, program menghitung jumlah koneksi per kolom ke dalam `h_col_ptr`. Setelah dihitung, dilakukan penjumlahan beruntun (*Prefix Sum*) pada `h_col_ptr` agar angka-angka tersebut berubah fungsi menjadi **titik koordinat memori** bagi GPU.
* **Perhitungan NNZ:** Total koneksi (*Number of Non-Zeros*) secara otomatis didapatkan dari indeks paling ujung array *Prefix Sum* (`h_col_ptr[N]`).

### TAHAP 2: Alokasi Memori GPU dan Persiapan
Tahap ini adalah proses pemindahan data dari RAM Komputer ke VRAM GPU.

* **Alokasi Kavling GPU (`cudaMalloc`):** Memesan ruang fisik di dalam VRAM sesuai ukuran array yang telah dihitung di Tahap 1. Tipe data `int` dan `float` dialokasikan memori byte-nya secara presisi.
* **Transfer Data (`cudaMemcpy`):** Mengirim array `h_col_ptr`, `h_row_idx`, dan `h_val` melintasi *motherboard* menuju memori perangkat (*Host to Device*).
* **Inisialisasi `cuSPARSE`:** Membuat `cusparseHandle_t` sebagai penanda bahwa kita akan memanggil mesin perkalian matriks resmi dari NVIDIA.
* **Pengaturan Pasukan GPU (`threads1D` & `blocks1D`):** Membagi beban kerja ke dalam kelompok (Block). Setiap kelompok berisi 256 pekerja (Thread) yang akan beroperasi secara paralel. Rumus `(N + threads1D - 1) / threads1D` memastikan pembulatan ke atas agar tidak ada data yang terlewat.
* **Eksekusi Kernel `initial_normalize_kernel`:** Ribuan pekerja GPU secara serentak menjumlahkan bobot di masing-masing kolom, lalu membagi setiap elemen dengan totalnya. Graf kini resmi menjadi Matriks Markov (Stokastik).

*(Proses berlanjut ke Tahap 3: Iterasi MCL dan Tahap 4: Ekspor Data...)*
