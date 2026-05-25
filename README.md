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

### TAHAP 0: Pendeklarasian Parameter MCL
Sebelum algoritma berjalan, program menetapkan empat parameter utama yang mengontrol perilaku *Markov Clustering*:
* **`inflation_p` (1.3f):** Tingkat inflasi (pangkat) untuk mempertegas probabilitas. Nilai 1.3 dipilih untuk menjaga keseimbangan antara memperkuat klaster fungsional dan mencegah hilangnya produk akibat inflasi yang terlalu ekstrem (*over-inflation*).
* **`prune_threshold` (1e-3f):** Batas pemotongan (*pruning*). Nilai probabilitas di bawah 0.001 akan dianggap sebagai *noise* dan diubah menjadi 0.0 untuk menghemat memori GPU.
* **`convergence_threshold` (1e-5f):** Batas toleransi kestabilan. Jika *Global Chaos* (selisih perubahan nilai antar iterasi) sudah di bawah 0.00001, graf dianggap sudah stabil (konvergen).
* **`max_iterations` (1000):** Batas aman pengulangan maksimal untuk mencegah *looping* tanpa batas.

---

### TAHAP 1: Pembacaan Dataset dan Pemetaan Indeks (Data Preprocessing)
Tahap ini krusial untuk mengubah data mentah berbentuk *Edgelist* menjadi struktur data berurutan yang siap diproses oleh matriks.

**1. Format Edgelist Input**
Data awal berupa file CSV (`edgelist_Aw_Cosine_10000.csv`) yang berisi koneksi antar-produk dan nilai kemiripannya (*Cosine Similarity*).
| Product_A | Product_B | Weight |
| :---: | :---: | :---: |
| 1045 | 1045 | 1.0 |
| 1045 | 890 | 0.45 |

**2. Membuka File Stream (`ifstream`)**
```cpp
string file_name = "edgelist_Aw_Cosine_10000.csv";
ifstream file(file_name);
if (!file.is_open()) { cerr << "Error: File tidak ditemukan!" << endl; return 1; }

```cpp
