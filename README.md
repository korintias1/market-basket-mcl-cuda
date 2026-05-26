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
| 1045 | 890 | 0.45 |
| 1045 | 1045 | 1.0 |

**2. Membuka File Stream (`ifstream`)**
```cpp
string file_name = "edgelist_Aw_Cosine_10000.csv";
ifstream file(file_name);
if (!file.is_open()) { cerr << "Error: File tidak ditemukan!" << endl; return 1; }
```

Program menggunakan ifstream` untuk membuka jalur komunikasi ke file CSV. Kode `getline(file, line)` pertama dipanggil untuk melompati baris pertama (judul kolom) agar tidak ikut terhitung sebagai data.

**3. Pemetaan ID Produk (Mapping)**

Di dunia nyata, ID Produk (Product_A) sering kali berupa angka acak atau melompat-lompat `misal: 1045, 890, 3002`. Namun, matriks komputasi mewajibkan indeks dimulai berurutan dari `0, 1, 2, ... dst`.
```cpp
unordered_map<int, int> map_index;
vector<int> reverse_map;
int N = 0;
```
*   **`map_index` (Kamus Maju)**: Bertugas menerjemahkan ID asli menjadi indeks matriks.
    *   *Contoh*: ID `1045` diterjemahkan menjadi indeks `0`.
*   **`reverse_map` (Kamus Mundur)**: Menyimpan urutan ID asli. Ini sangat penting untuk **Tahap 4 (Ekspor)** agar komputer bisa menerjemahkan kembali indeks `0` menjadi ID `1045` saat dicetak ke CSV hasil.
*   **Variabel `N`**: Bertindak sebagai penghitung jumlah produk unik yang ditemukan.

**4. Parsing dan Pengekstrakan Baris**
```cpp
while (getline(file, line)) {
    stringstream ss(line);
    getline(ss, str_A, ','); getline(ss, str_B, ','); getline(ss, str_W, ',');
    // ...
```
Program membaca file baris demi baris menggunakan looping `while`. Fitur `stringstream` dibantu dengan pemisah koma (`,`) digunakan untuk memotong satu teks utuh menjadi potongan-potongan data.

**Simulasi Visualisasi Ekstraksi 1 Baris Data:**

Bayangkan komputer sedang membaca baris data CSV Anda. Teks utuh tersebut ditangkap oleh variabel `line`. Mesin `stringstream` kemudian memindai teks tersebut dari kiri ke kanan dan memotongnya setiap kali bertemu tanda koma (`,`).
| Teks Mentah (`line`) | Proses Pemotongan | Variabel Tujuan | Hasil Akhir (String) |
| :--- | :--- | :--- | :--- |
| `"1045,890,0.45"` | ➔ Potongan Pertama | `str_A` | `"1045"` |
| | ➔ Potongan Kedua | `str_B` | `"890"` |
| | ➔ Potongan Ketiga | `str_W` | `"0.45"` |

**5. Konversi dan Penyimpanan ke Array (`temp_edges`)**

```cpp
    int id_a = stoi(str_A);
    int id_b = stoi(str_B);
    float w = parse_weight(str_W);

    if (map_index.find(id_a) == map_index.end()) { map_index[id_a] = N++; reverse_map.push_back(id_a); }
    if (map_index.find(id_b) == map_index.end()) { map_index[id_b] = N++; reverse_map.push_back(id_b); }

    temp_edges.push_back({map_index[id_a], map_index[id_b], w});
}
```

Setelah teks terpotong menjadi tiga variabel string, program wajib mengubah wujud teks tersebut menjadi angka matematis murni. Selanjutnya, program melakukan pemetaan (*Mapping*) agar ID Produk yang angkanya acak atau sangat besar bisa dirapatkan menjadi urutan indeks matriks yang rapi (selalu dimulai dari 0).

**Simulasi Visualisasi Proses Pemetaan (Data Baris Pertama: "1045", "890", "0.45"):**
Anggaplah ini adalah baris data pertama yang diproses oleh program, sehingga memori indeks awal masih bernilai nol (`N = 0`). Berikut adalah urutan kejadian di dalam memori komputer:

| Tahap Proses | Variabel / Perintah | Nilai / Hasil Memori | Penjelasan Logika |
| :--- | :--- | :--- | :--- |
| **1. Konversi Tipe Data** | `id_a` (Integer) | `1045` | Teks `"1045"` diubah menggunakan `stoi` menjadi bilangan bulat. |
| | `id_b` (Integer) | `890` | Teks `"890"` diubah menjadi bilangan bulat. |
| | `w` (Float) | `0.45` | Teks `"0.45"` diubah menggunakan `parse_weight` menjadi desimal. |
| **2. Pendaftaran ID A** | `map_index[1045]` | `0` | Komputer mengecek kamus. Karena ID 1045 belum ada, ia didaftarkan sebagai **Indeks 0**. Nilai `N` bertambah menjadi 1. |
| | `reverse_map` | `[1045]` | Komputer mengingat bahwa Indeks 0 adalah milik ID 1045 untuk diekspor nanti. |
| **3. Pendaftaran ID B** | `map_index[890]` | `1` | ID 890 belum ada di kamus, maka ia didaftarkan sebagai **Indeks 1**. Nilai `N` bertambah menjadi 2. |
| | `reverse_map` | `[1045, 890]` | Komputer mengingat bahwa Indeks 1 adalah milik ID 890. |
| **4. Masuk ke Keranjang** | `temp_edges.push_back` | `{0, 1, 0.45}` | Baris data sukses masuk ke dalam array penampungan sementara menggunakan wujud indeks matriksnya, bukan ID aslinya. |

Berkat proses pemetaan ini, komputer GPU terbebas dari keharusan membuat matriks raksasa kosong hanya untuk menyesuaikan dengan ID produk yang melompat-lompat. Matriks dipastikan selalu rapat dan padat, mulai dari indeks `0` sampai jumlah produk unik terakhir.

**6. Pengurutan Data (Sorting) dan Pembentukan Format CSC**

```cpp
    // Urutkan edge untuk format CSR/CSC
    sort(temp_edges.begin(), temp_edges.end(), [](const Edge& a, const Edge& b) {
        if (a.c != b.c) return a.c < b.c;
        return a.r < b.r;
    });

    vector<int> h_col_ptr(N + 1, 0);
    vector<int> h_row_idx;
    vector<float> h_val;

    for (const auto& e : temp_edges) {
        h_col_ptr[e.c + 1]++;
        h_row_idx.push_back(e.r);
        h_val.push_back(e.w);
    }
    for (int i = 0; i < N; ++i) h_col_ptr[i + 1] += h_col_ptr[i];

    int nnz = h_col_ptr[N];
    cout << "Data terbaca. N = " << N << " Produk | Non-Zero (Koneksi) = " << nnz << endl;
```

Setelah seluruh data masuk ke dalam keranjang `temp_edges`, data tersebut masih acak. GPU dan pustaka NVIDIA cuSPARSE mewajibkan struktur data yang sangat disiplin. Tahap ini bertugas mengurutkan dan menyusun data ke dalam format memori **Compressed Sparse Column (CSC)**.

**Langkah-langkah Proses Pembentukan CSC:**

1. **Pengurutan Ketat (Sorting):** Fungsi `sort` dibekali aturan khusus (*lambda function*). Aturan pertama: urutkan data berdasarkan **Kolom** (`c`) dari terkecil ke terbesar. Aturan kedua: jika ada data di kolom yang sama, urutkan berdasarkan **Baris** (`r`).
2. **Pembuatan 3 Array CSC:** Program menyiapkan memori kosong di CPU (RAM) untuk tiga pilar utama format Sparse:
   * `h_col_ptr`: Peta penunjuk batas kolom (ukurannya `N + 1`, diisi `0` semua di awal).
   * `h_row_idx`: Daftar indeks baris (dinamis).
   * `h_val`: Daftar bobot/probabilitas koneksi (dinamis).
3. **Pencatatan Data & Histogram:** *Looping* pertama mengambil data satu per satu. Indeks baris (`e.r`) dan bobot (`e.w`) langsung didorong masuk ke vektornya masing-masing. Bersamaan dengan itu, program menghitung jumlah koneksi per kolom (`h_col_ptr[e.c + 1]++`).
4. **Penjumlahan Beruntun (Prefix Sum):** *Looping* kedua bertugas mengubah "jumlah koneksi" di `h_col_ptr` menjadi "titik koordinat memori" melalui penjumlahan kumulatif.
5. **Ekstraksi Total Koneksi:** Kotak memori paling ujung dari hasil *Prefix Sum* secara otomatis berisi total keseluruhan koneksi yang valid (*Number of Non-Zeros* atau NNZ).

---

**Simulasi Visualisasi 1: Pencatatan Data ke 3 Array (Histogram)**
Anggaplah kita memiliki 3 produk (N = 3) dan data `temp_edges` kita yang sudah diurutkan berisi 3 koneksi: 
1. `c = 0`, `r = 1`, `w = 0.5`
2. `c = 0`, `r = 2`, `w = 0.9`
3. `c = 1`, `r = 0`, `w = 0.4`

Berikut adalah proses pengisian data ke dalam memori saat *looping* pertama berjalan:

| Tahap Iterasi | Data yang Dibaca (`e`) | `h_col_ptr` (Jumlah per Kolom) | `h_row_idx` (Daftar Baris) | `h_val` (Daftar Bobot) |
| :--- | :--- | :--- | :--- | :--- |
| **Kondisi Awal** | - | `[0, 0, 0, 0]` | `[]` | `[]` |
| **Putaran 1** | Kolom `0`, Baris `1`, Bobot `0.5` | `[0, 1, 0, 0]` *(Indeks ke-1 bertambah)* | `[1]` | `[0.5]` |
| **Putaran 2** | Kolom `0`, Baris `2`, Bobot `0.9` | `[0, 2, 0, 0]` *(Indeks ke-1 bertambah)* | `[1, 2]` | `[0.5, 0.9]` |
| **Putaran 3** | Kolom `1`, Baris `0`, Bobot `0.4` | `[0, 2, 1, 0]` *(Indeks ke-2 bertambah)* | `[1, 2, 0]` | `[0.5, 0.9, 0.4]` |

---

**Simulasi Visualisasi 2: Perubahan `h_col_ptr` (Prefix Sum)**
Array `h_row_idx` dan `h_val` sudah selesai diisi. Kini, program mengeksekusi *looping* kedua untuk menjumlahkan `h_col_ptr` secara beruntun agar nilainya berubah menjadi titik koordinat memori.

| Tahap Eksekusi | Isi Array `h_col_ptr` | Penjelasan |
| :--- | :--- | :--- |
| **Selesai Histogram** | `[0, 2, 1, 0]` | Kolom 0 ada 2 data, Kolom 1 ada 1 data, Kolom 2 kosong. |
| **Prefix Sum (i = 0)** | `[0, 2, 1, 0]` | Kotak ke-1 dijumlahkan dengan Kotak ke-0 (2 + 0 = 2). |
| **Prefix Sum (i = 1)** | `[0, 2, 3, 0]` | Kotak ke-2 dijumlahkan dengan Kotak ke-1 (1 + 2 = 3). |
| **Prefix Sum (i = 2)** | `[0, 2, 3, 3]` | Kotak ke-3 dijumlahkan dengan Kotak ke-2 (0 + 3 = 3). |

---



**Hasil Akhir Format CSC:**
Dari serangkaian proses simulasi di atas, wujud akhir format matriks CSC yang tercipta dan siap dikirim untuk diproses oleh GPU NVIDIA adalah sebagai berikut:
* `h_col_ptr` = **`{0, 2, 3, 3}`**
* `h_row_idx` = **`{1, 2, 0}`**
* `h_val`     = **`{0.5, 0.9, 0.4}`**

*(Dari kotak memori paling ujung `h_col_ptr[3]`, secara otomatis didapatkan total **NNZ = 3** koneksi).*

### TAHAP 2: Alokasi Memori GPU dan Persiapan Pasukan Komputasi (CUDA)

Setelah matriks CSC terbentuk rapi di memori utama komputer (RAM/Host), tahap selanjutnya adalah memindahkan data tersebut ke kartu grafis (VRAM/Device) dan menyiapkan parameter eksekusi paralel.

```cpp
    // --- TAHAP 2: ALOKASI MEMORI GPU ---
    int *d_col_ptr, *d_row_idx, *d_nnz_per_col;
    float *d_val, *d_chaos;

    CHECK_CUDA(cudaMalloc(&d_col_ptr, (N + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_row_idx, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_val, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_nnz_per_col, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_chaos, N * sizeof(float)));
```

**1. Pemesanan Kavling Memori GPU (`cudaMalloc`)**
Di dunia CUDA, terdapat konvensi penamaan standar: awalan `h_` untuk variabel di CPU (*Host*) dan awalan `d_` untuk variabel di GPU (*Device*). 
Fungsi `cudaMalloc` bertugas mengalokasikan ruang fisik di memori VRAM GPU secara presisi:
* `d_col_ptr`: Diberi ruang sebesar `N + 1` kotak bertipe *integer* (4 byte).
* `d_row_idx` dan `d_val`: Diberi ruang sebesar `nnz` (total jumlah koneksi valid).
* `d_nnz_per_col` dan `d_chaos`: Disiapkan sebagai memori kosong untuk menampung laporan hasil *pruning* dan nilai konvergensi pada Tahap 3 nanti.

```cpp
    CHECK_CUDA(cudaMemcpy(d_col_ptr, h_col_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_row_idx, h_row_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_val, h_val.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
```

**2. Transfer Data Lintas Perangkat (`cudaMemcpy`)**
Ini adalah momen krusial saat array CSC melintasi perangkat keras komputer. Perintah `cudaMemcpyHostToDevice` secara eksplisit menyuruh CPU untuk menyalin isi dari `h_col_ptr`, `h_row_idx`, dan `h_val` menuju alamat memori `d_` yang sudah dipesan di GPU sebelumnya.

```cpp
    // Setup cuSPARSE
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));
```

**3. Inisialisasi Pustaka cuSPARSE**
Karena algoritma ini memanfaatkan fungsi bawaan dari NVIDIA untuk perkalian matriks tingkat lanjut, kita wajib membuat sebuah *Handle*. `cusparseHandle_t` dapat diibaratkan sebagai "Sesi Sinyal Utama" yang akan selalu dipanggil setiap kali kita menyuruh GPU melakukan operasi matriks *Sparse*.

```cpp
    int threads1D = 256;
    int blocks1D = (N + threads1D - 1) / threads1D;
```

**4. Formasi Pasukan Pekerja GPU (Threads & Blocks)**
GPU beroperasi dengan membagi pekerjaan kepada ribuan pekerja (Thread) yang dikelompokkan ke dalam regu (Block). 
* **`threads1D = 256`**: Setiap 1 regu diatur agar berisi tepat 256 pekerja.
* **Rumus Pembulatan `blocks1D`**: Agar seluruh produk ($N$) mendapat kebagian pekerja, digunakan rumus `(N + 256 - 1) / 256`. Jika total produk misalnya 10.000, maka akan dibentuk 40 regu kerja (total 10.240 pekerja). Kelebihan 240 pekerja nantinya akan diam / tidak memproses apa-apa.

```cpp
    // Normalisasi Awal
    initial_normalize_kernel<<<blocks1D, threads1D>>>(N, d_col_ptr, d_val);
    CHECK_CUDA(cudaDeviceSynchronize());
```

**5. Eksekusi Paralel (Kernel Launch)**
Tanda `<<< ... >>>` adalah aba-aba yang menyuruh ribuan pekerja GPU (sesuai formasi `blocks1D` dan `threads1D`) untuk menyerbu fungsi `initial_normalize_kernel` secara serentak. 
Di dalam fungsi ini, setiap pekerja memegang tepat 1 Kolom Produk. Mereka bertugas menjumlahkan total bobot (*weight*) di kolom tersebut, lalu membagi setiap nilai dengan totalnya. Hasilnya, matriks berubah wujud menjadi **Matriks Markov (Matriks Stokastik)** di mana jumlah setiap kolom pasti sama dengan 1.0. 
Perintah `cudaDeviceSynchronize()` memastikan CPU diam menunggu sampai seluruh pekerja GPU selesai menormalisasi data.

**5B. Penjelasan dan Simulasi Kernel Normalisasi Awal (`initial_normalize_kernel`)**

```cpp
// 1. Normalisasi Awal
__global__ void initial_normalize_kernel(int N, const int* col_ptr, float* val) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        int start = col_ptr[col];
        int end = col_ptr[col + 1];
        float sum = 0.0f;
        for (int i = start; i < end; ++i) {
            sum += val[i];
        }
        if (sum > 0.0f) {
            for (int i = start; i < end; ++i) {
                val[i] /= sum;
            }
        }
    }
}
```

Fungsi berlabel `__global__` ini adalah kode yang dieksekusi murni di dalam inti (Cores) GPU NVIDIA. Tujuannya adalah mengubah matriks koneksi biasa menjadi **Matriks Markov (Stokastik)**, di mana total peluang (bobot) dari setiap produk (kolom) harus berjumlah tepat `1.0`.

**Alur Logika Pekerja GPU (Thread):**
1. **Pemetaan Pekerja (`int col = ...`):** Setiap *Thread* GPU diberi ID unik. ID ini digunakan secara langsung untuk mewakili satu Kolom Produk. (Contoh: Thread 0 mengurus Kolom 0, Thread 1 mengurus Kolom 1).
2. **Pengecekan Batas (`if (col < N)`):** Mencegah *Thread* fiktif (kelebihan pekerja dari pembulatan `blocks1D`) memproses memori yang tidak ada.
3. **Membaca Peta Sparse (`start` & `end`):** Karena nilai berderet memanjang di array `val`, GPU harus melihat array `col_ptr` untuk mengetahui di indeks ke berapa data kolomnya dimulai dan diakhiri.
4. **Penjumlahan (Pass 1):** GPU menjumlahkan seluruh bobot di kolom tersebut (`sum += val[i]`).
5. **Pembagian (Pass 2):** Jika kolom tersebut memiliki koneksi (`sum > 0.0`), setiap bobot dibagi dengan total `sum` agar menjadi probabilitas pecahan yang jika ditotal hasilnya `1.0`.

---

**Simulasi Visualisasi Eksekusi Paralel GPU (Normalisasi)**

Mari kita gunakan hasil format CSC dari Tahap 6 sebelumnya: 
* `col_ptr` = `{0, 2, 3, 3}` (Penunjuk batas)
* `val` = `{0.5, 0.9, 0.4}` (Bobot asli)
* Jumlah Produk `N = 3`.

Di dalam GPU, **Thread 0, Thread 1, dan Thread 2 akan bekerja secara serentak (bersamaan) di detik yang sama.** Berikut adalah simulasi apa yang terjadi di dalam otak tiap-tiap pekerja GPU:

| Tindakan | Thread 0 (Menangani Kolom 0) | Thread 1 (Menangani Kolom 1) | Thread 2 (Menangani Kolom 2) |
| :--- | :--- | :--- | :--- |
| **1. Baca Batas Awal (`start`)** | `col_ptr[0]` ➔ Indeks **`0`** | `col_ptr[1]` ➔ Indeks **`2`** | `col_ptr[2]` ➔ Indeks **`3`** |
| **2. Baca Batas Akhir (`end`)** | `col_ptr[1]` ➔ Indeks **`2`** | `col_ptr[2]` ➔ Indeks **`3`** | `col_ptr[3]` ➔ Indeks **`3`** |
| **3. Cari Nilai di Array `val`** | Membaca indeks ke-`0` dan `1` <br> *(Isi: `0.5` dan `0.9`)* | Membaca indeks ke-`2` <br> *(Isi: `0.4`)* | Membaca indeks ke-`3` sampai `3` <br> *(Kosong/Tidak ada data)* |
| **4. Hitung Total (`sum`)** | `0.5 + 0.9` = **`1.4`** | `0.4` = **`0.4`** | Tidak ada proses = **`0.0`** |
| **5. Operasi Pembagian (`val[i] /= sum`)** | Indeks 0: `0.5 / 1.4` = **`0.357`** <br> Indeks 1: `0.9 / 1.4` = **`0.643`** | Indeks 2: `0.4 / 0.4` = **`1.0`** | Syarat `sum > 0.0` gagal. <br> Diabaikan. |

**Hasil Akhir di Memori GPU (`d_val`):**
Setelah ketiga *Thread* selesai bekerja secara paralel, array nilai di memori GPU otomatis diperbarui menjadi:
* `val` baru = **`{0.357, 0.643, 1.0}`**

*(Kini setiap kolom secara matematis sudah valid menjadi probabilitas stokastik).*

```cpp
    vector<float> h_chaos(N, 0.0f);
    vector<int> h_nnz_per_col(N, 0);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
```

**6. Wadah Laporan & Stopwatch Internal GPU**
* Vektor `h_chaos` dan `h_nnz_per_col` disiapkan di RAM sebagai keranjang kosong penerima laporan dari GPU di setiap akhir iterasi.
* `cudaEvent_t` digunakan karena fungsi waktu bawaan CPU (seperti `time.h`) tidak akurat untuk mengukur kecepatan GPU. Perintah `cudaEventRecord(start)` ditekan tepat 1 milidetik sebelum Tahap 3 (Iterasi MCL) dimulai, untuk mengukur kecepatan konvergensi secara absolut.

### TAHAP 3: Iterasi MCL Sparse (Ekspansi dan Perkalian cuSPARSE)

Di dalam algoritma Markov Clustering (MCL), tahap **Ekspansi** adalah proses mengalikan matriks dengan dirinya sendiri (Matriks C = Matriks A $\times$ Matriks A). Tujuannya adalah untuk menemukan jalur koneksi baru antar-produk. Karena kita menggunakan matriks *Sparse* dan memori GPU, perkalian ini memerlukan prosedur khusus melalui pustaka NVIDIA cuSPARSE.

**1. Pendaftaran Identitas Matriks (Descriptor)**
```cpp
        // 1. EKSPANSI (cuSPARSE M * M)
        cusparseSpMatDescr_t matA, matC;
        CHECK_CUSPARSE(cusparseCreateCsr(&matA, N, N, nnz, d_col_ptr, d_row_idx, d_val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
```
GPU pada dasarnya "buta" dan hanya melihat array `d_col_ptr`, `d_row_idx`, dan `d_val` sebagai tumpukan angka acak di memori VRAM. Agar mesin cuSPARSE mengenali ketiga array tersebut sebagai satu kesatuan struktur data matriks, kita harus mendaftarkannya secara resmi menggunakan fitur *Descriptor* (bisa diibaratkan sebagai pembuatan "KTP" untuk matriks).

Fungsi `cusparseCreateCsr` bertugas mengikat data mentah tersebut dengan aturan pembacaan yang ketat. Berikut adalah rincian formulir identitas yang diserahkan ke mesin cuSPARSE:

| Parameter API | Variabel Input | Penjelasan Logika |
| :--- | :--- | :--- |
| **Target Descriptor** | `&matA` | Cetakan *descriptor* kosong yang akan diisi identitas matriks awal. |
| **Dimensi Matriks** | `N, N` | Ukuran baris dan kolom. Karena ini graf antar-produk, bentuknya mutlak persegi ($N \times N$). |
| **Jumlah Data (NNZ)** | `nnz` | Total jumlah koneksi yang saat ini hidup di dalam matriks. |
| **Alamat Memori** | `d_col_ptr`, `d_row_idx`, `d_val` | Tiga lokasi array fisik yang sebelumnya sudah ditransfer ke memori GPU di Tahap 2. |
| **Tipe Data Indeks 1** | `CUSPARSE_INDEX_32I` | Memberitahu GPU bahwa array batas kolom (`col_ptr`) bertipe *Integer* 32-bit. |
| **Tipe Data Indeks 2** | `CUSPARSE_INDEX_32I` | Memberitahu GPU bahwa array indeks baris (`row_idx`) bertipe *Integer* 32-bit. |
| **Titik Awal Indeks** | `CUSPARSE_INDEX_BASE_ZERO` | Mengunci aturan sistem bahwa perhitungan indeks array selalu dimulai dari angka `0` (standar C++). |
| **Tipe Data Nilai** | `CUDA_R_32F` | Memberitahu GPU bahwa array bobot (`val`) berbentuk angka desimal biasa (*Real Float* 32-bit). |

> **Catatan Penting: Trik Ilusi Memori (CSR vs CSC)**
> Anda mungkin menyadari sebuah kejanggalan: Data graf kita berformat **CSC** (berbasis kolom), tetapi kita mendaftarkannya menggunakan fungsi `cusparseCreateCsr` (berbasis baris). Mengapa demikian?
> Ini adalah trik rekayasa standar untuk mengakali mesin SpGEMM cuSPARSE yang murni dirancang hanya untuk operasi baris (CSR). 
> 
> Di dalam arsitektur memori, terdapat hukum mutlak: **Susunan fisik array matriks CSC adalah kembaran identik dari susunan fisik array matriks Transpose dalam format CSR.**
> Saat kita menyuapkan array CSC kita ke dalam fungsi CSR ini, mesin GPU meyakini bahwa ia sedang menerima Matriks $A$ yang berstatus di-transpose ($A^T$). Ia kemudian mengalikan $A^T \times A^T$ menggunakan operasi baris, yang secara matematis menghasilkan luaran $(A \times A)^T$ berformat CSR. Hebatnya, hasil luaran $(A \times A)^T$ berformat CSR tersebut otomatis merupakan **kembaran identik** dari Matriks $A \times A$ dalam format CSC yang sesungguhnya kita butuhkan. Melalui tipuan cermin aljabar ini, kita berhasil melakukan perhitungan kolom memanggil kolom menggunakan mesin berbasis baris milik NVIDIA.

**2. Persiapan Wadah Hasil (Matriks C) dan Surat Perintah Kerja**
```cpp
        // Setup descriptor matriks hasil C (belum tau jumlah elemennya)
        int* d_C_col_ptr;
        CHECK_CUDA(cudaMalloc(&d_C_col_ptr, (N + 1) * sizeof(int)));
        CHECK_CUSPARSE(cusparseCreateCsr(&matC, N, N, 0, d_C_col_ptr, nullptr, nullptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

        cusparseSpGEMMDescr_t spgemmDesc;
        CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));
```

Berbeda dengan matriks padat (*Dense*) yang hasil perkaliannya selalu $N \times N$, hasil perkalian matriks *Sparse* akan menciptakan koneksi-koneksi baru yang jumlah total akhirnya tidak bisa ditebak oleh CPU pada saat ini. Oleh karena itu, kita membuat identitas matriks hasil (`matC`) secara bertahap.

**Logika Pembuatan Matriks C (Identitas Sementara):**
Kita mengibaratkan Matriks C sebagai proyek pembangunan gedung yang belum selesai.

| Parameter Spesifik `matC` | Nilai yang Diinput | Alasan Logis |
| :--- | :--- | :--- |
| **Peta Batas Kolom (`col_ptr`)** | `d_C_col_ptr` | Kita sudah bisa memesan memori ini (`cudaMalloc`) karena ukuran dimensi batas kolom akan selalu pasti, yaitu $N + 1$. |
| **Jumlah Koneksi (`nnz`)** | `0` | Diset nol untuk sementara, karena komputer belum tahu berapa total koneksi baru yang akan terlahir dari hasil perkalian Ekspansi. |
| **Daftar Baris & Bobot** | `nullptr` (Kosong) | Dibiarkan hampa tanpa alokasi memori, karena ukurannya mutlak bergantung pada hasil perhitungan `nnz` di tahap selanjutnya. |

**Inisialisasi `spgemmDesc` (Surat Perintah Kerja):**
Operasi *Sparse General Matrix-Matrix Multiplication* (SpGEMM) adalah algoritma komputasi yang sangat kompleks dan terdiri dari banyak langkah. GPU memerlukan wadah khusus untuk mencatat rencana estimasi memori, perhitungan kerangka struktur, hingga eksekusi akhirnya. Objek `spgemmDesc` bertindak sebagai "Buku Catatan Mandor Proyek" yang meresmikan dimulainya tahap perkalian tersebut.

**3. Estimasi Memori dan Komputasi Struktur (Metode "Tanya-Bayar-Kerja")**

```cpp
        float alpha = 1.0f, beta = 0.0f;
        size_t bufferSize1 = 0, bufferSize2 = 0;
        void* dBuffer1 = nullptr; void* dBuffer2 = nullptr;

        // Tanya GPU butuh memori berapa (Estimasi tahap 1)
        CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matA, &beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, nullptr));
        CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1));
        CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matA, &beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, dBuffer1));

        // Tanya GPU butuh memori berapa untuk perhitungan (Estimasi tahap 2)
        CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matA, &beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, nullptr));
        CHECK_CUDA(cudaMalloc(&dBuffer2, bufferSize2));
        CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matA, &beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, dBuffer2));
```

Karena jumlah koneksi (NNZ) hasil dari Ekspansi ($Matriks A \times Matriks A$) tidak bisa diprediksi di awal dan bisa membengkak drastis (memunculkan banyak koneksi "Teman dari Teman"), GPU memerlukan kertas coretan atau memori sementara (*buffer*) yang sangat besar. 

Pustaka cuSPARSE menggunakan pola kodingan unik yang bisa diibaratkan seperti menyewa mandor proyek: **Tanya (Ukur) ➔ Bayar (Sewa Memori) ➔ Kerja (Eksekusi)**. Pola ini dilakukan dua kali:
1. **Fase `workEstimation`:** GPU menyurvei bentuk matriks secara kasar.
2. **Fase `compute`:** GPU mulai melakukan simulasi perkalian logis untuk merangkai struktur pasti dari Matriks C (belum menghitung nilai desimalnya).

**Bedah Parameter API cuSPARSE (Dari Kiri ke Kanan):**
Rumus baku komputasi ini adalah $C = \alpha (A \times A) + \beta C$. Kedua fungsi di atas (`workEstimation` dan `compute`) menggunakan isi parameter yang sama persis:

| Parameter di Dalam Kurung | Variabel / Argumen | Logika & Penjelasan |
| :--- | :--- | :--- |
| **Sesi GPU** | `handle` | Menyerahkan tongkat komando ke mesin cuSPARSE. |
| **Operasi Matriks Kiri** | `CUSPARSE_OPERATION_NON_TRANSPOSE` | Matriks pertama tidak dibalik (baris dan kolom dibiarkan normal). |
| **Operasi Matriks Kanan** | `CUSPARSE_OPERATION_NON_TRANSPOSE` | Matriks kedua juga tidak dibalik. |
| **Skalar Pengali Awal** | `&alpha` | Nilai `1.0f`, agar hasil perkalian Matriks A nilainya utuh (dikali 1). |
| **Identitas Matriks Kiri** | `matA` | Menggunakan "KTP" Matriks A yang sudah disiapkan sebelumnya. |
| **Identitas Matriks Kanan** | `matA` | Menggunakan matriks yang sama, karena kita melakukan Ekspansi diri sendiri. |
| **Skalar Pengali Tambahan** | `&beta` | Nilai `0.0f`, agar isi Matriks C yang masih hampa dianggap nol murni. |
| **Identitas Matriks Hasil** | `matC` | Target penampungan, tempat GPU nanti akan menyusun matriks hasil. |
| **Tipe Data Operasi** | `CUDA_R_32F` | Perhitungan dilakukan di mode bilangan pecahan desimal biasa (32-bit *Float*). |
| **Algoritma Internal** | `CUSPARSE_SPGEMM_DEFAULT` | Menyuruh GPU membagi beban kerja (*Thread Block*) secara otomatis dengan cara paling optimal. |
| **Surat Perintah Kerja** | `spgemmDesc` | Buku catatan tempat GPU menulis progres pengerjaannya. |
| **Wadah Ukuran Memori** | `&bufferSize1` / `&bufferSize2` | Tempat GPU menuliskan angka (dalam *byte*) mengenai seberapa besar memori yang ia butuhkan. |
| **Wadah Memori Coretan** | `nullptr` **ATAU** `dBuffer1`/`dBuffer2` | *Ini kuncinya.* Pemanggilan pertama diisi `nullptr` (hanya meminta angka estimasi ukuran). Setelah memori dialokasikan (`cudaMalloc`), pemanggilan kedua diisi alamat memori resminya (`dBuffer...`) agar GPU bisa mulai bekerja. |

Setelah baris kodingan ini terlewati, struktur "bangunan" Matriks C yang memuat letak-letak koneksi baru sudah terbentuk dengan jelas di dalam mesin GPU, siap untuk disalin ke memori permanen.

---

**Simulasi Logika Perkalian Matriks Sparse di dalam `compute`**
Berkat trik ilusi memori yang kita lakukan di awal, eksekusi fisik perkalian di dalam memori tetap berjalan mulus menggunakan algoritma berbasis "Kolom memanggil Kolom". 
Mari kita gunakan data 3 Produk ($N = 3$) yang nilainya **sudah dinormalisasi** dari tahap sebelumnya: 
* `col_ptr` = `[0, 2, 3, 3]`
* `row_idx` = `[1, 2, 0]`
* `val` = `[0.357, 0.643, 1.0]`

Berikut adalah proses internal fisik yang terjadi pada data array saat mencari koneksi baru untuk mengisi Matriks C:

| Target Eksekusi | Apa yang Dicari pada Data Array Memori | Eksekusi Perhitungan & Hasil |
| :--- | :--- | :--- |
| **Mencari Isi Kolom 0** | Cek Matriks Kanan di Kolom 0. Ada data di Baris **1** (0.357) dan Baris **2** (0.643). | • Panggil Kolom **1** Kiri: Ketemu data Baris 0 (1.0). <br>➔ Dikali: `0.357 * 1.0` = **`0.357`** <br>• Panggil Kolom **2** Kiri: Kosong. <br>➔ **Hasil Kolom 0:** Tercipta 1 koneksi di Baris 0 (0.357). |
| **Mencari Isi Kolom 1** | Cek Matriks Kanan di Kolom 1. Ada data di Baris **0** (1.0). | • Panggil Kolom **0** Kiri: Ketemu data Baris 1 (0.357) dan Baris 2 (0.643). <br>➔ Dikali: `1.0 * 0.357` = **`0.357`** dan `1.0 * 0.643` = **`0.643`** <br>➔ **Hasil Kolom 1:** Tercipta 2 koneksi di Baris 1 (0.357) dan Baris 2 (0.643). |
| **Mencari Isi Kolom 2** | Cek Matriks Kanan di Kolom 2. Indeks awal dan akhir sama (batas 3 sampai 3). | • Karena di Matriks Kanan kosong, GPU otomatis melompati proses ini. <br>➔ **Hasil Kolom 2:** Tetap kosong (0 koneksi). |

**Hasil Akhir Matriks C (Disimpan dalam Format CSC)**
Dari simulasi perkalian tersebut, Matriks C kini memegang total `C_nnz = 3` koneksi baru. Program secara otomatis menyusun dan menyimpan wujud fisik Matriks C ke dalam 3 buah array CSC yang berjejer rapi di memori GPU:
* `C_col_ptr` = **`[0, 1, 3, 3]`** *(Dari batas 0 ke 1 ada 1 data. Batas 1 ke 3 ada 2 data. Batas 3 ke 3 kosong)*
* `C_row_idx` = **`[0, 1, 2]`** *(Baris 0 milik Kolom 0. Baris 1 dan 2 milik Kolom 1)*
* `C_val` = **`[0.357, 0.357, 0.643]`** *(Hasil angka perkalian sesuai urutan baris)*

---

**Manajemen Pasukan Pekerja GPU (Dynamic Load Balancing)**
Pada tahap ini tidak ada kodingan pemanggilan regu pekerja secara manual seperti `<<<blocks, threads>>>`. Fungsi cuSPARSE ini adalah sebuah *Black Box* (Kotak Hitam), di mana NVIDIA menyembunyikan logika penjadwalan *thread*-nya dari *programmer*.

Alasannya adalah sifat matriks *Sparse* yang tidak seimbang (*imbalanced*). Jika NVIDIA memaksakan aturan baku "1 Thread mengerjakan 1 Kolom", GPU akan mengalami **Thread Divergence**. *Thread* yang menangani kolom kosong akan menganggur total, sementara *Thread* yang menangani kolom padat akan bekerja sendirian terlalu lama.

Sebagai gantinya, pustaka cuSPARSE menggunakan **Penyeimbangan Beban Dinamis**:
* **Ringan (1-5 koneksi):** Diserahkan kepada **1 Thread** GPU tunggal.
* **Sedang (Puluhan koneksi):** Dikerjakan bergotong-royong oleh **1 Warp** (Grup berisi 32 *Thread*).
* **Berat (Ribuan koneksi):** Diserbu secara masif oleh **1 Block penuh** (256 atau 512 *Thread* sekaligus).

**4. Ekstraksi Ukuran, Alokasi Memori Permanen, dan Salin Data (Tahap Akhir SpGEMM)**

```cpp
        // Ambil info jumlah koneksi baru hasil perkalian
        int64_t C_rows_64, C_cols_64, C_nnz_64;
        CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &C_rows_64, &C_cols_64, &C_nnz_64));
        int C_nnz = (int)C_nnz_64;

        int *d_C_row_idx; float *d_C_val;
        CHECK_CUDA(cudaMalloc(&d_C_row_idx, C_nnz * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_C_val, C_nnz * sizeof(float)));
        CHECK_CUSPARSE(cusparseCsrSetPointers(matC, d_C_col_ptr, d_C_row_idx, d_C_val));

        // Eksekusi Copy Perkalian Sebenarnya
        CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matA, &beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));
```

Setelah tahap `compute` selesai, GPU secara internal sudah mengetahui wujud pasti dan jumlah koneksi dari Matriks C. Namun, informasi tersebut masih tertahan di dalam GPU. Kodingan di atas berfungsi untuk menarik laporan angka dari GPU, menyewa memori permanen yang ukurannya akurat, dan meresmikan pemindahan data.

**Bedah Logika dan Parameter API Tahap Akhir:**

**A. Bertanya Hasil Survei kepada GPU (`cusparseSpMatGetSize`)**
Fungsi ini bertugas menginterogasi GPU mengenai dimensi matriks hasil.
| Parameter | Logika & Penjelasan |
| :--- | :--- |
| `matC` | "KTP" matriks target yang mau kita interogasi informasinya. |
| `&C_rows_64` | Tempat GPU menuliskan angka tunggal jumlah baris (pasti senilai $N$). |
| `&C_cols_64` | Tempat GPU menuliskan angka tunggal jumlah kolom (pasti senilai $N$). |
| `&C_nnz_64` | **Kunci Utama:** Tempat GPU menuliskan angka pasti total koneksi (NNZ) hasil Ekspansi. |

*(Menggunakan simulasi 3 produk dari tahap sebelumnya, GPU akan merespons dengan mengisi variabel `C_nnz_64` dengan angka bulat **`3`**. Angka ini kemudian dikonversi menjadi integer standar 32-bit di variabel `C_nnz`)*.

**B. Alokasi Memori Permanen (`cudaMalloc`)**
Karena CPU sekarang sudah memegang angka pasti jumlah koneksi yang terbentuk (`C_nnz = 3`), CPU dengan yakin menyewa lahan memori yang presisi (tidak kurang dan tidak mubazir):
* `d_C_row_idx` disewa sebesar 3 kotak bertipe *Integer* (untuk menyimpan `[0, 1, 2]`).
* `d_C_val` disewa sebesar 3 kotak bertipe *Float* (untuk menyimpan `[0.357, 0.357, 0.643]`).

**C. Memperbarui Identitas Matriks C (`cusparseCsrSetPointers`)**
Di awal proyek, array baris dan nilai pada Matriks C kita isi dengan `nullptr` (kosong). Kini kita harus memperbaruinya.
| Parameter | Logika & Penjelasan |
| :--- | :--- |
| `matC` | "KTP" Matriks C yang akan di-*update*. |
| `d_C_col_ptr` | Memori batas kolom (ukuran $N+1$) yang sudah dialokasikan sejak awal. |
| `d_C_row_idx` | Memori indeks baris baru yang ukurannya sudah menyesuaikan nilai `C_nnz`. |
| `d_C_val` | Memori nilai bobot probabilitas baru yang juga sudah menyesuaikan nilai `C_nnz`. |

**D. Pemindahan Data Final (`cusparseSpGEMM_copy`)**
Ini adalah gong penutup operasi *Sparse*. Seluruh isi parameter di dalam kurung fungsi ini **100% sama persis** dengan fungsi `workEstimation` dan `compute` sebelumnya.
| Mengapa Parameternya Harus Sama Persis? |
| :--- |
| Pemanggilan ulang dengan `handle`, matriks identitas yang sama, dan "Surat Perintah Kerja" (`spgemmDesc`) yang identik adalah cara GPU NVIDIA mengenali bahwa ini adalah **kelanjutan dari proyek komputasi yang sama**. GPU akan langsung mencari "kertas coretan" dari tahap komputasi sebelumnya, mengambil angka-angka CSC hasil Ekspansi yang sudah matang, lalu menyalinnya secara fisik ke dalam alamat memori permanen `matC` yang baru saja kita resmikan. |

Melalui keseluruhan tarian operasi SpGEMM ini, algoritma MCL berhasil meledakkan koneksi "Teman dari Teman" dan menampungnya dengan aman ke dalam struktur memori GPU (VRAM) tanpa takut terjadi *Crash* atau kekurangan ruang.

**4.1. Mengapa Alurnya Terbalik (Hitung Dulu, Baru Alokasi Memori, Lalu Salin)?**

Bagi *programmer* yang terbiasa dengan bahasa tingkat tinggi, alur eksekusi SpGEMM ini sering kali terasa membingungkan. Mengapa kita melakukan operasi perkalian (`compute`) *sebelum* menciptakan memori penampungnya (`cudaMalloc`), lalu mengapa harus disalin lagi (`copy`)? 

Untuk memahami keharusan teknis ini, mari kita gunakan **Analogi Ujian Matematika**:

Bayangkan mesin GPU adalah Anda yang sedang mengerjakan ujian aljabar perkalian matriks raksasa, dan `matC` adalah "Stopmap Jawaban Resmi" Anda. Aturan ujiannya sangat ketat: *Kertas jawaban yang dimasukkan ke dalam stopmap jumlah halamannya harus sama persis dengan jumlah baris jawaban akhir, tidak boleh kurang dan tidak boleh sisa (mubazir).*

Karena Anda mengalikan matriks *Sparse*, Anda tidak akan tahu berapa jumlah persis koneksi ("Teman dari Teman") yang tercipta sebelum selesai menghitung semuanya. Berikut adalah apa yang sebenarnya terjadi di dalam mesin GPU pada setiap baris kodingan tersebut:

| Tahap Eksekusi | Analogi Ujian Matematika | Proses Fisik di Dalam GPU |
| :--- | :--- | :--- |
| **1. `compute`** | **Menghitung di Kertas Buram.** Anda menghitung semua soal dan mencorat-coret hasilnya secara berantakan di atas Kertas Buram (`dBuffer2`). | GPU melakukan perkalian masif. Seluruh hasil angka desimal sementara berceceran di dalam *buffer* internal mesin NVIDIA yang formatnya berantakan dan tidak bisa kita akses langsung. Saat ini, `matC` masih berupa stopmap kosong (`nullptr`). |
| **2. `GetSize`** | **Menghitung Total Baris Jawaban.** Setelah selesai menghitung di kertas buram, Anda menghitung: *"Oh, hasil ujian saya ternyata memakan tepat 1.500 baris."* | CPU menginterogasi GPU untuk mengekstrak angka `C_nnz` (Total Koneksi Baru) dari hasil perhitungan di *buffer* tadi. |
| **3. `cudaMalloc`** | **Membeli Kertas Jawaban Resmi.** Karena sudah tahu butuh 1.500 baris, Anda baru berani memesan Kertas Jawaban Resmi bergaris sebanyak jumlah tersebut. | CPU akhirnya menyewa lahan memori VRAM yang ukurannya sangat presisi (`d_C_row_idx` dan `d_C_val`), sehingga tidak ada satu *byte* pun memori yang terbuang sia-sia. |
| **4. `SetPointers`** | **Memasukkan Kertas ke Stopmap.** Anda menjepret kertas jawaban kosong yang baru dibeli ke dalam Stopmap Jawaban Resmi (`matC`). | CPU mengikatkan alamat memori array CSC yang baru dibuat ke dalam *descriptor* identitas `matC`. Sekarang `matC` sudah resmi memiliki wujud fisik penampung. |
| **5. `copy`** | **Menyalin (Memindahkan) Jawaban.** Anda memindahkan tulisan dari Kertas Buram yang berantakan ke atas Kertas Jawaban Resmi di dalam stopmap dengan sangat rapi. |  <br> GPU mengambil data mentah dari *buffer* internal (`dBuffer2`), menyusunnya menjadi format array CSC yang terstruktur rapi, lalu menuangkan/menyalin angka-angka tersebut secara permanen ke dalam vektor-vektor milik `matC`. *Buffer* sementara kemudian bisa dihapus untuk menghemat VRAM. |

Melalui pola kerja ini, kita berhasil melakukan operasi dinamis yang sangat memakan memori di dalam lingkungan bahasa C++ yang menuntut manajemen memori statis dan presisi tinggi. Kertas buram (`dBuffer`) berfungsi sebagai pelindung agar program tidak *crash* akibat *Out-of-Memory*, dan operasi `copy` memastikan matriks hasil kita siap digunakan untuk tahap algoritma MCL selanjutnya (Inflasi & Pruning).
