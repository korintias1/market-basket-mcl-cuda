#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <locale>
#include <cuda_runtime.h>
#include <cusparse.h> // LIBRARY RESMI PERKALIAN SPARSE NVIDIA

using namespace std;

// ==========================================================
// MACRO DETEKSI ERROR CUDA & CUSPARSE
// ==========================================================
#define CHECK_CUDA(func) \
{ \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        cerr << "\n[CUDA ERROR] " << cudaGetErrorString(status) << " di baris " << __LINE__ << endl; \
        exit(1); \
    } \
}

#define CHECK_CUSPARSE(func) \
{ \
    cusparseStatus_t status = (func); \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        cerr << "\n[CUSPARSE ERROR] Kode: " << status << " di baris " << __LINE__ << endl; \
        exit(1); \
    } \
}

float parse_weight(string s) {
    replace(s.begin(), s.end(), ',', '.');
    float val = 0.0f;
    stringstream ss(s);
    ss.imbue(locale::classic());
    ss >> val;
    return val;
}

// ==========================================================
// KERNEL GPU: KUSTOM SPARSE
// ==========================================================
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

// 2. Paket 3-in-1: Inflasi -> Pruning -> Normalisasi -> Hitung Chaos
__global__ void inflate_prune_normalize_chaos_kernel(int N, const int* col_ptr, float* val, 
                                                     float power, float threshold, 
                                                     int* nnz_per_col, float* chaos_arr) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        int start = col_ptr[col];
        int end = col_ptr[col + 1];
        float sum = 0.0f;
        int valid_nnz = 0;

        // Pass 1: Inflasi & Prune
        for (int i = start; i < end; ++i) {
            float v = val[i];
            if (v > 0.0f) {
                v = powf(v, power);
                if (v < threshold) v = 0.0f; // Pruning
                val[i] = v;
                sum += v;
                if (v > 0.0f) valid_nnz++;
            }
        }
        
        nnz_per_col[col] = valid_nnz; // Simpan untuk Compaction nanti

        // Pass 2: Normalisasi Ulang & Hitung Local Chaos
        float max_val = 0.0f;
        float sum_sq = 0.0f;
        if (sum > 0.0f) {
            for (int i = start; i < end; ++i) {
                if (val[i] > 0.0f) {
                    float norm_v = val[i] / sum;
                    val[i] = norm_v;
                    if (norm_v > max_val) max_val = norm_v;
                    sum_sq += norm_v * norm_v;
                }
            }
        }
        chaos_arr[col] = max_val - sum_sq;
    }
}

// 3. Compaction: Pemindahan data yang selamat dari Pruning ke Memori Baru
__global__ void compact_sparse_kernel(int N, const int* old_col_ptr, const int* old_row_idx, const float* old_val,
                                      const int* new_col_ptr, int* new_row_idx, float* new_val) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        int old_start = old_col_ptr[col];
        int old_end = old_col_ptr[col + 1];
        int new_idx = new_col_ptr[col];

        for (int i = old_start; i < old_end; ++i) {
            if (old_val[i] > 0.0f) {
                new_val[new_idx] = old_val[i];
                new_row_idx[new_idx] = old_row_idx[i];
                new_idx++;
            }
        }
    }
}

// ==========================================================
// KODE HOST (CPU)
// ==========================================================
int main() {
    cout << "=== PROGRAM MCL CUDA (SPARSE MATRIX & CUSPARSE) ===" << endl;

    float inflation_p = 3.0f; 
    float prune_threshold = 1e-3f; 
    float convergence_threshold = 1e-5f; 
    int max_iterations = 1000; 

    // --- TAHAP 1: BACA DATA & BENTUK CSC/CSR ---
    string file_name = "edgelist_Aw_Cosine_10000.csv";
    ifstream file(file_name);
    if (!file.is_open()) { cerr << "Error: File tidak ditemukan!" << endl; return 1; }

    string line, str_A, str_B, str_W;
    getline(file, line);

    unordered_map<int, int> map_index;
    vector<int> reverse_map;
    int N = 0;

    struct Edge { int r, c; float w; };
    vector<Edge> temp_edges;

    while (getline(file, line)) {
        stringstream ss(line);
        getline(ss, str_A, ','); getline(ss, str_B, ','); getline(ss, str_W, ',');
        if(str_A.empty()) continue;

        int id_a = stoi(str_A);
        int id_b = stoi(str_B);
        float w = parse_weight(str_W);

        if (map_index.find(id_a) == map_index.end()) { map_index[id_a] = N++; reverse_map.push_back(id_a); }
        if (map_index.find(id_b) == map_index.end()) { map_index[id_b] = N++; reverse_map.push_back(id_b); }

        temp_edges.push_back({map_index[id_a], map_index[id_b], w});
    }
    file.close();

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

    // --- TAHAP 2: ALOKASI MEMORI GPU ---
    int *d_col_ptr, *d_row_idx, *d_nnz_per_col;
    float *d_val, *d_chaos;

    CHECK_CUDA(cudaMalloc(&d_col_ptr, (N + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_row_idx, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_val, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_nnz_per_col, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_chaos, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_col_ptr, h_col_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_row_idx, h_row_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_val, h_val.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));

    // Setup cuSPARSE
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    int threads1D = 256;
    int blocks1D = (N + threads1D - 1) / threads1D;

    // Normalisasi Awal
    initial_normalize_kernel<<<blocks1D, threads1D>>>(N, d_col_ptr, d_val);
    CHECK_CUDA(cudaDeviceSynchronize());

    vector<float> h_chaos(N, 0.0f);
    vector<int> h_nnz_per_col(N, 0);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    // --- TAHAP 3: ITERASI MCL SPARSE ---
    for (int iter = 0; iter < max_iterations; ++iter) {
        
        // 1. EKSPANSI (cuSPARSE M * M)
        cusparseSpMatDescr_t matA, matC;
        CHECK_CUSPARSE(cusparseCreateCsr(&matA, N, N, nnz, d_col_ptr, d_row_idx, d_val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        
        // Setup descriptor matriks hasil C (belum tau jumlah elemennya)
        int* d_C_col_ptr;
        CHECK_CUDA(cudaMalloc(&d_C_col_ptr, (N + 1) * sizeof(int)));
        CHECK_CUSPARSE(cusparseCreateCsr(&matC, N, N, 0, d_C_col_ptr, nullptr, nullptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

        cusparseSpGEMMDescr_t spgemmDesc;
        CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));

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

        // 2. INFLASI, PRUNE, NORMALISASI, CHAOS
        inflate_prune_normalize_chaos_kernel<<<blocks1D, threads1D>>>(N, d_C_col_ptr, d_C_val, inflation_p, prune_threshold, d_nnz_per_col, d_chaos);
        CHECK_CUDA(cudaDeviceSynchronize());

        // 3. COMPACTION (Pemadatan Matriks agar Hemat Memori)
        CHECK_CUDA(cudaMemcpy(h_nnz_per_col.data(), d_nnz_per_col, N * sizeof(int), cudaMemcpyDeviceToHost));
        
        vector<int> h_new_col_ptr(N + 1, 0);
        for (int i = 0; i < N; ++i) h_new_col_ptr[i + 1] = h_new_col_ptr[i] + h_nnz_per_col[i];
        int new_nnz = h_new_col_ptr[N];

        int *d_new_col_ptr, *d_new_row_idx; float *d_new_val;
        CHECK_CUDA(cudaMalloc(&d_new_col_ptr, (N + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_new_row_idx, new_nnz * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_new_val, new_nnz * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_new_col_ptr, h_new_col_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice));

        // Pindahkan sisa elemen ke matriks ringkas
        compact_sparse_kernel<<<blocks1D, threads1D>>>(N, d_C_col_ptr, d_C_row_idx, d_C_val, d_new_col_ptr, d_new_row_idx, d_new_val);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Cek Global Chaos
        CHECK_CUDA(cudaMemcpy(h_chaos.data(), d_chaos, N * sizeof(float), cudaMemcpyDeviceToHost));
        float global_chaos = 0.0f;
        for (int i = 0; i < N; i++) if (h_chaos[i] > global_chaos) global_chaos = h_chaos[i];

        cout << "Iterasi " << iter + 1 << " | NNZ Aktif: " << new_nnz << " | Global Chaos: " << global_chaos << endl;

        // Bersihkan memori lama, Tukar pointer
        cudaFree(d_col_ptr); cudaFree(d_row_idx); cudaFree(d_val);
        cudaFree(d_C_col_ptr); cudaFree(d_C_row_idx); cudaFree(d_C_val);
        cudaFree(dBuffer1); cudaFree(dBuffer2);
        cusparseDestroySpMat(matA); cusparseDestroySpMat(matC); cusparseSpGEMM_destroyDescr(spgemmDesc);

        d_col_ptr = d_new_col_ptr; d_row_idx = d_new_row_idx; d_val = d_new_val; nnz = new_nnz;

        if (global_chaos < convergence_threshold) {
            cout << "\n>> KONVERGENSI TERCAPAI pada Iterasi ke-" << iter + 1 << "!" << endl;
            break;
        }
    }

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float milliseconds = 0; cudaEventElapsedTime(&milliseconds, start, stop);

    // --- TAHAP 4: EKSPOR HASIL ---
    cout << "\n=================================================" << endl;
    cout << "Komputasi MCL SPARSE Selesai." << endl;
    cout << "WAKTU EKSEKUSI GPU : " << milliseconds / 1000.0 << " detik." << endl;
    cout << "Koneksi Tersisa Akhir: " << nnz << " (Format Sparse)" << endl;
    cout << "=================================================\n" << endl;

    vector<int> final_col_ptr(N + 1);
    vector<int> final_row_idx(nnz);
    vector<float> final_val(nnz);

    CHECK_CUDA(cudaMemcpy(final_col_ptr.data(), d_col_ptr, (N + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(final_row_idx.data(), d_row_idx, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(final_val.data(), d_val, nnz * sizeof(float), cudaMemcpyDeviceToHost));

    ofstream file_matrix("mcl_matrix_final_sparse.csv");
    file_matrix << "Product_A,Product_B,Weight\n";
    ofstream file_attr("mcl_attributes_final_sparse.csv");
    file_attr << "Product_ID,Cluster_ID,Status_Titik\n";

    for (int col = 0; col < N; ++col) {
        float max_val = -1.0f;
        int attractor_idx = -1;

        int start = final_col_ptr[col];
        int end = final_col_ptr[col + 1];

        for (int i = start; i < end; ++i) {
            int row = final_row_idx[i];
            float v = final_val[i];
            
            file_matrix << reverse_map[row] << "," << reverse_map[col] << "," << v << "\n";
            if (v > max_val) {
                max_val = v;
                attractor_idx = row;
            }
        }

        if (attractor_idx != -1) {
            int current_product_id = reverse_map[col];
            int cluster_id = reverse_map[attractor_idx];
            string status = (col == attractor_idx) ? "Host" : "Anggota";
            file_attr << current_product_id << "," << cluster_id << "," << status << "\n";
        }
    }

    file_matrix.close(); file_attr.close();
    cout << "Ekspor berhasil! File CSV siap." << endl;

    cudaFree(d_col_ptr); cudaFree(d_row_idx); cudaFree(d_val); cudaFree(d_nnz_per_col); cudaFree(d_chaos);
    cusparseDestroy(handle); cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}