//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H

#include <iostream>
#include <vector>
#include <array>
#include <stdexcept>
#include <initializer_list>
#include <numeric>
#include <functional>
#include <string>

namespace utec::algebra {

template <typename T, int n>
class Tensor {
    std::array<int, n> dims{};
    std::array<int, n> cut;
    std::vector<T> data;

    int calculate_size() const {
        int total = 1;
        for (auto d : dims) total *= d;
        return total;
    }

    void make_cut() {
        cut[n - 1] = 1;
        for (int i = n - 2; i >= 0; i--){
            cut[i] = dims[i + 1] * cut[i + 1];
        }
    }
public:
    
    template <typename... Args>
    Tensor (Args... dimensions) {
        if (sizeof...(Args) != n) {
            throw std::logic_error("Number of dimensions do not match with " + std::to_string(n));
        }
        
        std::array<int, n> tmp{}; //pasarlo a vector para ahorrar problemas
        std::vector<int> values = { static_cast<int>(dimensions)... };

        for (int i = 0; i < n; ++i){
            tmp[i] = values[i];
        }

        dims = tmp;

        data.resize(calculate_size(), T{});

        // make search flat if necessary to avoid extra calc
        make_cut();
    }

    //constructor with other dims
    Tensor(const std::array<int, n>& shape) {
        dims = shape;
        data.resize(calculate_size(), T{});
        make_cut();
    }


    const std::array<int, n>& shape() const { return dims; }

    void fill(const T& value) {
        std::fill(data.begin(), data.end(), value);
    }

    Tensor& operator = (std::initializer_list<T> list) {
        if (list.size() != data.size()) {
            throw std::logic_error("Data size does not match tensor size");
        }
        std::copy(list.begin(), list.end(), data.begin());
        return *this;
    }

    // just verify size and change dims
    template <typename... Args>
    void reshape(Args... new_dims) {
        if (sizeof...(new_dims) != n) {
            throw std::logic_error("Number of dimensions do not match with " + std::to_string(n));
        }

        std::array<int, n> tmp{};
        std::vector<int> values = { static_cast<int>(new_dims)... };

        int new_size = 1;
        for (auto v : values) {
            new_size *= v;
        }

        if (new_size != static_cast<int>(data.size())) {
            throw std::logic_error("Reshape size does not match tensor size");
        }

        for (int i = 0; i < n; ++i){
            tmp[i] = values[i];
        }
        dims = tmp;
        make_cut();
    }

    // hacer funcion para () porque se una con y sin const
    template <typename... Idxs>
    int flat_index(Idxs... indexes) const {
        static_assert(sizeof...(Idxs) == n, "Number of indexes must match tensor dimensions");
        std::array<int, n> idx = { static_cast<int>(indexes)... };

        int index = 0;
        int multiplier = 1;
        for (int i = n - 1; i >= 0; --i) {
            index += idx[i] * multiplier;
            multiplier *= dims[i];
        }
        return index;
    }

    template <typename... Idxs>
    T& operator()(Idxs... indexes) {
        return data[flat_index(indexes...)];
    }

    template <typename... Idxs>
    const T& operator()(Idxs... indexes) const {
        return data[flat_index(indexes...)];
    }

    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
    auto cbegin() const { return data.cbegin(); }
    auto cend() const { return data.cend(); }
    
    //scalar
    template<typename F>
    Tensor scalar_op(const T& scalar, F func) const {
        Tensor<T, n> result(dims);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = func(data[i], scalar);
        }
        return result;
    }

    Tensor operator + (const T& scalar) const { return scalar_op(scalar, std::plus<T>()); }

    Tensor operator - (const T& scalar) const { return scalar_op(scalar, std::minus<T>()); }

    Tensor operator * (const T& scalar) const { return scalar_op(scalar, std::multiplies<T>()); }

    Tensor operator / (const T& scalar) const { return scalar_op(scalar, std::divides<T>()); }

    //tensor operators with broadcasting
    std::array<int, n> broadcast_dims(const Tensor& other) const {
        std::array<int, n> out{};
        for (int i = 0; i < n; ++i)
            out[i] = std::max(dims[i], other.dims[i]);
        return out;
    }

    int size() const {
        return std::accumulate(dims.begin(), dims.end(), int(1), std::multiplies<>());
    }

    template <typename Op>
    Tensor broadcast(const Tensor& other, Op op) const {
        for (size_t i = 0; i < n; ++i) {
            if (dims[i] != other.dims[i] && dims[i] != 1 && other.dims[i] != 1) {
                throw std::logic_error("Shapes do not match and they are not compatible for broadcasting");
            }
        }

        auto out_dims = broadcast_dims(other);
        Tensor<T, n> result(out_dims);// result shape

        int total = 1;
        for (int d : out_dims) total *= d; //result size

        std::array<int, n> idx;
        for (int i = 0; i < total; ++i) {
            int temp = i;
            for (int j = n - 1; j >= 0; --j) {
                idx[j] = temp % out_dims[j];
                temp /= out_dims[j];
            }

            // input in the indexes
            int flatA = 0, flatB = 0;
            for (int j = 0; j < n; ++j) {
                int idxA = (dims[j] == 1) ? 0 : idx[j];
                int idxB = (other.dims[j] == 1) ? 0 : idx[j];
                flatA += idxA * cut[j];
                flatB += idxB * other.cut[j];
            }
            result.data[i] = op(data[flatA], other.data[flatB]);
        }
        return result;
    }
    Tensor operator + (const Tensor& other) const { return broadcast(other, std::plus<T>()); }

    Tensor operator - (const Tensor& other) const { return broadcast(other, std::minus<T>()); }

    Tensor operator * (const Tensor& other) const { return broadcast(other, std::multiplies<T>()); }

    Tensor operator / (const Tensor& other) const { return broadcast(other, std::divides<T>()); }

/*
    friend std::ostream& operator << (std::ostream& os, const Tensor& t) {
        os << "[ ";
        for (const auto& v : t.data) os << v << " ";
        os << "]";
        return os;
    }*/
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        std::function<void(int, int, std::vector<int>&)> print_rec;

        print_rec = [&](int dim, int offset, std::vector<int>& idx) {
            if (dim == n - 1) {
                for (int i = 0; i < t.dims[dim]; ++i) {
                    idx[dim] = i;
                    int flat = 0;
                    for (int j = 0; j < n; ++j)
                        flat += idx[j] * t.cut[j];
                    os << t.data[flat];
                    if (i < t.dims[dim] - 1) os << " ";
                }
            } 
            else {
                os << "{\n";
                for (int i = 0; i < t.dims[dim]; ++i) {
                    idx[dim] = i;
                    print_rec(dim + 1, offset, idx);
                    if (i < t.dims[dim] - 1) os << "\n";
                }
                os << "\n}";
            }
        };
        std::vector<int> idx(n, 0);
        print_rec(0, 0, idx);
        return os;
    }

    template<typename T_val, int n_dims, typename Func>
    auto apply(const Tensor<T_val, n_dims>& input, Func func) {
        Tensor<T_val, n_dims> result(input.shape());

        auto in_it = input.cbegin();
        for (auto out_it = result.begin(); out_it != result.end(); ++out_it, ++in_it) {
            *out_it = func(*in_it);
        }
        
        return result;
    }
    
    template<typename T_, int n_>
    friend Tensor<T_, n_> transpose_2d(const Tensor<T_, n_>& t);

    template<typename T_, int n_>
    friend Tensor<T_, n_> matrix_product(const Tensor<T_, n_>& A, const Tensor<T_, n_>& B);
};
//use operators when scalar is to the left
template<typename T, int n>
Tensor<T, n> operator+(const T& change, const Tensor<T, n>& tensor) { return tensor + change;  }

template<typename T, int n>
Tensor<T, n> operator*(const T& scalar, const Tensor<T, n>& tensor) { return tensor * scalar; }
//non conmutatives
template<typename T, int n>
Tensor<T, n> operator-(const T& scalar, const Tensor<T, n>& tensor) {
    Tensor<T, n> result(tensor.dims);
    const auto total = tensor.data.size();
    for (size_t i = 0; i < total; ++i) {
        result.data[i] = static_cast<T>(scalar) - tensor.data[i];
    }
    return result;
}

template<typename T, int n>
Tensor<T, n> operator/(const T& scalar, const Tensor<T, n>& tensor) {
    Tensor<T, n> result(tensor.dims);
    const auto total = tensor.data.size();
    for (size_t i = 0; i < total; ++i) {
        // if you want to guard divide-by-zero you could check tensor.data[i] == 0 here
        result.data[i] = static_cast<T>(scalar) / tensor.data[i];
    }
    return result;
}

// transpose multidim
template<typename T, int n>
Tensor<T, n> transpose_2d(const Tensor<T, n>& t) {
    if (n < 2) {
        throw std::logic_error("Cannot transpose 1D tensor: need at least 2 dimensions");
    }

    std::array<int, n> new_shape = t.shape();
    std::swap(new_shape[n - 1], new_shape[n - 2]);

    Tensor<T, n> result(new_shape);

    int total = 1;
    for (auto d : t.shape()) {
        total *= d;
    } //result size

    std::array<int, n> index{};
    for (int i = 0; i < total; ++i) {
        // for flat dim in total
        int temp = i;
        for (int j = n - 1; j >= 0; --j) {
            index[j] = temp % t.shape()[j];
            temp /= t.shape()[j];
        }

        // swat last dimentions
        std::array<int, n> out_idx = index;
        std::swap(out_idx[n - 1], out_idx[n - 2]);

        // find value of index
        int flat_in = 0, flat_out = 0;
        for (int j = 0; j < n; ++j) {
            flat_in += index[j] * t.cut[j];
            flat_out += out_idx[j] * result.cut[j];
        }
        //add value to result
        result.data[flat_out] = t.data[flat_in];
    }
    return result;
}
template <typename T, int n>
Tensor<T, n> matrix_product(const Tensor<T, n>& A, const Tensor<T, n>& B) {
    if (n < 2){
        throw std::logic_error("Matrix product requires at least 2D tensors");
    }
    auto A_dims = A.shape();
    auto B_dims = B.shape();
    // if 2d 1
    int batchA = (n > 2) ? A_dims[0] : 1;
    int batchB = (n > 2) ? B_dims[0] : 1;
    // from las dims if ApxAq and BpxBq
    int Ap = A_dims[n - 2];//m
    int Aq = A_dims[n - 1];// must be
    int Bp = B_dims[n - 2];// the same
    int Bq = B_dims[n - 1];//nB

    if (Aq != Bp){
        throw std::logic_error("Matrix dimensions are incompatible for multiplication");
    }

    if (batchA != batchB && n > 2){
        throw std::logic_error("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
    }
    // 2 need other logic cuase it wont fit in shape calc
    // output dims
    std::array<int, n> out_shape{};
    if (n == 2) {
        out_shape = { Ap, Bq };
    } 
    else {
        out_shape[0] = batchA;
        out_shape[n - 2] = Ap;
        out_shape[n - 1] = Bq;
        for (int i = 1; i < n - 2; ++i){
            out_shape[i] = A_dims[i];
        }
    }

    Tensor<T, n> result(out_shape);
    result.fill(T{});
    //multiplication
    int total_batches = (n > 2) ? batchA : 1;
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < total_batches; ++b) {
        for (int i = 0; i < Ap; ++i) {
            for (int j = 0; j < Bq; ++j) {
                T sum = 0;
                for (int kk = 0; kk < Aq; ++kk) {
                    int idxA = (n > 2) ? b * A.cut[0] + i * A.cut[n - 2] + kk * A.cut[n - 1] : i * A.cut[0] + kk * A.cut[1];

                    int idxB = (n > 2) ? b * B.cut[0] + kk * B.cut[n - 2] + j * B.cut[n - 1] : kk * B.cut[0] + j * B.cut[1];

                    sum += A.data[idxA] * B.data[idxB];
                }

                int idxC = (n > 2) ? b * result.cut[0] + i * result.cut[n - 2] + j * result.cut[n - 1] : i * result.cut[0] + j * result.cut[1];

                result.data[idxC] = sum;
            }
        }
    }
    return result;
}
}

#endif //PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
