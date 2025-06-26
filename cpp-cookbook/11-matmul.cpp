#include <iostream>
#include <vector>

using Row = std::vector<int>;
using Matrix = std::vector<Row>;

Matrix matmul(const Matrix &A, const Matrix &B) {
    int m = A.size();
    int n = A[0].size();
    int p = B[0].size();

    Matrix C(m, Row(p, 0));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < p; ++j)
            for (int k = 0; k < n; ++k)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

void printMatrix(const Matrix &M) {
    for (const auto &row : M) {
        for (const auto &elem : row)
            std::cout << elem << " ";
        std::cout << std::endl;
    }
}

int main() {
    Matrix A = {{1, 2, 3}, {4, 5, 6}};
    Matrix B = {{7, 8}, {9, 10}, {11, 12}};
    Matrix C = matmul(A, B);
    printMatrix(C);
}
/*
58 64
139 154
*/
