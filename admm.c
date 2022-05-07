#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include<gsl/gsl_vector.h>
#include<gsl/gsl_matrix.h>

#define BUFFER_SIZE 2000
#define LOCAL_DATA_SIZE 2000
#define PARAMETER_SIZE 10
#define MAX_ITER 50
#define CLASS_SIZE 6
#define lambda 0.5

int main(int argc, char *argv[]){
    int npes, myrank;
    MPI_Status status;
    FILE * fp;
    char line[BUFFER_SIZE];
    char * tok;
    double data;

    MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);  // total number of processes
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    gsl_matrix *A = gsl_matrix_calloc(LOCAL_DATA_SIZE, PARAMETER_SIZE);
    gsl_matrix *b = gsl_matrix_calloc(LOCAL_DATA_SIZE, CLASS_SIZE);
    gsl_matrix *x = gsl_maxtrix_calloc(PARAMETER_SIZE, CLASS_SIZE);
    gsl_matrix *z = gsl_maxtrix_calloc(PARAMETER_SIZE, CLASS_SIZE);
    gsl_matrix *u = gsl_maxtrix_calloc(PARAMETER_SIZE, CLASS_SIZE);
    gsl_matrix *u_hat = gsl_maxtrix_calloc(PARAMETER_SIZE, CLASS_SIZE);
    gsl_matrix *x_hat = gsl_maxtrix_calloc(PARAMETER_SIZE, CLASS_SIZE);
    gsl_matrix *x_u = gsl_maxtrix_calloc(PARAMETER_SIZE, CLASS_SIZE);
    gsl_matrix * ATA_rho = gsl_matrix_calloc(PARAMETER_SIZE, PARAMETER_SIZE);
    

    /* read in local data */
    fp = fopen("A.txt", "r");
    for (int i = 0; i < LOCAL_DATA_SIZE; i++){
        for (int j = 0; j < PARAMETER_SIZE; j++){
            fscanf(fp, "%lf", &data);
            gsl_matrix_set(A, i, j, data);
            
        }
    }
    fclose(fp);

    fp = fopen("b.txt", "r");
    for (int i = 0; i < LOCAL_DATA_SIZE; i++){
        for (int j = 0; j < CLASS_SIZE; j++){
            fscanf(fp, "%lf", &data);
            gsl_matrix_set(b, i, j, data);
        }
    }
    fclose(fp);

    /* precalc ATA_rho = A transpose * A + rho * I*/
    gsl_matrix_set_identity(ATA_rho);
    gsl_blas_dsyrk(CblasLower, CblasTrans, 1, A, rho, ATA_rho); // A_trans * A + rho * I
    

    /* start update */
    for (int itr = 0; itr < MAX_ITER; itr++){
        /* update u -> u_hat */
        gsl_matrix_memcpy(u_hat, u);
        gsl_matrix_add(u_hat, x); 
        gsl_matrix_sub(u_hat, z);  

        /* update x -> x_hat */
        gsl_matrix_memcpy(x_hat, z);
        gsl_matrix_sub(x_hat, u);
        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, A, b, rho, x_hat);
        gsl_blas_dtrsm(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 1, ATA_rho, x_hat);

        /* reduce x_hat + u_hat -> z */
        gsl_matrix_memcpy(x_u, x_hat);
        gsl_matrix_add(x_u, u_hat);
        MPI_Allreduce(x_u->data, z->data, PARAMETER_SIZE * CLASS_SIZE, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        /* normalize */
        gsl_matrix_scale(z, 1.0 / (double)npes);
        /* proximal operator */
        soft_threshold(z, lambda / ((double)npes * rho));
    }

    if (myrank == 0){
        fp = open("result.dat", "w");
        gsl_matrix_fprintf(fp, z, "%lf");
        fclose(fp);
    }
    MPI_Finalize();


    gsl_matrix_free(A);
    gsl_matrix_free(x);
    gsl_matrix_free(z);
    gsl_matrix_free(u);
    gsl_matrix_free(u_hat);
    gsl_matrix_free(x_hat);
    gsl_matrix_free(x_u);
    gsl_matrix_free(ATA_rho);

    return 0;
}

void soft_threshold(gsl_matrix * a, double k){
    double tmp;

    for (int i = 0; i < a->size1; i++){
        for (int j = 0; j < a->size2; j++){
            tmp = gsl_matrix_get(a, i, j);
            if (tmp > k){
                gsl_matrix_set(a, i,j, tmp - k);
            }
            else if (tmp <= k){
                gsl_matrix_set(a, i,j, 0);
            }
            else{
                gsl_matrix_set(a, i,j, tmp + k);
            }
        }
    }
}