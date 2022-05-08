#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include<gsl/gsl_vector.h>
#include<gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include "mpi.h"
#include <time.h>
int **X_tr;
int *y_tr;

#define num 13986
#define large 2700
#define class_size 6

int main(int argc, char** argv){
    struct timespec start, stop;
    double time;
    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
    MPI_Init(&argc,&argv);
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    
    
    
    
    //SGD
    gsl_matrix *X_tr = gsl_matrix_calloc(num, large);
    gsl_matrix *y_tr = gsl_matrix_calloc(num, 1);
    int data;
    FILE* in;
    in = fopen("Xtrain.txt","rb");
    if (!in){
        return 0;
    }else{
        fseek(in,0,SEEK_END);
        int len = ftell(in);
        fseek(in,0, SEEK_SET);
        char* buffer = (char*) malloc (len * sizeof(char));
        fread(buffer,sizeof(char),len,in);
        int t = 0;
        for (int i = 0; i < num; i++){
            for (int j = 0; j < large; j++){
                int data = 0;
                while(t < len & buffer[t] != ' '){
                    data = data * 10 + buffer[t] - '0';
                    t++;
                }
                t++;
                gsl_matrix_set(X_tr,i,j,data);
            }
        }
        free(buffer);
    }
    fclose(in);
//    for (int i = 0; i< num; i++)
//        for (int j = 0; j < large; j++)
//            printf("%g", gsl_matrix_get(X_tr,i,j));
    in = fopen("Ytrain.txt","rb");
    if (!in){
        return 0;
    }else{
        fseek(in,0,SEEK_END);
        int len = ftell(in);
        fseek(in,0, SEEK_SET);
        char* buffer = (char*) malloc (len * sizeof(char));
        fread(buffer,sizeof(char),len,in);
        int t = 0;
        for (int i = 0; i < num; i++){
            int data = 0;
            while(t < len & buffer[t] != ' '){
                data = data * 10 + buffer[t] - '0';
                t++;
            }
            t++;
            gsl_matrix_set(y_tr,i,0,data);
        }
        free(buffer);
    }
    fclose(in);
    gsl_matrix *w = gsl_matrix_calloc(large, 1);
    float random_value = 0.0;
    for (int i = 0; i < large; i++){
        random_value = rand()%256 * 1.0;
        gsl_matrix_set(w, i, 0, random_value);
        //printf("%f \n",gsl_matrix_get(w,i,0));
    }
    gsl_matrix *b = gsl_matrix_calloc(1, 1);
    random_value = rand()%256 * 1.0;
    gsl_matrix_set(b, 0, 0, random_value);
    gsl_matrix *lw = gsl_matrix_calloc(large, 1);;
    gsl_matrix *lb = gsl_matrix_calloc(1, 1);
    gsl_matrix *lww = gsl_matrix_calloc(large, 1);;
    gsl_matrix *lbb = gsl_matrix_calloc(1, 1);
    
//    float b = rand()/100;
    float lr = 0.0002;
    int epoch = 1;
    int n_epochs = 1000;
    double divideby = 1.02;
    while (epoch <= n_epochs){
        for (int i = 0; i < large; i++)
            gsl_matrix_set(lw,i, 0, 0.0);
        gsl_matrix_set(lb,0,0,0.0);
        gsl_matrix *y_pred = gsl_matrix_calloc(num, 1);
        double loss = 0;
        int startIndex = num/size * (rank);
        int endIndex = num/size * (rank+1);
        for (int i = startIndex; i < endIndex; i++){
            double ytr = gsl_matrix_get(y_tr, i, 0);
            double Xdot = 0.0;
            gsl_matrix *X_part2 = gsl_matrix_calloc(1,large);
            gsl_matrix *X_part3 = gsl_matrix_calloc(large,1);
            for (int j = 0; j < large;j++){
                gsl_matrix_set(X_part2,0,j,gsl_matrix_get(X_tr,i,j));
                gsl_matrix_set(X_part3,j,0,gsl_matrix_get(X_tr,i,j));
            }
            for (int j = 0; j < large; j++){
                Xdot += gsl_matrix_get(X_part2,0,j) * gsl_matrix_get(w,j,0);
            }
            ytr = ytr - Xdot - gsl_matrix_get(b,0,0);
            gsl_matrix_scale(X_part3, -2.0*ytr);
            gsl_matrix_add(lww, X_part3);
            
            gsl_matrix_set(lbb,0,0,gsl_matrix_get(lbb,0,0) -2*ytr);
            gsl_matrix_free(X_part2);
            gsl_matrix_free(X_part3);
        }
        MPI_Allreduce(lww->data, lw->data, large, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(lbb->data, lb->data, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        gsl_matrix_scale(lw, lr);
//            for (int j = 0;j<large;j++)
//                printf("lw: %f\n",gsl_matrix_get(lw,j,0));
        gsl_matrix_sub(w,lw);
//            for (int j = 0;j<large;j++)
//                printf("w: %f\n",gsl_matrix_get(w,j,0));
        gsl_matrix_scale(lb, lr);
        gsl_matrix_sub(b,lb);
        //printf("%f %f %f %f\n", ytr, Xdot, gsl_matrix_get(w,0,0),gsl_matrix_get(w,1,0));
        double y_predicted = 0.0;
        for (int i = 0; i< num; i++){
            for (int j = 0; j< large; j++)
                y_predicted += gsl_matrix_get(X_tr, i, j) * gsl_matrix_get(w,j,0);
            y_predicted += gsl_matrix_get(b,0,0);
    //            printf("%f ", y_predicted);
            gsl_matrix_set(y_pred,i,0,y_predicted);
        }
        for (int i = 0; i< num; i++){
            loss = (gsl_matrix_get(y_pred,i,0)-gsl_matrix_get(y_tr,i,0)) * (gsl_matrix_get(y_pred,i,0)-gsl_matrix_get(y_tr,i,0));
        }
        loss = loss / large;
        //printf("Epoch: %d, Loss: %.3f\n",epoch, loss);
        epoch+=1;
        lr = lr/divideby;
        gsl_matrix_free(y_pred);
   }
//    for (int i = 0; i< large;i++){
//        float d =  gsl_matrix_get(w, i, 0);
//        printf("%f ",d);
//    }
//    printf("%f", gsl_matrix_get(b,0,0));
//        return w,b
    gsl_matrix_free(X_tr);
    gsl_matrix_free(y_tr);
    gsl_matrix_free(w);
    gsl_matrix_free(b);
    gsl_matrix_free(lw);
    gsl_matrix_free(lb);
    gsl_matrix_free(lww);
    gsl_matrix_free(lbb);
    
    
    
    
//    FILE* in1;
//    in1 = fopen("./seg_train/forest/13798.jpg","rb");
//    fseek(in1,0,SEEK_END);
//    int len1 = ftell(in1);
//    unsigned char *a1 = (unsigned char*) malloc (sizeof(unsigned char)*len1);
//    fseek(in1,0, SEEK_SET);
//    fread(a1,sizeof(char),len1,in1);
//    printf("%d ",len1);
//   FILE* in2;
//
//    in2 = fopen("0.jpg","rb");
//    fseek(in2,0,SEEK_END);
//    int len2 = ftell(in2);
//    unsigned char *a2 = (unsigned char*) malloc (sizeof(unsigned char)*len2);
//    fseek(in2,0, SEEK_SET);
//    fread(a2,sizeof(char),len2,in2);
//    printf("%d ",len2);
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    printf("Number of image fed into the classifier = %d Execution time = %f sec,\n", num * n_epochs, time);
    
//
    MPI_Finalize();
    
    return 0;
}
