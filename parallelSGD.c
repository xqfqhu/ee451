#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include<gsl/gsl_vector.h>
#include<gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
//#include "mpi.h"
int **X_tr;
int *y_tr;

int read_file(char *file,int classifier,int* s)
{
    int i;
    FILE* in;
    in = fopen(file,"rb");
    if (!in){
        return 0;
    }else{
        fseek(in,0,SEEK_END);
        int len = ftell(in);
        if(len >= 10000){
            fseek(in,0, SEEK_SET);
            char* buffer = (char*) malloc (len * sizeof(char));
            fread(buffer,sizeof(char),len,in);
            for (i = 0; i < 10000; i++){
                X_tr[*s][i] = buffer[i];
            }
            y_tr[*s] = classifier;
            *s = *s + 1;
            free(buffer);
        }
        return 1;
    }
}

#define num 2//13986
#define large 2//22500
#define class_size 6

int main(int argc, char** argv){
    int* y = (int*) malloc (class_size*sizeof(int));
    for (int i=0;i<class_size;i++){
        y[i] = i;
    }
    
//    MPI_Init(&argc,&argv);
//    int size, rank;
//    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
//    MPI_Comm_size(MPI_COMM_WORLD,&size);
    
//    if (rank == 0){
    
//    X_tr = (int**) malloc (sizeof(int*) * num);
//    for (int i = 0; i < num; i++){
//        X_tr[i] = (int*) malloc(sizeof(int*)*10000);
//    }
//    y_tr = (int*) malloc(sizeof(int)* num);
//    //read training_data
//    int s = 0;
//    for(int i = 0;i < 6;i++){
//        DIR *d;
//        struct dirent *dir;
//        char name[30] ;
//        switch(i){
//            case 0:
//                strcpy(name, "./seg_train/buildings");
//                d = opendir(name);
//                if (d) {
//                    while ((dir = readdir(d)) != NULL) {
//                        if (strlen(dir->d_name)>=5){
//                            //printf("%s\n", dir->d_name);
//                            char file1[50],file2[10];
//                            strcpy(file1, name);
//                            strcpy(file2,dir->d_name);
//                            strcat(file1,"/");
//                            strcat(file1,file2);
//                            read_file(file1,i,&s);
//
//                        }
//                    }
//                }
//                closedir(d);
//                break;
//            case 1:
//                strcpy(name, "./seg_train/forest");
//                d = opendir(name);
//                if (d) {
//                    while ((dir = readdir(d)) != NULL) {
//                        if (strlen(dir->d_name)>=4){
//                            //printf("%s\n", dir->d_name);
//                            FILE* in;
//                            char file1[60],file2[30];
//                            strcpy(file1, name);
//                            strcpy(file2,dir->d_name);
//                            strcat(file1,"/");
//                            strcat(file1,file2);
//                            read_file(file1,i,&s);
//                        }
//                    }
//
//                }
//                closedir(d);
//                break;
//            case 2:
//                strcpy(name, "./seg_train/glacier");
//                d = opendir(name);
//                if (d) {
//                    while ((dir = readdir(d)) != NULL) {
//                        if (dir->d_name != "." && dir->d_name != ".."){
//                            //printf("%s\n", dir->d_name);
//                            FILE* in;
//                            char file1[50],file2[30];
//                            strcpy(file1, name);
//                            strcpy(file2,dir->d_name);
//                            strcat(file1,"/");
//                            strcat(file1,file2);
//                            read_file(file1,i,&s);
//                        }
//                    }
//
//                }
//                closedir(d);
//                break;
//            case 3:
//                strcpy(name, "./seg_train/mountain");
//                d = opendir(name);
//                if (d) {
//                    while ((dir = readdir(d)) != NULL) {
//                        if (dir->d_name != "." && dir->d_name != ".."){
//                            //printf("%s\n", dir->d_name);
//                            FILE* in;
//                            char file1[50],file2[30];
//                            strcpy(file1, name);
//                            strcpy(file2,dir->d_name);
//                            strcat(file1,"/");
//                            strcat(file1,file2);
//                            read_file(file1,i,&s);
//                        }
//                    }
//
//                }
//                closedir(d);
//                break;
//            case 4:
//                strcpy(name, "./seg_train/sea");
//                d = opendir(name);
//                if (d) {
//                    while ((dir = readdir(d)) != NULL) {
//                        if (dir->d_name != "." && dir->d_name != ".."){
//                            //printf("%s\n", dir->d_name);
//                            FILE* in;
//                            char file1[50],file2[30];
//                            strcpy(file1, name);
//                            strcpy(file2,dir->d_name);
//                            strcat(file1,"/");
//                            strcat(file1,file2);
//                            read_file(file1,i,&s);
//                        }
//                    }
//
//                }
//                closedir(d);
//                break;
//            case 5:
//                strcpy(name, "./seg_train/street");
//                d = opendir(name);
//                if (d) {
//                    while ((dir = readdir(d)) != NULL) {
//                        if (dir->d_name != "." && dir->d_name != ".."){
//                            //printf("%s\n", dir->d_name);
//                            FILE* in;
//                            char file1[50],file2[30];
//                            strcpy(file1, name);
//                            strcpy(file2,dir->d_name);
//                            strcat(file1,"/");
//                            strcat(file1,file2);
//                            read_file(file1,i,&s);
//                        }
//                    }
//
//                }
//                closedir(d);
//                break;
//        }
//
//    }
    
    
    
    
    //SGD
    gsl_matrix *X_tr = gsl_matrix_calloc(num, large);
    gsl_matrix *y_tr = gsl_matrix_calloc(num, 1);
    int data;
    FILE* in;
    in = fopen("a.txt","rb");
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
    in = fopen("b.txt","rb");
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
        printf("%f \n",gsl_matrix_get(w,i,0));
    }
    gsl_matrix *b = gsl_matrix_calloc(1, 1);
    random_value = rand()%256 * 1.0;
    gsl_matrix_set(b, 0, 0, random_value);
    gsl_matrix *lw = gsl_matrix_calloc(large, 1);;
    gsl_matrix *lb = gsl_matrix_calloc(1, 1);
    
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
        for (int i = 0; i < num; i++){
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
            gsl_matrix_add(lw, X_part3);
            
            gsl_matrix_set(lb,0,0,gsl_matrix_get(lb,0,0) -2*ytr);
        }
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
        printf("Epoch: %d, Loss: %.3f\n",epoch, loss);
        epoch+=1;
        lr = lr/divideby;
   }
    for (int i = 0; i< large;i++){
        float d =  gsl_matrix_get(w, i, 0);
        printf("%f ",d);
    }
    printf("%f", gsl_matrix_get(b,0,0));
//        return w,b
    
    
    
    
    
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
    
    
//
//    MPI_Finalize();
    
    return 0;
}
