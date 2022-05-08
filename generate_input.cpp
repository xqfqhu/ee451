#include <dirent.h> 
#include <string.h>
#include<gsl/gsl_matrix.h>
#include <iostream>
#include <fstream>
#include <sstream>


#define DIM 150*150*3
#define CLASS_SIZE 6

void readX(std::ifstream &is, gsl_matrix * A, int &idx, int &suffix){
    std::string line;
    std::string tok;
    int i;
    char filename[] = "X0.dat";
    FILE * fp_write;
    while (true){
        for (i = idx; i < 3000; i++){
            std::getline(is, line);
            std::stringstream streamLine(line);
            int j;
            for (j = 0; j < DIM; j++){
                std::getline(streamLine, tok, ' ');
                gsl_matrix_set(A, i, j, stof(tok));
            }
            
            if (is.eof()){
                break;
            }
        }
        /*store */
        if (i != 3000){
            idx = i;
            break;
        }
        else{
            idx = 0;
            filename[1] = '0' + suffix;
            fp_write = fopen(filename, "w");
            gsl_matrix_fprintf(fp_write, A, "%lf");
            fclose(fp_write);
            suffix++;
        }
    }

}
void readY(std::ifstream &is, gsl_matrix * A, int &idx, int &suffix){
    char buffer[5];
    int i;
    std::string tok;
    std::string line;
    double value;
    char filename[] = "Y0.dat";
    FILE * fp_write;
    while (true){
        for (i = idx; i < 3000; i++){
            std::getline(is, line);
            
            
            for (int j = 0; j < CLASS_SIZE; j++){
                
                gsl_matrix_set(A, i, j, 0);
            }
            gsl_matrix_set(A, i, stoi(line), 1);
            
            if (is.eof()){
                break;
            }
        }
        /*store */
        if (i != 3000){
            idx = i;
            break;
        }
        else{
            idx = 0;
            filename[1] = '0' + suffix;
            fp_write = fopen(filename, "w");
            gsl_matrix_fprintf(fp_write, A, "%lf");
            fclose(fp_write);
            suffix++;
        }
    }

}
int main(int argc, char *argv[]){
    
    int num = 0;
    int num_Xfile = atoi(argv[1]);
    int idx = 0;
    int suffix = 0;
    char filenameX[] = "X0.dat";
    char filenameY[] = "Y0.dat";
    FILE * fp_write;
    gsl_matrix *A = gsl_matrix_calloc(3000, DIM);

    for (int i = 2; i < 2 + num_Xfile; i++){
        std::ifstream is(argv[i]);
        if (is.is_open()){
            
        
            readX(is, A, idx, suffix);
            is.close();
        }
        else{
            exit(1);
        }
    }
    if (idx != 0){
        gsl_matrix *B = gsl_matrix_calloc(idx, DIM);
        for (int i = 0; i < idx; i++){
            for (int j = 0; j < DIM; j++){
                gsl_matrix_set(B, i, j, gsl_matrix_get(A, i, j));
            }
        }
        filenameX[1] += suffix;
        fp_write = fopen(filenameX, "w");
        gsl_matrix_fprintf(fp_write, B, "%lf");
        fclose(fp_write);
        gsl_matrix_free(B);
        printf("file %s has %d entries\n", filenameX, idx);
    }
    gsl_matrix_free(A);


    idx = 0;
    suffix = 0;
    
    A = gsl_matrix_calloc(3000, CLASS_SIZE);

    for (int i = 2 + num_Xfile; i < 2 + 2* num_Xfile; i++){
        std::ifstream is(argv[i]);
        if (is.is_open()){
            readY(is, A, idx, suffix);
            is.close();
        }
        else{
            exit(1);
        }
    }
    if (idx != 0){
        gsl_matrix *B = gsl_matrix_calloc(idx, CLASS_SIZE);
        for (int i = 0; i < idx; i++){
            for (int j = 0; j < CLASS_SIZE; j++){
                gsl_matrix_set(B, i, j, gsl_matrix_get(A, i, j));
            }
        }
        filenameY[1] += suffix;
        fp_write = fopen(filenameY, "w");
        gsl_matrix_fprintf(fp_write, B, "%lf");
        fclose(fp_write);
        gsl_matrix_free(B);
        printf("file %s has %d entries\n", filenameY, idx);
    }
    gsl_matrix_free(A);
}