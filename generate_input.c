#include <dirent.h> 
#include <stdio.h>
#define img_x 150
#define img_y 150
#define CLASS_SIZE 6
int main(int argc, char *argv[]){
    DIR *d;
    struct dirent *dir;
    int num = 0;
    if (argc <= 2){
        return 1;
    }
    d = opendir(argv[1]);
    if (d) {
        while ((dir = readdir(d)) != NULL) {
            num++;
        }
        closedir(d);
    }
    unsigned char *a = (unsigned char*) malloc (sizeof(unsigned char) * num * img_x * img_y);
    
    d = opendir(argv[1]);
    if (d) {
        while ((dir = readdir(d)) != NULL) {
            if (!(fp=fopen(dir->d_name, "rb"))){
                printf("can not open file\n");
                return 1;
            }
	        fread(a, sizeof(unsigned char), img_x * img_y, fp);
	        fclose(fp);
            a += (img_x * img_y);
        }
        closedir(d);
    }

    gsl_matrix *A = gsl_matrix_calloc(num, img_x * img_y);
    for (int i = 0; i < num; i++){
        for (int j = 0; j < img_x * img_y; j++){
            gsl_matrix_set(A, i, j, (double) a[i * img_x + j]);
        }
    }
    free(a);

    gsl_vector *y = gsl_matrix_calloc(num * CLASS_SIZE);
    for (int i = 0; i < num; i++){
        gsl_matrix_set(y, i, atoi(argv[2]), 1);
    }

    fp = open("A.dat", "w");
    gsl_matrix_fprintf(fp, A, "%lf");
    fclose(fp);

    fp = open("y.dat", "w");
    gsl_matrix_fprintf(fp, y, "%lf");
    fclose(fp);
}