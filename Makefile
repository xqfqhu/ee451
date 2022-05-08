all: admm
admm: admm.c
	mpicc -o admm admm.c `gsl-config --cflags --libs`
clean:
	rm -f admm