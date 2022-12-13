#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>
#include <metis.h>
#include "mmio_highlevel.h"

#define VALUE_TYPE float
#define MPI_VALUE_TYPE MPI_FLOAT
#define NTIMES 20

struct MATRIX
{
    int rownum;
    int colnum;
    int nnznum;
    int *rowptr;
    int *colidx;
    VALUE_TYPE *val;
    int *nodeid;
};

struct COMMUN
{
    int infocount;
    int *sendid;
    int *recvid;
    int *index;
};

double gettime(struct timeval t1, struct timeval t2);
void initmtx(struct MATRIX *matrix);
void spmv_serial(int m, int *RowPtr, int *ColIdx, VALUE_TYPE *Val, VALUE_TYPE *Y, VALUE_TYPE *X);
void readmtx(char *filename, struct MATRIX *sourcematrix);
void mtx_display(struct MATRIX *matrix, char *filename);
void mtx_reorder(struct MATRIX *matrix, int *reorder_res, int num_parts);
int *mtx_partition(struct MATRIX *matrix, int part);
void vec_reorder(VALUE_TYPE *vec, int *reorder, int num_parts, int length);
void dividematrixbyrow(struct MATRIX sourcematrix, struct MATRIX *mtx, int partnum);
void dividematrixbynnz(struct MATRIX sourcematrix, struct MATRIX *mtx, int partnum);
void dividevector(VALUE_TYPE *sourcevector, VALUE_TYPE **vector, int partnum, int length, int *left_bound,int *right_bound);
int binary_search_right_boundary_kernel(const int *row_pointer, const int key_input, const int size);
void free_matrix(struct MATRIX mtx);
void writeresults(char *filename_res, char *filename, int m, int n, int nnzR, double rec_time, double spmv_time, double time, double GFlops, int processors, int nthreads);
void vec_restore(VALUE_TYPE *vec, int *reorder, int num_parts, int length);

int main(int argc, char **argv)
{
    int p, id, parts, *reorder;
    VALUE_TYPE *x, *sourcevector;
    char *filename;
    struct MATRIX sourcematrix;
    struct timeval t1, t2;
    double time;

    // Init MPI enviorment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Status status;
    MPI_Request request;
    struct MATRIX sub_local_mtx, sub_remote_mtx;
    struct COMMUN vec_commun;

    int *subm = (int *)malloc(sizeof(int) * p);
    int *submadd = (int *)malloc(sizeof(int) * (p + 1));
    int *subm2 = (int *)malloc(sizeof(int) * p);
    int *submadd2 = (int *)malloc(sizeof(int) * (p + 1));
    memset(subm, 0, sizeof(int) * p);
    memset(submadd, 0, sizeof(int) * (p + 1));
    memset(subm2, 0, sizeof(int) * p);
    memset(submadd2, 0, sizeof(int) * (p + 1));
    if (id == 0)
    {
        printf("process num:%d, thread num:%d \n",p,atoi(argv[2]));
        filename = argv[1];
        // master id read mtx
        readmtx(filename, &sourcematrix);
        // create comm struct
        struct COMMUN *comm = (struct COMMUN *)malloc(p * sizeof(struct COMMUN));
        for (int i = 0; i < p; i++)
        {
            comm[i].infocount = 0;
            comm[i].recvid = (int *)malloc(sizeof(int) * sourcematrix.colnum);
            comm[i].sendid = (int *)malloc(sizeof(int) * sourcematrix.colnum);
            comm[i].index = (int *)malloc(sizeof(int) * sourcematrix.colnum);
        }
        
        // create input vector
        sourcevector = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * sourcematrix.colnum);
        for (int i = 0; i < sourcematrix.colnum; i++)
            sourcevector[i] = rand() % 10 + 1; // 1.0;

        // printf("Input matrix:\n");
        // for(int i=0;i<=sourcematrix.rownum;i++) printf("%d ",sourcematrix.rowptr[i]);
        // printf("\n");
        // for(int i=0;i<sourcematrix.nnznum;i++) printf("%d %f\n",sourcematrix.colidx[i],sourcematrix.val[i]);
        // printf("\n");

        //预处理 start
        struct timeval t3, t4;
        gettimeofday(&t3,NULL);

        // reorder
        parts = p;
        if(parts>1) {
            reorder = (int *)malloc(sizeof(int) * sourcematrix.rownum);
            reorder = mtx_partition(&sourcematrix, parts);

            // for(int i=0;i<sourcematrix.rownum;i++) printf("%d ",reorder[i]);
            // printf("\n");
            mtx_reorder(&sourcematrix, reorder, parts);
            vec_reorder(sourcevector, reorder, parts, sourcematrix.colnum);
        }
        gettimeofday(&t1,NULL);
        double metis_time=gettime(t3,t1);
        printf("metis time: %.2f\n",metis_time);
        // printf("After reorder matrix:\n");
        // for(int i=0;i<=sourcematrix.rownum;i++) printf("%d ",sourcematrix.rowptr[i]);
        // printf("\n");
        // for(int i=0;i<sourcematrix.nnznum;i++) printf("%d %f\n",sourcematrix.colidx[i],sourcematrix.val[i]);
        // printf("\n");

        // printf("After reorder vector:\n");
        // for(int i=0;i<sourcematrix.colnum;i++) printf("%f\n",sourcevector[i]);

        // Divide a matrix into diagonal and off-diagonal matrices
        struct MATRIX local_mtx, remote_mtx;
        local_mtx.rownum = sourcematrix.rownum;
        local_mtx.rowptr = (int *)malloc(sizeof(int) * (local_mtx.rownum + 1));
        local_mtx.colidx = (int *)malloc(sizeof(int) * sourcematrix.nnznum);
        local_mtx.val = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * sourcematrix.nnznum);

        remote_mtx.rownum = sourcematrix.rownum;
        remote_mtx.rowptr = (int *)malloc(sizeof(int) * (remote_mtx.rownum + 1));
        remote_mtx.colidx = (int *)malloc(sizeof(int) * sourcematrix.nnznum);
        remote_mtx.val = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * sourcematrix.nnznum);

        int *subrowadd = (int *)malloc((p + 1) * sizeof(int));
        int *left_bound = (int *)malloc(sizeof(int) * (p + 1));

        for (int i = 0; i < p; i++)
        {
            subrowadd[i] = floor((double)sourcematrix.rownum / (double)p) * i;
            left_bound[i] = floor((double)sourcematrix.colnum / (double)p) * i;
        }
        subrowadd[p] = sourcematrix.rownum;
        left_bound[p] = sourcematrix.colnum;

        int *rowptr_start = (int *)malloc(sizeof(int) * local_mtx.rownum);
        int *rowptr_end = (int *)malloc(sizeof(int) * local_mtx.rownum);
        int *nnznum = (int *)malloc(sizeof(int) * p);
        memset(nnznum,0,sizeof(int)*p);
        //get rowptr_start rowptr_end and nnznum
        for (int i = 0; i < p; i++)
        {
            for(int j = subrowadd[i]; j < subrowadd[i + 1]; j++) {
                rowptr_start[j] = sourcematrix.nnznum;
                rowptr_end[j] = -1;
                for (int k = sourcematrix.rowptr[j]; k < sourcematrix.rowptr[j + 1]; k++) {
                    if (sourcematrix.colidx[k] >= left_bound[i] && sourcematrix.colidx[k] < left_bound[i + 1]) {
                        if(k<rowptr_start[j]) rowptr_start[j]=k; 
                        if(k>rowptr_end[j]) rowptr_end[j]=k;
                        nnznum[i]++;
                    }
                }
            }
        }
        //get lower_bound
        int lower_bound=0;
        for(int i=0;i<p;i++) {
            if(nnznum[i]>lower_bound) lower_bound=nnznum[i];
        }
        //move rowptr_end and right_bound
        int *right_bound = (int *)malloc(sizeof(int) * p);
        for(int i=0;i<p;i++) 
            right_bound[i]=left_bound[i+1]-1;

        for(int i=0;i<p;i++) {
            while(right_bound[i] + 1<local_mtx.colnum && nnznum[i]<lower_bound) {
                //get next end_bound
                int right=local_mtx.colnum;
                for(int j = subrowadd[i]; j < subrowadd[i + 1]; j++) {
                    if(rowptr_end[j]+1<sourcematrix.rowptr[j+1]) {
                        int next_col_id=sourcematrix.colidx[rowptr_end[j]+1];
                        if(next_col_id<right) right=next_col_id;
                    }
                }
                right_bound[i]=right;
                //move rowptr_end
                for(int j = subrowadd[i]; j < subrowadd[i + 1]; j++) {
                    if(rowptr_end[j]+1<sourcematrix.rowptr[j+1]) {
                        int next_col_id=sourcematrix.colidx[rowptr_end[j]+1];
                        if(next_col_id==right) {
                            rowptr_end[j]=next_col_id;
                            nnznum[i]++;
                        }
                    }
                }
            }
        }

        free(nnznum);
        free(rowptr_start);
        free(rowptr_end);

        // for(int i=0;i<p;i++) 
        //     printf("left_bound:%d, right_bound:%d\n",left_bound[i],right_bound[i]);

        int tmp_remote_nnz = 0, tmp_local_nnz = 0;
        remote_mtx.rowptr[0] = 0;
        local_mtx.rowptr[0] = 0;
        // Calculate the number of non-zero elements of the local and remote matrices, using the subvecadd array to calculate
        for (int i = 0; i < p; i++)
        {
            for (int j = subrowadd[i]; j < subrowadd[i + 1]; j++)
            {
                for (int k = sourcematrix.rowptr[j]; k < sourcematrix.rowptr[j + 1]; k++)
                {
                    if (sourcematrix.colidx[k] < left_bound[i] || sourcematrix.colidx[k] > right_bound[i])
                    {
                        remote_mtx.colidx[tmp_remote_nnz] = sourcematrix.colidx[k];
                        remote_mtx.val[tmp_remote_nnz] = sourcematrix.val[k];
                        tmp_remote_nnz++;
                    }
                    else
                    {
                        local_mtx.colidx[tmp_local_nnz] = sourcematrix.colidx[k];
                        local_mtx.val[tmp_local_nnz] = sourcematrix.val[k];
                        tmp_local_nnz++;
                    }
                }
                remote_mtx.rowptr[j + 1] = tmp_remote_nnz;
                local_mtx.rowptr[j + 1] = tmp_local_nnz;
            }
        }
        local_mtx.nnznum = tmp_local_nnz;
        local_mtx.colidx = (int *)realloc(local_mtx.colidx, sizeof(int) * local_mtx.nnznum);
        local_mtx.val = (VALUE_TYPE *)realloc(local_mtx.val, sizeof(VALUE_TYPE) * local_mtx.nnznum);

        remote_mtx.nnznum = tmp_remote_nnz;
        remote_mtx.colidx = (int *)realloc(remote_mtx.colidx, sizeof(int) * remote_mtx.nnznum);
        remote_mtx.val = (VALUE_TYPE *)realloc(remote_mtx.val, sizeof(VALUE_TYPE) * remote_mtx.nnznum);

        free(subrowadd);

        // printf("Divid local and remote matrix\n");
        // printf("local matrix\n");
        // for(int i=0;i<=local_mtx.rownum;i++) printf("%d ",local_mtx.rowptr[i]);
        // printf("\n");
        // for(int i=0;i<local_mtx.nnznum;i++) printf("%d %f\n",local_mtx.colidx[i],local_mtx.val[i]);
        // printf("\n");
        // printf("remote matrix\n");
        // for(int i=0;i<=remote_mtx.rownum;i++) printf("%d ",remote_mtx.rowptr[i]);
        // printf("\n");
        // for(int i=0;i<remote_mtx.nnznum;i++) printf("%d %f\n",remote_mtx.colidx[i],remote_mtx.val[i]);
        // printf("\n");

        // Create a submatrix
        struct MATRIX *local_mtx_arr = (struct MATRIX *)malloc(p * sizeof(struct MATRIX));
        struct MATRIX *remote_mtx_arr = (struct MATRIX *)malloc(p * sizeof(struct MATRIX));

        // Divide local matrix equally by row
        dividematrixbyrow(local_mtx, local_mtx_arr, p);

        // Divide remote matrix by the non-zero element uniformity
        dividematrixbynnz(remote_mtx, remote_mtx_arr, p);

        //preprocess end
        gettimeofday(&t4,NULL);
        double pre_time=gettime(t3,t4);
        double did_time=gettime(t1,t4);
        printf("divid time: %.2f\n",did_time);
        printf("preprocess time: %.2f\n",pre_time);

        for (int i = 0; i < p; i++)
        {
            subm[i] = local_mtx_arr[i].rownum;
            submadd[i] = i * local_mtx_arr[0].rownum;
            subm2[i] = remote_mtx_arr[i].rownum;
            if (i > 0)
                submadd2[i] = submadd2[i - 1] + remote_mtx_arr[i - 1].rownum;
        }
        submadd[p] = sourcematrix.rownum;
        submadd2[p] = sourcematrix.rownum;

        VALUE_TYPE **vector = (VALUE_TYPE **)malloc(sizeof(VALUE_TYPE *) * p);
        for (int i = 0; i < p; i++)
        {
            vector[i] = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * sourcematrix.colnum);
            memset(vector[i], 0, sizeof(VALUE_TYPE) * sourcematrix.colnum);
        }

        // divide vector equally by row
        dividevector(sourcevector, vector, p, sourcematrix.colnum, left_bound, right_bound);

        // printf("subvector:\n");
        // for(int j=0;j<p;j++) {
        //     for(int i=0;i<sourcematrix.colnum;i++) printf("%f\n",vector[j][i]);
        // }

        // compute vector index those need to be communicated

        int communicate_total_num = 0;
        int *flag = (int *)malloc(sizeof(int) * sourcematrix.colnum);
        for (int i = 0; i < p; i++)
        {
            memset(flag, 0, sizeof(int) * sourcematrix.colnum);
            for (int k = 0; k < remote_mtx_arr[i].nnznum; k++)
            {
                if ((remote_mtx_arr[i].colidx[k] < left_bound[i] || remote_mtx_arr[i].colidx[k] > right_bound[i]) && (flag[remote_mtx_arr[i].colidx[k]] == 0))
                {
                    int templength = floor(sourcematrix.colnum / p);
                    int need1 = (remote_mtx_arr[i].colidx[k] / templength) > (p - 1) ? (p - 1) : (remote_mtx_arr[i].colidx[k] / templength);
                    int need2 = remote_mtx_arr[i].colidx[k] - need1 * templength;
                    comm[need1].recvid[comm[need1].infocount] = i;
                    comm[need1].sendid[comm[need1].infocount] = need1;
                    comm[need1].index[comm[need1].infocount] = remote_mtx_arr[i].colidx[k];
                    flag[remote_mtx_arr[i].colidx[k]] = 1;
                    comm[need1].infocount++;
                    communicate_total_num++;
                }
            }
        }
        free(flag);
        free(left_bound);
        free(right_bound);

        sub_local_mtx.rownum = local_mtx_arr[id].rownum;
        sub_local_mtx.colnum = sourcematrix.colnum;
        sub_local_mtx.nnznum = local_mtx_arr[id].nnznum;
        initmtx(&sub_local_mtx);
        sub_local_mtx.rowptr = local_mtx_arr[id].rowptr;
        sub_local_mtx.colidx = local_mtx_arr[id].colidx;
        sub_local_mtx.val = local_mtx_arr[id].val;

        sub_remote_mtx.rownum = remote_mtx_arr[id].rownum;
        sub_remote_mtx.colnum = sourcematrix.colnum;
        sub_remote_mtx.nnznum = remote_mtx_arr[id].nnznum;
        initmtx(&sub_remote_mtx);
        sub_remote_mtx.rowptr = remote_mtx_arr[id].rowptr;
        sub_remote_mtx.colidx = remote_mtx_arr[id].colidx;
        sub_remote_mtx.val = remote_mtx_arr[id].val;

        vec_commun.infocount = comm[0].infocount;
        vec_commun.recvid = (int *)malloc(sizeof(int) * vec_commun.infocount);
        vec_commun.sendid = (int *)malloc(sizeof(int) * vec_commun.infocount);
        vec_commun.index = (int *)malloc(sizeof(int) * vec_commun.infocount);
        vec_commun.recvid = comm[0].recvid;
        vec_commun.sendid = comm[0].sendid;
        vec_commun.index = comm[0].index;

        x = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * sourcematrix.colnum);
        x = vector[id];


        // scatter matrix and vector
        for (int i = 1; i < p; i++)
        {
            MPI_Send(&local_mtx_arr[i].rownum, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(&sourcematrix.colnum, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
            MPI_Send(&local_mtx_arr[i].nnznum, 1, MPI_INT, i, 3, MPI_COMM_WORLD);
            MPI_Send(local_mtx_arr[i].rowptr, local_mtx_arr[i].rownum + 1, MPI_INT, i, 4, MPI_COMM_WORLD);
            MPI_Send(local_mtx_arr[i].colidx, local_mtx_arr[i].nnznum, MPI_INT, i, 5, MPI_COMM_WORLD);
            MPI_Send(local_mtx_arr[i].val, local_mtx_arr[i].nnznum, MPI_VALUE_TYPE, i, 6, MPI_COMM_WORLD);
            MPI_Send(vector[i], sourcematrix.colnum, MPI_VALUE_TYPE, i, 7, MPI_COMM_WORLD);
            MPI_Send(&comm[i].infocount, 1, MPI_INT, i, 8, MPI_COMM_WORLD);
            MPI_Send(comm[i].recvid, comm[i].infocount, MPI_INT, i, 9, MPI_COMM_WORLD);
            MPI_Send(comm[i].index, comm[i].infocount, MPI_INT, i, 10, MPI_COMM_WORLD);
            MPI_Send(comm[i].sendid, comm[i].infocount, MPI_INT, i, 11, MPI_COMM_WORLD);

            MPI_Send(&remote_mtx_arr[i].rownum, 1, MPI_INT, i, 12, MPI_COMM_WORLD);
            MPI_Send(&sourcematrix.colnum, 1, MPI_INT, i, 13, MPI_COMM_WORLD);
            MPI_Send(&remote_mtx_arr[i].nnznum, 1, MPI_INT, i, 14, MPI_COMM_WORLD);
            MPI_Send(remote_mtx_arr[i].rowptr, remote_mtx_arr[i].rownum + 1, MPI_INT, i, 15, MPI_COMM_WORLD);
            MPI_Send(remote_mtx_arr[i].colidx, remote_mtx_arr[i].nnznum, MPI_INT, i, 16, MPI_COMM_WORLD);
            MPI_Send(remote_mtx_arr[i].val, remote_mtx_arr[i].nnznum, MPI_VALUE_TYPE, i, 17, MPI_COMM_WORLD);
        }
    }
    else
    {
        // receive matrix and vector
        MPI_Recv(&sub_local_mtx.rownum, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&sub_local_mtx.colnum, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
        MPI_Recv(&sub_local_mtx.nnznum, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);
        initmtx(&sub_local_mtx);
        MPI_Recv(sub_local_mtx.rowptr, sub_local_mtx.rownum + 1, MPI_INT, 0, 4, MPI_COMM_WORLD, &status);
        MPI_Recv(sub_local_mtx.colidx, sub_local_mtx.nnznum, MPI_INT, 0, 5, MPI_COMM_WORLD, &status);
        MPI_Recv(sub_local_mtx.val, sub_local_mtx.nnznum, MPI_VALUE_TYPE, 0, 6, MPI_COMM_WORLD, &status);
        x = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * sub_local_mtx.colnum);
        MPI_Recv(x, sub_local_mtx.colnum, MPI_VALUE_TYPE, 0, 7, MPI_COMM_WORLD, &status);
        MPI_Recv(&vec_commun.infocount, 1, MPI_INT, 0, 8, MPI_COMM_WORLD, &status);
        vec_commun.recvid = (int *)malloc(sizeof(int) * vec_commun.infocount);
        vec_commun.index = (int *)malloc(sizeof(int) * vec_commun.infocount);
        vec_commun.sendid = (int *)malloc(sizeof(int) * vec_commun.infocount);
        MPI_Recv(vec_commun.recvid, vec_commun.infocount, MPI_INT, 0, 9, MPI_COMM_WORLD, &status);
        MPI_Recv(vec_commun.index, vec_commun.infocount, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);
        MPI_Recv(vec_commun.sendid, vec_commun.infocount, MPI_INT, 0, 11, MPI_COMM_WORLD, &status);

        MPI_Recv(&sub_remote_mtx.rownum, 1, MPI_INT, 0, 12, MPI_COMM_WORLD, &status);
        MPI_Recv(&sub_remote_mtx.colnum, 1, MPI_INT, 0, 13, MPI_COMM_WORLD, &status);
        MPI_Recv(&sub_remote_mtx.nnznum, 1, MPI_INT, 0, 14, MPI_COMM_WORLD, &status);
        initmtx(&sub_remote_mtx);
        MPI_Recv(sub_remote_mtx.rowptr, sub_remote_mtx.rownum + 1, MPI_INT, 0, 15, MPI_COMM_WORLD, &status);
        MPI_Recv(sub_remote_mtx.colidx, sub_remote_mtx.nnznum, MPI_INT, 0, 16, MPI_COMM_WORLD, &status);
        MPI_Recv(sub_remote_mtx.val, sub_remote_mtx.nnznum, MPI_VALUE_TYPE, 0, 17, MPI_COMM_WORLD, &status);
    }

    // Calculate the nodeid array, which is used when finding the accepted vector index
    sub_remote_mtx.nodeid = (int *)malloc(sizeof(int) * sub_remote_mtx.nnznum);
    int templength = floor(sub_remote_mtx.colnum / p), divisor = 0;
    for (int i = 0; i < sub_remote_mtx.rownum; i++)
    {
        for (int j = sub_remote_mtx.rowptr[i]; j < sub_remote_mtx.rowptr[i + 1]; j++)
        {
            divisor = sub_remote_mtx.colidx[j] / templength;
            sub_remote_mtx.nodeid[j] = divisor > (p - 1) ? (p - 1) : divisor;
        }
    }

    // start communicate
    MPI_Request *vec_request = (MPI_Request *)malloc(sizeof(MPI_Request) * (p - 1));
    MPI_Status *vec_status = (MPI_Status *)malloc(sizeof(MPI_Status) * (p - 1));
    VALUE_TYPE *local_y = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * sub_local_mtx.rownum);
    VALUE_TYPE *remote_y = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * sub_remote_mtx.rownum);

    int send_cnt[p], recv_cnt[p];
    int **comm_vec_ind = (int **)malloc(sizeof(int *) * p);
    VALUE_TYPE **comm_vec_val = (VALUE_TYPE **)malloc(sizeof(VALUE_TYPE *) * p);
    for (int i = 0; i < p; i++)
    {
        send_cnt[i] = 0;
        recv_cnt[i] = 0;
        comm_vec_ind[i] = (int *)malloc(sizeof(int) * vec_commun.infocount);
        comm_vec_val[i] = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * vec_commun.infocount);
    }
    for (int i = 0; i < vec_commun.infocount; i++)
    {
        comm_vec_val[vec_commun.recvid[i]][send_cnt[vec_commun.recvid[i]]] = x[vec_commun.index[i]];
        comm_vec_ind[vec_commun.recvid[i]][send_cnt[vec_commun.recvid[i]]] = vec_commun.index[i];
        send_cnt[vec_commun.recvid[i]]++;
    }

    MPI_Alltoall(send_cnt, 1, MPI_INT, recv_cnt, 1, MPI_INT, MPI_COMM_WORLD);
    //printf("communicate over\n");
    // overlap
    VALUE_TYPE **recv_vec = (VALUE_TYPE **)malloc(sizeof(VALUE_TYPE *) * p);
    for (int q = 0; q < p; q++)
        if (recv_cnt[q])
            recv_vec[q] = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * recv_cnt[q]);

    int nthreads = atoi(argv[2]);
    omp_set_num_threads(nthreads);
    // printf("thread number %d\n", nthreads);

    int *pos = (int *)malloc(sizeof(int) * p);
    double local_time=1000000000;
    double remote_time=1000000000;
    double sdrc_time=1000000000;

    for (int r = 0; r < NTIMES; r++)
    {   
        MPI_Barrier(MPI_COMM_WORLD);
        if (id == 0) gettimeofday(&t1, NULL);
        for (int i = 0; i < p; i++)
        {
            if (i != id)
            {
                //id send msg to i,include vector's index and value 
                //printf("id %d send %d number to id%d\n", id, temp[i], i);
                //MPI_Send(&send_cnt[i], 1, MPI_INT, i, 11, MPI_COMM_WORLD);
                MPI_Isend(comm_vec_val[i], send_cnt[i], MPI_FLOAT, i, 13, MPI_COMM_WORLD, &request);
                //printf("id %d %d %4.2f\n", id, commun_vec_index[i][0], commun_vec[i][0]);
            }
        }
        for (int i = 0; i < p; i++)
        {
            if (i != id)
            {
                //MPI_Recv(&recv_cnt[i], 1, MPI_INT, i, 11, MPI_COMM_WORLD, &status);
                MPI_Recv(recv_vec[i], recv_cnt[i], MPI_FLOAT, i, 13, MPI_COMM_WORLD, &status);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (id == 0) {
            gettimeofday(&t2, NULL);
            double cur_time=gettime(t1,t2);
            sdrc_time+=cur_time;
        }
    }

    for (int i = 0; i < p; i++)
        pos[i] = 0;
    for (int i = 0; i < sub_remote_mtx.nnznum; i++)
        if (!x[sub_remote_mtx.colidx[i]])
            x[sub_remote_mtx.colidx[i]] = recv_vec[sub_remote_mtx.nodeid[i]][pos[sub_remote_mtx.nodeid[i]]++];
    
        // find balanced points
    int *csrSplitter = (int *)malloc((nthreads+1) * sizeof(int));
    int stridennz = ceil((double)sub_local_mtx.nnznum/(double)nthreads);
    #pragma omp parallel for
    for (int tid = 0; tid <= nthreads; tid++)
    {
            int boundary = tid * stridennz;
            boundary = boundary > sub_local_mtx.nnznum ? sub_local_mtx.nnznum : boundary;
            csrSplitter[tid] = binary_search_right_boundary_kernel(sub_local_mtx.rowptr, boundary, sub_local_mtx.rownum + 1) - 1;
    }
    
    for (int iter = 0; iter < NTIMES; iter++)
    {        
        MPI_Barrier(MPI_COMM_WORLD);
        if(id==0) gettimeofday(&t1,NULL);
// Compute local spmv
// #pragma omp parallel for
//         for (int i = 0; i < sub_local_mtx.rownum; i++)
//         {
//             local_y[i]=0;
//             for (int j = sub_local_mtx.rowptr[i]; j < sub_local_mtx.rowptr[i + 1]; j++)
//             {
//                 local_y[i] += sub_local_mtx.val[j] * x[sub_local_mtx.colidx[j]];
//             }
//         }
        #pragma omp parallel for
        for (int tid = 0; tid < nthreads; tid++)
        {
            for (int u = csrSplitter[tid]; u < csrSplitter[tid+1]; u++)
            {
                local_y[u] = 0;
                for (int h = sub_local_mtx.rowptr[u]; h < sub_local_mtx.rowptr[u+1]; h++)
                {
                    local_y[u] += sub_local_mtx.val[h] * x[sub_local_mtx.colidx[h]];
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (id == 0) {
            gettimeofday(&t2, NULL);
            double cur_time=gettime(t1,t2);
            local_time+=cur_time;
        }
    }

    stridennz = ceil((double)sub_remote_mtx.nnznum/(double)nthreads);
    #pragma omp parallel for
    for (int tid = 0; tid <= nthreads; tid++)
    {
            int boundary = tid * stridennz;
            boundary = boundary > sub_remote_mtx.nnznum ? sub_remote_mtx.nnznum : boundary;
            csrSplitter[tid] = binary_search_right_boundary_kernel(sub_remote_mtx.rowptr, boundary, sub_remote_mtx.rownum + 1) - 1;
    }

    for (int iter = 0; iter < NTIMES; iter++)
    {
// #pragma omp parallel for
//         for (int i = 0; i < sub_remote_mtx.rownum; i++) {
//             remote_y[i]=0;
//             for (int j = sub_remote_mtx.rowptr[i]; j < sub_remote_mtx.rowptr[i + 1]; j++)
//                 remote_y[i] += sub_remote_mtx.val[j] * x[sub_remote_mtx.colidx[j]];
//         }
        MPI_Barrier(MPI_COMM_WORLD);
        if(id==0) gettimeofday(&t1,NULL);
        #pragma omp parallel for
        for (int tid = 0; tid < nthreads; tid++)
        {
            for (int u = csrSplitter[tid]; u < csrSplitter[tid+1]; u++)
            {
                remote_y[u] = 0;
                for (int h = sub_remote_mtx.rowptr[u]; h < sub_remote_mtx.rowptr[u+1]; h++)
                {
                    remote_y[u] += sub_remote_mtx.val[h] * x[sub_remote_mtx.colidx[h]];
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (id == 0) {
            gettimeofday(&t2, NULL);
            double cur_time=gettime(t1,t2);
            remote_time+=cur_time;
        }
    }
    /*
    printf("Output sub_remote_mtx\n");
    for(int i=0;i<sub_remote_mtx.rownum;i++) printf("remote_y:%f\n",remote_y[i]);
    */
    //printf("overlap over\n");
    // gather
    MPI_Bcast(subm, p, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(submadd, p + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(subm2, p, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(submadd2, p + 1, MPI_INT, 0, MPI_COMM_WORLD);
    VALUE_TYPE *result1 = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * sourcematrix.rownum);
    VALUE_TYPE *result2 = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * sourcematrix.rownum);
    MPI_Gatherv(local_y, sub_local_mtx.rownum, MPI_VALUE_TYPE, result1, subm, submadd, MPI_VALUE_TYPE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(remote_y, sub_remote_mtx.rownum, MPI_VALUE_TYPE, result2, subm2, submadd2, MPI_VALUE_TYPE, 0, MPI_COMM_WORLD);
    //printf("gather over\n");

    if (id == 0)
    {
        sdrc_time/=NTIMES;
        local_time/=NTIMES;
        remote_time/=NTIMES;
        time=local_time+remote_time;
        double GFlops = 2 * sourcematrix.nnznum / (sdrc_time + time) / pow(10, 6);
        printf("send + recv time: %4.2f ms\n",sdrc_time);
        printf("local spmv time: %4.2f ms\n",local_time);
        //printf("wait time: %4.2f ms\n",wait_time/NTIMES);
        printf("remote spmv time: %4.2f ms\n",remote_time);
        printf("local spmv time + remote spmv time %4.2f ms\n", time);
        printf("overlap spmv time %4.2f ms, %4.2f GFlops\n", time+sdrc_time, GFlops);
        //writeresults("DistSpMV_Balanced.csv", filename, sourcematrix.rownum, sourcematrix.colnum, sourcematrix.nnznum,sdrc_time,time, sdrc_time+time, GFlops, p, nthreads);

        VALUE_TYPE *result = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * sourcematrix.rownum);

        for (int i = 0; i < sourcematrix.rownum; i++)
        {
            result[i] = result1[i] + result2[i];
            //    printf("local_val:%f,remote_val:%f\n",result1[i],result2[i]);
            //printf("%f ",result[i]);
        }
        //printf("\n");

        int errorcnt = 0;
        VALUE_TYPE *y_golden = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * sourcematrix.rownum);
        memset(y_golden, 0, sizeof(VALUE_TYPE) * sourcematrix.rownum);
        spmv_serial(sourcematrix.rownum, sourcematrix.rowptr, sourcematrix.colidx, sourcematrix.val, y_golden, sourcevector);

        for (int i = 0; i < sourcematrix.rownum; i++)
        {
            // printf("rowid=%d, result=%f, y_golden=%f\n",i,result[i],y_golden[i]);
            if (result[i] != y_golden[i])
            {
                errorcnt++;
                // printf("result[%d] %f, y_golden[%d] %f\n", i, result[i], i, y_golden[i]);
            }
        }
        printf("error count %d \n", errorcnt);

        VALUE_TYPE *restore_result;
        restore_result = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * sourcematrix.colnum);
        for (int i = 0; i < sourcematrix.rownum; i++)
            restore_result[i]=result[i];
        vec_restore(restore_result, reorder, parts, sourcematrix.colnum);
        
        //output resvector
        // for (int i = 0; i < sourcematrix.rownum; i++)
        //     printf("%f ",restore_result[i]);
        // printf("\n");
    }

    MPI_Finalize();
}

double gettime(struct timeval t1, struct timeval t2)
{
    return (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
}

void initmtx(struct MATRIX *matrix)
{
    matrix->rowptr = (int *)malloc(sizeof(int) * (matrix->rownum + 1));
    matrix->colidx = (int *)malloc(sizeof(int) * matrix->nnznum);
    matrix->val = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * matrix->nnznum);
}

void spmv_serial(int m, int *RowPtr, int *ColIdx, VALUE_TYPE *Val, VALUE_TYPE *Y, VALUE_TYPE *X)
{
    for (int i = 0; i < m; i++)
    {
        Y[i] = 0;
        for (int j = RowPtr[i]; j < RowPtr[i + 1]; j++)
        {
            Y[i] += Val[j] * X[ColIdx[j]];
        }
    }
}

void readmtx(char *filename, struct MATRIX *sourcematrix)
{
    int isSymmetric;
    printf("filename = %s\n", filename);
    mmio_info(&sourcematrix->rownum, &sourcematrix->colnum, &sourcematrix->nnznum, &isSymmetric, filename);
    sourcematrix->rowptr = (int *)malloc((sourcematrix->rownum + 1) * sizeof(int));
    sourcematrix->colidx = (int *)malloc(sourcematrix->nnznum * sizeof(int));
    sourcematrix->val = (VALUE_TYPE *)malloc(sourcematrix->nnznum * sizeof(VALUE_TYPE));
    mmio_data(sourcematrix->rowptr, sourcematrix->colidx, sourcematrix->val, filename);
    printf("Matrix A is %i by %i, #nonzeros = %i\n", sourcematrix->rownum, sourcematrix->colnum, sourcematrix->nnznum);
    for (int i = 0; i < sourcematrix->nnznum; i++)
        sourcematrix->val[i] = 1.0;
}

void mtx_display(struct MATRIX *matrix, char *filename)
{
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%d %d %d\n", matrix->rownum, matrix->colnum, matrix->nnznum);
    for (int i = 0; i < matrix->rownum; i++)
        for (int j = matrix->rowptr[i]; j < matrix->rowptr[i + 1]; j++)
            fprintf(fp, "%d %d 1\n", i, matrix->colidx[j]);
    fclose(fp);
}

void mtx_reorder(struct MATRIX *matrix, int *reorder_res, int num_parts)
{
    struct MATRIX reordered_matrix;
    reordered_matrix.rownum = matrix->rownum;
    reordered_matrix.colnum = matrix->colnum;
    reordered_matrix.nnznum = matrix->nnznum;
    reordered_matrix.rowptr = (int *)malloc(sizeof(int) * (reordered_matrix.rownum + 1));
    int *temp_rowptr = (int *)malloc(sizeof(int) * (reordered_matrix.rownum + 1));
    reordered_matrix.colidx = (int *)malloc(sizeof(int) * reordered_matrix.nnznum);
    for (int i = 0; i <= reordered_matrix.rownum; i++)
    {
        reordered_matrix.rowptr[i] = 0;
        temp_rowptr[i] = 0;
    }
    for (int i = 0; i < reordered_matrix.nnznum; i++)
        reordered_matrix.colidx[i] = 0;
    reordered_matrix.val = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * reordered_matrix.nnznum);
    int *cnt = (int *)malloc(sizeof(int) * (num_parts + 1));
    for (int i = 0; i <= num_parts; i++)
        cnt[i] = 0;
    int *index = (int *)malloc(sizeof(int) * num_parts);
    for (int i = 0; i < num_parts; i++)
        index[i] = 0;
    int *temp_reorder = (int *)malloc(sizeof(int) * matrix->rownum);
    for (int i = 0; i < matrix->rownum; i++)
    {
        cnt[reorder_res[i]]++;
        temp_reorder[i] = 0;
    }
    exclusive_scan(cnt, num_parts + 1);
    for (int i = 0; i < num_parts; i++)
        index[i] = 0;
    for (int i = 0; i < matrix->rownum; i++)
    {
        temp_reorder[i] = cnt[reorder_res[i]] + index[reorder_res[i]];
        index[reorder_res[i]] = index[reorder_res[i]] + 1;
    }

    // row reorder
    for (int i = 0; i < matrix->rownum; i++)
        reordered_matrix.rowptr[temp_reorder[i]] = matrix->rowptr[i + 1] - matrix->rowptr[i];
    exclusive_scan(reordered_matrix.rowptr, (matrix->rownum + 1));

    for (int i = 0; i < matrix->rownum; i++)
    {
        int nnz_cnt = 0;
        for (int j = matrix->rowptr[i]; j < matrix->rowptr[i + 1]; j++)
        {
            reordered_matrix.colidx[reordered_matrix.rowptr[temp_reorder[i]] + nnz_cnt] = matrix->colidx[j];
            nnz_cnt += 1;
        }
    }

    // colnum reorder
    for (int i = 0; i < reordered_matrix.nnznum; i++)
        reordered_matrix.colidx[i] = temp_reorder[reordered_matrix.colidx[i]];
    for (int i = 0; i < reordered_matrix.nnznum; i++)
        reordered_matrix.val[i] = 1.0;

    *matrix = reordered_matrix;
    free(cnt);
    free(index);
    free(temp_reorder);
    free(temp_rowptr);
}

int *mtx_partition(struct MATRIX *matrix, int part)
{
    idx_t rownum = matrix->rownum, weights = 1, objval, parts = part;
    idx_t *xadj = (idx_t *)malloc(sizeof(idx_t) * (rownum + 1));
    idx_t *reorder = (idx_t *)malloc(sizeof(idx_t) * rownum);
    xadj[0] = 0;
    for (int i = 0; i < rownum; i++)
    {
        int flag = 0;
        for (int j = matrix->rowptr[i]; j < matrix->rowptr[i + 1]; j++)
        {
            if (matrix->colidx[j] == i)
            {
                flag = 1;
                break;
            }
        }
        if (flag == 1)
            xadj[i + 1] = matrix->rowptr[i + 1] - matrix->rowptr[i] - 1;
        else
            xadj[i + 1] = matrix->rowptr[i + 1] - matrix->rowptr[i];
    }
    for (int i = 1; i <= rownum; i++)
        xadj[i] += xadj[i - 1];
    idx_t *adjncy = (idx_t *)malloc(sizeof(idx_t) * xadj[rownum]);
    for (int i = 0; i < rownum; i++)
    {
        int tt = xadj[i];
        int ttt = 0;
        for (int j = matrix->rowptr[i]; j < matrix->rowptr[i + 1]; j++)
        {
            if (matrix->colidx[j] == i)
                continue;
            else
            {
                adjncy[tt + ttt] = matrix->colidx[j];
                ttt++;
            }
        }
    }
    METIS_PartGraphKway(&rownum, &weights, xadj, adjncy, NULL, NULL, NULL, &parts, NULL, NULL, NULL, &objval, reorder);
    int *order = (int *)malloc(sizeof(int) * rownum);
    for (int i = 0; i < rownum; i++)
        order[i] = reorder[i];
    free(reorder);
    free(adjncy);
    free(xadj);
    return order;
}

void vec_reorder(VALUE_TYPE *vec, int *reorder, int num_parts, int length)
{
    int *cnt = (int *)malloc(sizeof(int) * (num_parts + 1));
    for (int i = 0; i <= num_parts; i++)
        cnt[i] = 0;
    int *index = (int *)malloc(sizeof(int) * num_parts);
    for (int i = 0; i < num_parts; i++)
        index[i] = 0;
    int *temp_reorder = (int *)malloc(sizeof(int) * length);
    VALUE_TYPE *temp_vec = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * length);
    for (int i = 0; i < length; i++)
    {
        cnt[reorder[i]]++;
        temp_reorder[i] = 0;
        temp_vec[i] = 0;
    }
    exclusive_scan(cnt, num_parts + 1);
    for (int i = 0; i < num_parts; i++)
        index[i] = 0;
    for (int i = 0; i < length; i++)
    {
        temp_reorder[i] = cnt[reorder[i]] + index[reorder[i]];
        index[reorder[i]] = index[reorder[i]] + 1;
    }
    for (int i = 0; i < length; i++)
        temp_vec[temp_reorder[i]] = vec[i];
    for (int i = 0; i < length; i++)
        vec[i] = temp_vec[i];
}

void vec_restore(VALUE_TYPE *vec, int *reorder, int num_parts, int length)
{
    int *cnt = (int *)malloc(sizeof(int) * (num_parts + 1));
    for (int i = 0; i <= num_parts; i++)
        cnt[i] = 0;
    int *index = (int *)malloc(sizeof(int) * num_parts);
    for (int i = 0; i < num_parts; i++)
        index[i] = 0;
    int *temp_reorder = (int *)malloc(sizeof(int) * length);
    VALUE_TYPE *temp_vec = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * length);
    for (int i = 0; i < length; i++)
    {
        cnt[reorder[i]]++;
        temp_reorder[i] = 0;
        temp_vec[i] = 0;
    }
    exclusive_scan(cnt, num_parts + 1);
    for (int i = 0; i < num_parts; i++)
        index[i] = 0;
    for (int i = 0; i < length; i++)
    {
        temp_reorder[i] = cnt[reorder[i]] + index[reorder[i]];
        index[reorder[i]] = index[reorder[i]] + 1;
    }
    for (int i = 0; i < length; i++)
        temp_vec[i] = vec[temp_reorder[i]];
    for (int i = 0; i < length; i++)
        vec[i] = temp_vec[i];
}

void dividematrixbyrow(struct MATRIX sourcematrix, struct MATRIX *mtx, int partnum)
{
    int *subrowadd = (int *)malloc((partnum + 1) * sizeof(int));
    for (int i = 0; i < partnum; i++)
    {
        subrowadd[i] = floor(sourcematrix.rownum / partnum * i);
    }
    subrowadd[partnum] = sourcematrix.rownum;
    for (int i = 0; i < partnum; i++)
    {
        mtx[i].rownum = subrowadd[i + 1] - subrowadd[i];
        // mtx[i].colnum = sourcematrix.colnum;
    }
    int start, end;
    for (int i = 0; i < partnum; i++)
    {
        start = subrowadd[i];
        end = start + mtx[i].rownum;
        mtx[i].nnznum = sourcematrix.rowptr[end] - sourcematrix.rowptr[start];
    }
    int offset = 0;
    for (int i = 0; i < partnum; i++)
    {
        mtx[i].rowptr = (int *)malloc(sizeof(int) * (mtx[i].rownum + 1));
        memset(mtx[i].rowptr, 0, sizeof(int) * (mtx[i].rownum + 1));
        mtx[i].colidx = (int *)malloc(sizeof(int) * mtx[i].nnznum);
        memset(mtx[i].colidx, 0, sizeof(int) * mtx[i].nnznum);
        mtx[i].val = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * mtx[i].nnznum);
        memset(mtx[i].val, 0, sizeof(VALUE_TYPE) * mtx[i].nnznum);
        int temp;
        if (i == 0)
            temp = 0;
        else
            temp = sourcematrix.rowptr[subrowadd[i]];

        for (int j = 0; j < mtx[i].rownum + 1; j++)
        {
            mtx[i].rowptr[j] = sourcematrix.rowptr[subrowadd[i] + j] - temp;
        }

        for (int j = 0; j < mtx[i].nnznum; j++)
        {
            mtx[i].colidx[j] = sourcematrix.colidx[offset + j];
            mtx[i].val[j] = sourcematrix.val[offset + j];
        }
        offset += mtx[i].nnznum;
    }

    free(subrowadd);
}

void dividematrixbynnz(struct MATRIX sourcematrix, struct MATRIX *mtx, int partnum)
{
    // printf("Remote matrix:\n");
    // for(int i=0;i<=sourcematrix.rownum;i++) printf("%d ",sourcematrix.rowptr[i]);
    // printf("\n");
    // for(int i=0;i<sourcematrix.nnznum;i++) printf("%d %f\n",sourcematrix.colidx[i],sourcematrix.val[i]);
    // printf("\n");
    int *csrSplitter = (int *)malloc((partnum + 1) * sizeof(int));
    int stridennz = ceil((double)sourcematrix.nnznum / (double)partnum);

    for (int tid = 0; tid <= partnum; tid++)
    {
        int boundary = tid * stridennz;
        boundary = boundary > sourcematrix.nnznum ? sourcematrix.nnznum : boundary;
        csrSplitter[tid] = binary_search_right_boundary_kernel(sourcematrix.rowptr, boundary, sourcematrix.rownum + 1) - 1;
        // printf("tid:%d, csrSplitter=%d\n",tid,csrSplitter[tid]);
    }

    int nnz_sum = 0;
    for (int i = 0; i < partnum; i++)
    {
        int tmp_idx = 0;
        if (i > 0)
            mtx[i].rownum = csrSplitter[i + 1] - csrSplitter[i];
        else
            mtx[i].rownum = csrSplitter[i + 1];
        mtx[i].colnum = sourcematrix.colnum;
        mtx[i].nnznum = sourcematrix.rowptr[csrSplitter[i + 1]] - sourcematrix.rowptr[csrSplitter[i]];

        mtx[i].rowptr = (int *)malloc(sizeof(int) * (mtx[i].rownum + 1));
        mtx[i].colidx = (int *)malloc(sizeof(int) * mtx[i].nnznum);
        mtx[i].val = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * mtx[i].nnznum);

        // assign rowptr
        for (int j = 0; j <= mtx[i].rownum; j++)
        {
            if (i == 0)
                mtx[i].rowptr[j] = sourcematrix.rowptr[j];
            else
                mtx[i].rowptr[j] = sourcematrix.rowptr[j + csrSplitter[i]];
        }

        // assign colidx and val
        for (int j = 0; j < mtx[i].nnznum; j++)
        {
            mtx[i].colidx[tmp_idx] = sourcematrix.colidx[j + nnz_sum];
            mtx[i].val[tmp_idx] = sourcematrix.val[j + nnz_sum];
            tmp_idx++;
        }
        nnz_sum += mtx[i].nnznum;
        if (i != 0)
        {
            tmp_idx = mtx[i].rowptr[0];
            for (int j = 0; j <= mtx[i].rownum; j++)
                mtx[i].rowptr[j] -= tmp_idx;
        }
        // printf("id:%d, rownum:%d, colnum:%d, nnznum:%d\n",i,mtx[i].rownum,mtx[i].colnum,mtx[i].nnznum);
    }

    free(csrSplitter);
}

void dividevector(VALUE_TYPE *sourcevector, VALUE_TYPE **vector, int partnum, int length, int *left_bound,int *right_bound)
{
    for(int i=0;i<partnum;i++) {
        for(int j=left_bound[i];j<=right_bound[i];j++)
            vector[i][j]=sourcevector[j];
    }
}

int binary_search_right_boundary_kernel(const int *row_pointer, const int key_input, const int size)
{
    int start = 0;
    int stop = size - 1;
    int median;
    int key_median;
    while (stop >= start)
    {
        median = (stop + start) / 2;
        key_median = row_pointer[median];
        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }
    return start;
}

void free_matrix(struct MATRIX mtx)
{
    free(mtx.rowptr);
    free(mtx.colidx);
    free(mtx.val);
}

void writeresults(char *filename_res, char *filename, int m, int n, int nnzR, double rec_time, double spmv_time, double time, double GFlops, int processors, int nthreads)
{
    FILE *fres = fopen(filename_res, "a");
    if (fres == NULL) printf("Writing results fails.\n");
    fprintf(fres, "%s, %i, %i, %d, %lf ms, %lf ms, %lf ms, %lf GFlops, %i, %i\n", filename, m, n, nnzR, rec_time, spmv_time, time, GFlops, processors, nthreads);
    fclose(fres);
}