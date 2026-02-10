// Input heating box initialization for exotic energy injection

#include "OutputStructs.h"
#include "InputParameters.h"
#include "indexing.h"
#include "debugging.h"

int InitInputHeating(InputHeating *box) {
    int status;
    Broadcast_struct_global_noastro();

    int i, j, k;

    omp_set_num_threads(global_params.N_THREADS);

#pragma omp parallel private(i,j,k) num_threads(global_params.N_THREADS)
    {
#pragma omp for
        for (i=0; i<HII_DIM; i++){
            for (j=0; j<HII_DIM; j++){
                for (k=0; k<HII_DIM; k++){
                    box->input_heating[HII_R_INDEX(i,j,k)] = 0.;
                }
            }
        }
    }

    return 0;
}
