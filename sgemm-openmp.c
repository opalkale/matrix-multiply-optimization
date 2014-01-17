#include <nmmintrin.h>
#include <omp.h>
 
void sgemm( int m, int n, int d, float *A, float *C )
{
    const int STRIDE = 40;          
    const int blocksize = 120;       /*40, 40*/
    
    
    
    #pragma omp parallel
    {
        register __m128 cmat0;
        register __m128 cmat1;
        register __m128 cmat2;
        register __m128 cmat3;
        register __m128 cmat4;
        
        register __m128 cmat5;
        register __m128 cmat6;
        register __m128 cmat7;
        register __m128 cmat8;
        register __m128 cmat9;
        __m128 amat0, amat1;
        float* sum1;
        float* sum2;
        #pragma omp for
        /*for(int j1 = 0; j1 < n; j1 += blocksize) {*/
        for( int j = 0; j < n; j++ ) {
            for(int i1 = 0; i1 < n; i1 += blocksize) {
                for(int k1 = 0; k1 < m; k1 += blocksize) {
                    
                    
                    for( int i = i1; i < i1+blocksize && i < n/STRIDE*STRIDE; i+= STRIDE ) {
                        sum1 = C+i+j*n;
                        cmat0 = _mm_loadu_ps(sum1);
                        cmat1 = _mm_loadu_ps(sum1+4);
                        cmat2 = _mm_loadu_ps(sum1+8);
                        cmat3 = _mm_loadu_ps(sum1+12);
                        cmat4 = _mm_loadu_ps(sum1+16);
                        cmat5 = _mm_loadu_ps(sum1+20);
                        cmat6 = _mm_loadu_ps(sum1+24);
                        cmat7 = _mm_loadu_ps(sum1+28);
                        cmat8 = _mm_loadu_ps(sum1+32);
                        cmat9 = _mm_loadu_ps(sum1+36);
                        for( int k = k1; k < k1+blocksize && k < m; k++ ) {
                        
                            amat1 = _mm_load1_ps(A+j*(n+1)+k*n);
                                
                            sum2 = A+i+k*n;
                            
                            amat0 = _mm_loadu_ps(sum2);
                            amat0 = _mm_mul_ps(amat0, amat1);
                            cmat0 = _mm_add_ps(cmat0, amat0);
                            
                            amat0 = _mm_loadu_ps(sum2+4);
                            amat0 = _mm_mul_ps(amat0, amat1);
                            cmat1 = _mm_add_ps(cmat1, amat0);
                            
                            amat0 = _mm_loadu_ps(sum2+8);
                            amat0 = _mm_mul_ps(amat0, amat1);
                            cmat2 = _mm_add_ps(cmat2, amat0);
                            
                            amat0 = _mm_loadu_ps(sum2+12);
                            amat0 = _mm_mul_ps(amat0, amat1);
                            cmat3 = _mm_add_ps(cmat3, amat0);
                            
                            amat0 = _mm_loadu_ps(sum2+16);
                            amat0 = _mm_mul_ps(amat0, amat1);
                            cmat4 = _mm_add_ps(cmat4, amat0);
                            
                            amat0 = _mm_loadu_ps(sum2+20);
                            amat0 = _mm_mul_ps(amat0, amat1);
                            cmat5 = _mm_add_ps(cmat5, amat0);
                            
                            amat0 = _mm_loadu_ps(sum2+24);
                            amat0 = _mm_mul_ps(amat0, amat1);
                            cmat6 = _mm_add_ps(cmat6, amat0);
                            
                            amat0 = _mm_loadu_ps(sum2+28);
                            amat0 = _mm_mul_ps(amat0, amat1);
                            cmat7 = _mm_add_ps(cmat7, amat0);
                            
                            amat0 = _mm_loadu_ps(sum2+32);
                            amat0 = _mm_mul_ps(amat0, amat1);
                            cmat8 = _mm_add_ps(cmat8, amat0);
                            
                            amat0 = _mm_loadu_ps(sum2+36);
                            amat0 = _mm_mul_ps(amat0, amat1);
                            cmat9 = _mm_add_ps(cmat9, amat0);
                            
                            
                        }
                        _mm_storeu_ps(sum1, cmat0);
                        _mm_storeu_ps(sum1+4, cmat1);
                        _mm_storeu_ps(sum1+8, cmat2);
                        _mm_storeu_ps(sum1+12, cmat3);
                        _mm_storeu_ps(sum1+16, cmat4);
                        
                        _mm_storeu_ps(sum1+20, cmat5);
                        _mm_storeu_ps(sum1+24, cmat6);
                        _mm_storeu_ps(sum1+28, cmat7);
                        _mm_storeu_ps(sum1+32, cmat8);
                        _mm_storeu_ps(sum1+36, cmat9);
                        
                    }
                        

                        
                    
                }
            }
                
                
                
        
            
            
            for(int k1 = 0; k1 < m; k1 += blocksize) {
            
                for ( int i = n/STRIDE*STRIDE; i < n/4*4; i+=4) {
                    cmat0 = _mm_loadu_ps(C+i+j*n);
                    for( int k = k1; k < k1+blocksize && k < m; k++ ) {
                        amat1 = _mm_load1_ps(A+j*(n+1)+k*n);
                        
                        amat0 = _mm_loadu_ps(A+i+k*n);
                        amat0 = _mm_mul_ps(amat0, amat1);
                        cmat0 = _mm_add_ps(cmat0, amat0);
                    }
                    _mm_storeu_ps(C+i+j*n, cmat0);
                }
                for ( int i = n/4*4; i < n; i++) {
                    for( int k = k1; k < k1+blocksize && k < m; k++ ) {
                        C[i+j*n] += A[i+k*n]*A[j*(n+1)+k*n];
                    }
                }
                
            }
            
            
        }
    }
} 
