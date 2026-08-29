/* Probe the real fast/slow arena requirement of a plan. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "tigris.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "tigris_executor.h"
#include "tigris_kernels_s8.h"

static uint8_t *load_file(const char *p, uint32_t *n){
    FILE *f=fopen(p,"rb"); if(!f)return NULL; fseek(f,0,SEEK_END); long l=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t *b=malloc(l); if(fread(b,1,l,f)!=(size_t)l){free(b);fclose(f);return NULL;} fclose(f); *n=(uint32_t)l; return b;
}
int main(int argc,char**argv){
    if(argc<4){fprintf(stderr,"usage: %s plan.tgrs fast_bytes slow_bytes\n",argv[0]);return 2;}
    uint32_t fsz; uint8_t*fb=load_file(argv[1],&fsz); if(!fb){fprintf(stderr,"load fail\n");return 2;}
    tigris_plan_t plan;
    if(tigris_plan_load(fb,fsz,&plan)!=TIGRIS_OK){fprintf(stderr,"plan load fail\n");return 2;}
    uint32_t budget=(uint32_t)atol(argv[2]);
    uint32_t slow_size=(uint32_t)atol(argv[3]);
    uint32_t fast_size=budget+tigris_weight_decompression_overhead(&plan);
    uint16_t nt=plan.header->num_tensors;
    void*fast=malloc(fast_size), *slow=malloc(slow_size); void**ptrs=malloc(nt*sizeof(void*));
    tigris_mem_t mem;
    if(tigris_mem_init(&mem,ptrs,nt,fast,fast_size,slow,slow_size)!=TIGRIS_MEM_OK){printf("MEM_INIT_FAIL\n");return 1;}
    for(uint8_t i=0;i<plan.header->num_model_inputs;i++){
        uint16_t t=plan.model_inputs[i]; uint32_t sz=plan.tensors[t].size_bytes;
        tigris_mem_alloc_slow(&mem,t,sz); int8_t*d=ptrs[t];
        for(uint32_t j=0;j<sz;j++) d[j]=(int8_t)(((j*73u+17u)%201u)-100u);
    }
    size_t wss=tigris_executor_workspace_required(&plan); void*ws=malloc(wss);
    tigris_exec_stats_t st;
    tigris_exec_error_t e=tigris_run_with_workspace_buffer(&plan,&mem,tigris_dispatch_kernel_s8,NULL,&st,ws,wss);
    printf("fast_budget=%u slow=%u -> %s  fast_peak=%u slow_used=%u total=%u  (wss=%zu)\n",
           budget, slow_size, e==TIGRIS_EXEC_OK?"OK":tigris_exec_error_str(e),
           mem.fast_peak, mem.slow_used, mem.fast_peak+mem.slow_used, wss);
    return e==TIGRIS_EXEC_OK?0:1;
}
