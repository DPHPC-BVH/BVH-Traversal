/*
 *  Copyright (c) 2009-2011, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
    "Persistent while-while kernel" used in:

    "Understanding the Efficiency of Ray Traversal on GPUs",
    Timo Aila and Samuli Laine,
    Proc. High-Performance Graphics 2009
*/

#include "CudaTracerKernels.hpp"

//------------------------------------------------------------------------

#define NODES_ARRAY_OF_STRUCTURES           // Define for AOS, comment out for SOA.
#define TRIANGLES_ARRAY_OF_STRUCTURES       // Define for AOS, comment out for SOA.

#define LOAD_BALANCER_BATCH_SIZE        96  // Number of rays to fetch at a time. Must be a multiple of 32.
#define LOOP_NODE                       100 // Nodes: 1 = if, 100 = while.
#define LOOP_TRI                        100 // Triangles: 1 = if, 100 = while.

//#define DEBUG 1

extern "C" __device__ int g_warpCounter;    // Work counter for persistent threads.

//------------------------------------------------------------------------

extern "C" __global__ void queryConfig(void)
{
    g_config.bvhLayout = BVHLayout_Stackless;
    g_config.blockWidth = 32; // One warp per row.
    g_config.blockHeight = 4; // GTX1080 -> 8*128 threads?
    g_config.usePersistentThreads = 1;
}

//------------------------------------------------------------------------

TRACE_FUNC
{
    // Temporary data stored in shared memory to reduce register pressure.

    __shared__ RayStruct shared[32 * MaxBlockHeight + 1];
    RayStruct* aux = shared + threadIdx.x + (blockDim.x * threadIdx.y);

    // Traversal stack in CUDA thread-local memory.
    // Allocate 3 additional entries for spilling rarely used variables.

    int traversalIds[3];
    traversalIds[0] = threadIdx.x; // Forced to local mem => saves a register.
    traversalIds[1] = threadIdx.y;

    float4 childBbox[3];

    // Live state during traversal, stored in registers.

    float   origx, origy, origz;    // Ray origin.
    int     lastNodeAddr;           // current parent
    int     nodeAddr;               // Current node

    int     triAddr;                // Start of a pending triangle list.
    int     triAddr2;               // End of a pending triangle list.
    float   hitT;                   // t-value of the closest intersection.


    // Initialize persistent threads.

    __shared__ volatile int nextRayArray[MaxBlockHeight]; // Current ray index in global buffer.
    __shared__ volatile int rayCountArray[MaxBlockHeight]; // Number of rays in the local pool.
    nextRayArray[threadIdx.y] = 0;
    rayCountArray[threadIdx.y] = 0;

    // Persistent threads: fetch and process rays in a loop.

    do
    {
        int tidx = traversalIds[0]; // threadIdx.x
        int widx = traversalIds[1]; // threadIdx.y
        volatile int& localPoolRayCount = rayCountArray[widx];
        volatile int& localPoolNextRay = nextRayArray[widx];

        // Local pool is empty => fetch new rays from the global pool using lane 0.

        if (tidx == 0 && localPoolRayCount <= 0)
        {
            localPoolNextRay = atomicAdd(&g_warpCounter, LOAD_BALANCER_BATCH_SIZE);
            localPoolRayCount = LOAD_BALANCER_BATCH_SIZE;
        }

        // Pick 32 rays from the local pool.
        // Out of work => done.
        {
            int rayidx = localPoolNextRay + tidx;
            if (rayidx >= numRays)
                break;

            if (tidx == 0)
            {
                localPoolNextRay += 32;
                localPoolRayCount -= 32;
            }

            // Fetch ray.

            float4 o = FETCH_GLOBAL(rays, rayidx * 2 + 0, float4);
            float4 d = FETCH_GLOBAL(rays, rayidx * 2 + 1, float4);
            origx = o.x, origy = o.y, origz = o.z;
            aux->tmin = o.w;

            float ooeps = exp2f(-80.0f); // Avoid div by zero.
            aux->idirx = 1.0f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
            aux->idiry = 1.0f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
            aux->idirz = 1.0f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
            traversalIds[2] = rayidx; // Spill.

            // Setup traversal.

            //traversalStack[0] = EntrypointSentinel; // Bottom-most entry.
            nodeAddr = 0;   // Start from the root.
            lastNodeAddr = -1;
            triAddr  = 0;   // No pending triangle list.
            triAddr2 = 0;
            STORE_RESULT(rayidx, -1, 0.0f); // No triangle intersected so far.
            hitT     = d.w; // tmax

            #ifdef DEBUG
            // check root data
            float4 cnodes=FETCH_TEXTURE(nodesA, 3, float4); // (c0, c1, p, dim)
            int r_parent = __float_as_int(cnodes.z);
            if(r_parent != -1){
                printf("wrong root parent: %i", r_parent);
            }
            #endif

        }

        // Traversal loop.
        do{
            
            #ifdef DEBUG
                if(nodeAddr == 0 && lastNodeAddr == 0){
                    printf("Invalid node state 1!");
                    return;
                }
                if(nodeAddr == lastNodeAddr){
                    printf("Invalid node state 2!");
                    return;
                }
            #endif 
            
            
            // we are an internal node
            // fetch node data

            float4 cnodes=FETCH_TEXTURE(nodesA, nodeAddr*4+3, float4); // (c0, c1, p, dim)
            int nearChild = __float_as_int(cnodes.x);
            int farChild = __float_as_int(cnodes.y);
            int parent = __float_as_int(cnodes.z);
            int nch_idx = 0;
            int fch_idx = 1;

            // get near and far child
            int dim = __float_as_int(cnodes.w);
            #ifdef DEBUG
            if(dim < 0 || dim > 2){
                printf("Wrong dimension!!!!");
                return;
            }
            #endif

            

            float ray_dim = ((float*)aux)[dim];

            #ifdef DEBUG
            float ray_dim_2 = 0.0f;
            switch(dim){
                case 0: 
                    ray_dim_2 = aux->idirx;
                    break;
                case 1:
                    ray_dim_2 = aux->idiry;
                    break;
                case 2:
                    ray_dim_2 = aux->idirz;
                    break;
            }
            if(ray_dim != ray_dim_2){
                printf("Wrong parith!!");
                return;
            }
            #endif

            // TODO: index based calculation
            //float sign = copysignf (1.0f, ray_dim);

            if(ray_dim < 0.0f){
                swap(nearChild, farChild);
                swap(nch_idx, fch_idx);
            }

            #ifdef DEBUG
            if(lastNodeAddr != parent && lastNodeAddr != nearChild && lastNodeAddr != farChild && lastNodeAddr != -1){
                printf("Invalid state 3!!\n");
            }
            /*if(lastNodeAddr == nearChild && nodeAddr == 0){
                 printf("We returned to root from nearChild: %i\n", nearChild);
            }*/
            /*if(nodeAddr == 1 || nodeAddr == 2){
                printf("Parent should be root: %i", parent);
            }*/

            #endif


            

            if(lastNodeAddr == farChild){

                #ifdef DEBUG
                /*if(nodeAddr == 1){
                    printf("We returned from farchild of 1; nextConf: nodeAddr->%i lastNodeAddr->%i\n", parent, nodeAddr);
                }*/
                #endif

                lastNodeAddr = nodeAddr;
                nodeAddr = parent;


                continue;
            }

            // if we come from parent -> nearChild, if we come from sibling -> farChild
            int current_child = (lastNodeAddr == parent) ? nearChild : farChild;

            // 0 if currentChild is c0, 1 if currentchild is c1
            int current_child_idx = (lastNodeAddr == parent) ? nch_idx : fch_idx;


            #ifdef DEBUG

            /*if(nodeAddr == 0 && lastNodeAddr == -1 && current_child != nearChild && first_entry == 1){
                printf("Root nearchild fail! parent: %i, lasNodeAddr: %i\n",parent, lastNodeAddr);
            }*/
            
            first_entry = 0;
            #endif

            // fetch additional node data: 
            float4 nxy = FETCH_TEXTURE(nodesA, nodeAddr*4+current_child_idx, float4);  // (c0/1.lo.x, c0/1.hi.x, c0/1.lo.y, c0/1.hi.y)
            float4 nz   = FETCH_TEXTURE(nodesA, nodeAddr*4+2, float4);  // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
            

            // Intersect the ray against the current Child node
                
                float oodx  = origx * aux->idirx;
                float oody  = origy * aux->idiry;
                float oodz  = origz * aux->idirz;
                float c0lox = nxy.x * aux->idirx - oodx;
                float c0hix = nxy.y * aux->idirx - oodx;
                float c0loy = nxy.z * aux->idiry - oody;
                float c0hiy = nxy.w * aux->idiry - oody;

                float nz_x;
                float nz_y;

                // TODO: replace branch with pointer arith.
                if(current_child_idx == 0){
                    nz_x = nz.x;
                    nz_y = nz.y;
                }else{
                    nz_x = nz.z;
                    nz_y = nz.w;
                }
                
                float c0loz = nz_x * aux->idirz - oodz;
                float c0hiz = nz_y * aux->idirz - oodz;
                float c0min = max4(fminf(c0lox, c0hix), fminf(c0loy, c0hiy), fminf(c0loz, c0hiz), aux->tmin);
                float c0max = min4(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy), fmaxf(c0loz, c0hiz), hitT);
                bool traverseCurrentChild = (c0max >= c0min);


            if(traverseCurrentChild){
                // if we hit the BB -> go down a level
                lastNodeAddr = nodeAddr;
                nodeAddr = current_child;
               
            }else{
                // otherwise:
                // if we are nearChild  -> go to far child
                // else                 -> go up a level
                if(current_child == nearChild){
                    lastNodeAddr = nearChild;
                }else{
                    lastNodeAddr = nodeAddr;
                    nodeAddr = parent;
                }
                continue;
            }


            // if the child we want to go to is a leaf: do triangle instersection - and skip
             if (nodeAddr < 0 && triAddr >= triAddr2){
                float4 leaf=FETCH_TEXTURE(nodesA, (-nodeAddr-1)*4+3, float4);
                
                // skip child
                // goto next node: either sibling or parent
                nodeAddr = lastNodeAddr;
                lastNodeAddr = current_child;
                
                triAddr  = __float_as_int(leaf.x); // stored as int
                triAddr2 = __float_as_int(leaf.y); // stored as int

            }

// Triangle Intersection Start =============================================================================================================================
            // Intersect the ray against each triangle using Sven Woop's algorithm.

            for (int i = LOOP_TRI - 1; i >= 0 && triAddr < triAddr2; triAddr++, i--)
            {
                // Compute and check intersection t-value.

                float4 v00 = FETCH_GLOBAL(trisA, triAddr*4+0, float4);
                float4 v11 = FETCH_GLOBAL(trisA, triAddr*4+1, float4);

                float dirx  = 1.0f / aux->idirx;
                float diry  = 1.0f / aux->idiry;
                float dirz  = 1.0f / aux->idirz;

                float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
                float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
                float t = Oz * invDz;

                if (t > aux->tmin && t < hitT)
                {
                    // Compute and check barycentric u.

                    float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;
                    float Dx = dirx*v11.x + diry*v11.y + dirz*v11.z;
                    float u = Ox + t*Dx;

                    if (u >= 0.0f)
                    {
                        // Compute and check barycentric v.
                        float4 v22 = FETCH_GLOBAL(trisA, triAddr*4+2, float4);

                        float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
                        float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
                        float v = Oy + t*Dy;

                        if (v >= 0.0f && u + v <= 1.0f)
                        {
                            // Record intersection.
                            // Closest intersection not required => terminate.

                            hitT = t;
                            STORE_RESULT(traversalIds[2], FETCH_GLOBAL(triIndices, triAddr, int), t);

                            if (anyHit)
                            {
                                // set to root
                                nodeAddr = -1;
                                triAddr = triAddr2; // Breaks the do-while.
                                break;
                            }
                        }
                    }
                }
            } // triangle
// Triangle Intersection End =============================================================================================================================

trace_loop_end:
            ;


        }while(nodeAddr >= 0 || triAddr < triAddr2); // we returned to the root
        /*
        #ifdef DEBUG
        if(found_tri == 0){
            printf("no Triangle intersection found!!!!\n");
        }
        #endif
        */
    } while(aux); // persistent threads (always true)
}

//------------------------------------------------------------------------
