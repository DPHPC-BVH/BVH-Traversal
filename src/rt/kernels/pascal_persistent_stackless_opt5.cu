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

#define LOAD_BALANCER_BATCH_SIZE        32  // Number of rays to fetch at a time. Must be a multiple of 32.
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


    // Live state during traversal, stored in registers.

    float   origx, origy, origz;    // Ray origin.
    int     lastNodeAddr;           // current parent
    int     nodeAddr;               // Current node

    int     triAddr;                // Start of a pending triangle list.
    int     triAddr2;               // End of a pending triangle list.
    float   hitT;                   // t-value of the closest intersection.
    int hitIndex;

    float   tmin;
    int     rayidx;
    float   oodx;
    float   oody;
    float   oodz;
    //float3 raydir; 
    float   dirx;
    float   diry;
    float   dirz;
    float   idirx;
    float   idiry;
    float   idirz;


    // Initialize persistent threads.

    __shared__ volatile int nextRayArray[MaxBlockHeight]; // Current ray index in global buffer.
    __shared__ volatile int rayCountArray[MaxBlockHeight]; // Number of rays in the local pool.
    nextRayArray[threadIdx.y] = 0;
    rayCountArray[threadIdx.y] = 0;

    // Persistent threads: fetch and process rays in a loop.

    do
    {
        int tidx = threadIdx.x; 
        int widx = threadIdx.y;
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
            rayidx = localPoolNextRay + tidx;
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
            tmin = o.w;

            dirx  = d.x;
            diry  = d.y;
            dirz  = d.z;

            float ooeps = exp2f(-80.0f); // Avoid div by zero.
            idirx = 1.0f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
            idiry = 1.0f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
            idirz = 1.0f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
            oodx  = origx * idirx;
            oody  = origy * idiry;
            oodz  = origz * idirz;

            // Setup traversal.

            //traversalStack[0] = EntrypointSentinel; // Bottom-most entry.
            nodeAddr = 0;   // Start from the root.
            lastNodeAddr = -1;
            triAddr  = 0;   // No pending triangle list.
            triAddr2 = 0;
            STORE_RESULT(rayidx, -1, 0.0f); // No triangle intersected so far.
            hitT     = d.w; // tmax
            hitIndex = -1;

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

            const float4 cnodes=FETCH_TEXTURE(nodesA, nodeAddr*4+3, float4); // (c0, c1, p, dim)
            int nearChild = __float_as_int(cnodes.x);
            int farChild = __float_as_int(cnodes.y);
            int parent = __float_as_int(cnodes.z);
            //int nch_idx = 0;
            //int fch_idx = 1;

            // get near and far child
            //const int dim = __float_as_int(cnodes.w);
            #ifdef DEBUG
            if(dim < 0 || dim > 2){
                printf("Wrong dimension!!!!");
                return;
            }
            #endif

            

            //const float ray_dim = ((float*)&raydir)[dim];
            /*
            float ray_dim = 0.0f;
            switch(dim){
                case 0: 
                    ray_dim = idirx;
                    break;
                case 1:
                    ray_dim = idiry;
                    break;
                case 2:
                    ray_dim = idirz;
                    break;
            }
            */

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

            /*
            if(ray_dim < 0.0f){
                swap(nearChild, farChild);
                swap(nch_idx, fch_idx);
            }
            */

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
            const int current_child = (lastNodeAddr == parent) ? nearChild : farChild;

            // 0 if currentChild is c0, 1 if currentchild is c1
            const int current_child_idx = (lastNodeAddr == parent) ? 0 : 1;

            #ifdef DEBUG

            /*if(nodeAddr == 0 && lastNodeAddr == -1 && current_child != nearChild && first_entry == 1){
                printf("Root nearchild fail! parent: %i, lasNodeAddr: %i\n",parent, lastNodeAddr);
            }*/
            
            first_entry = 0;
            #endif

            // fetch additional node data: 
            const float4 nxy = FETCH_TEXTURE(nodesA, nodeAddr*4+current_child_idx, float4);  // (c0/1.lo.x, c0/1.hi.x, c0/1.lo.y, c0/1.hi.y)
            const float4 nz   = FETCH_TEXTURE(nodesA, nodeAddr*4+2, float4);  // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
            

            // Intersect the ray against the current Child node
                
                const float c0lox = nxy.x * idirx - oodx;
                const float c0hix = nxy.y * idirx - oodx;
                const float c0loy = nxy.z * idiry - oody;
                const float c0hiy = nxy.w * idiry - oody;

                float nz_x;
                float nz_y;

                if(current_child_idx == 0){
                    nz_x = nz.x;
                    nz_y = nz.y;
                }else{
                    nz_x = nz.z;
                    nz_y = nz.w;
                }
                
                const float c0loz = nz_x * idirz - oodz;
                const float c0hiz = nz_y * idirz - oodz;
                
                const float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin);
                const float c0max = spanEndKepler  (c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT);
                //float c0min = max4(fminf(c0lox, c0hix), fminf(c0loy, c0hiy), fminf(c0loz, c0hiz), aux->tmin);
                //float c0max = min4(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy), fmaxf(c0loz, c0hiz), hitT);
                const int traverseCurrentChild = (c0max >= c0min);

            //const int isNearChild = current_child == nearChild;
            //lastNodeAddr = traverseCurrentChild*nodeAddr + (1-traverseCurrentChild)*(isNearChild*nearChild +(1-isNearChild)*nodeAddr);
            //nodeAddr = traverseCurrentChild*current_child + (1-traverseCurrentChild)*((1-isNearChild)*parent + isNearChild*nodeAddr);
            
            
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
                const float4 leaf=FETCH_TEXTURE(nodesA, (-nodeAddr-1)*4+3, float4);
                
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

                const float4 v00 = FETCH_GLOBAL(trisA, triAddr*4+0, float4);
                const float4 v11 = FETCH_GLOBAL(trisA, triAddr*4+1, float4);
                const float4 v22 = FETCH_GLOBAL(trisA, triAddr*4+2, float4);

                //const float dirx  = 1.0f / idirx;
                //const float diry  = 1.0f / idiry;
                //const float dirz  = 1.0f / idirz;

                const float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
                const float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
                float t = Oz * invDz;

                if (t > tmin && t < hitT)
                {
                    // Compute and check barycentric u.

                    const float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;
                    const float Dx = dirx*v11.x + diry*v11.y + dirz*v11.z;
                    const float u = Ox + t*Dx;

                    if (u >= 0.0f)
                    {
                        // Compute and check barycentric v.
                    
                        const float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
                        const float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
                        const float v = Oy + t*Dy;

                        if (v >= 0.0f && u + v <= 1.0f)
                        {
                            // Record intersection.
                            // Closest intersection not required => terminate.

                            hitT = t;
                            hitIndex = triAddr;

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


        }while(nodeAddr >= 0); // we returned to the root

        if (hitIndex == -1) { STORE_RESULT(rayidx, -1, hitT); }
        else                { STORE_RESULT(rayidx, FETCH_TEXTURE(triIndices, hitIndex, int), hitT); }

        /*
        #ifdef DEBUG
        if(found_tri == 0){
            printf("no Triangle intersection found!!!!\n");
        }
        #endif
        */
    } while(true); // persistent threads (always true)
}

//------------------------------------------------------------------------
