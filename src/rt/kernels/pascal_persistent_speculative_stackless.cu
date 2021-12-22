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
    g_config.bvhLayout = BVHLayout_Stackless2;
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
    int     leafAddr;
    int     hitIndex;  

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
            leafAddr = 0;   // No postponed leaf.
            nodeAddr = 0;   // Start from the root.
            lastNodeAddr = EntrypointSentinel;
            triAddr  = 0;   // No pending triangle list.
            triAddr2 = 0;
            STORE_RESULT(rayidx, -1, 0.0f); // No triangle intersected so far.
            hitT     = d.w; // tmax
            hitIndex = -1;

        }

        // Traversal loop.


        while(nodeAddr != EntrypointSentinel){

            // we are an internal node
            // fetch node data

            int current_child;
            bool searchingLeaf = true;

            while(nodeAddr >= 0 && nodeAddr != EntrypointSentinel){

            const float4 cnodes=FETCH_TEXTURE(nodesA, nodeAddr*4+3, float4); // (c0, c1, p, dim)
            int nearChild = __float_as_int(cnodes.x);
            int farChild = __float_as_int(cnodes.y);
            int parent = __float_as_int(cnodes.z);
            int nch_idx = 0;
            int fch_idx = 1;

            // get near and far child
            const int dim = __float_as_int(cnodes.w);
            #ifdef DEBUG
            if(dim < 0 || dim > 2){
                printf("Wrong dimension!!!!");
                return;
            }
            #endif

            

            //const float ray_dim = ((float*)&raydir)[dim];
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


            // TODO: index based calculation
            //float sign = copysignf (1.0f, ray_dim);

            if(ray_dim < 0.0f){
                swap(nearChild, farChild);
                swap(nch_idx, fch_idx);
            }

            if(lastNodeAddr == farChild){

                lastNodeAddr = nodeAddr;
                nodeAddr = parent;
                continue;
            }

            // if we come from parent -> nearChild, if we come from sibling -> farChild
            current_child = (lastNodeAddr == parent) ? nearChild : farChild;

            // 0 if currentChild is c0, 1 if currentchild is c1
            const int current_child_idx = (lastNodeAddr == parent) ? nch_idx : fch_idx;


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
                //const float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin);
                //const float c0max = spanEndKepler  (c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT);
                float c0min = max4(fminf(c0lox, c0hix), fminf(c0loy, c0hiy), fminf(c0loz, c0hiz), tmin);
                float c0max = min4(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy), fmaxf(c0loz, c0hiz), hitT);
                const int traverseCurrentChild = (c0max >= c0min);
            
            
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
                
                //continue;
            }

            if (nodeAddr < 0 && leafAddr >= 0)
            {
                searchingLeaf = false;
                leafAddr = nodeAddr;
                nodeAddr = lastNodeAddr;
                lastNodeAddr = current_child;
            }

            // All SIMD lanes have found a leaf => process them.

            if(!__any((int)searchingLeaf))
                    break;



            
            } // inner traversal loop
            

// Triangle Intersection Start =============================================================================================================================
            // Intersect the ray against each triangle using Sven Woop's algorithm.

            while (leafAddr < 0)
            {
                // Fetch the start and end of the triangle list.

#ifdef NODES_ARRAY_OF_STRUCTURES
                float4 leaf=FETCH_TEXTURE(nodesA, (-leafAddr-1)*4+3, float4);
#else
                float4 leaf=FETCH_TEXTURE(nodesD, (-nodeAddr-1), float4);
#endif
                int triAddr  = __float_as_int(leaf.x);              // stored as int
                int triAddr2 = __float_as_int(leaf.y);              // stored as int

                // Intersect the ray against each triangle using Sven Woop's algorithm.

                for(; triAddr < triAddr2; triAddr++)
                {
                    // Compute and check intersection t-value.

#ifdef TRIANGLES_ARRAY_OF_STRUCTURES
                    float4 v00 = FETCH_GLOBAL(trisA, triAddr*4+0, float4);
                    float4 v11 = FETCH_GLOBAL(trisA, triAddr*4+1, float4);
#else
                    float4 v00 = FETCH_GLOBAL(trisA, triAddr, float4);
                    float4 v11 = FETCH_GLOBAL(trisB, triAddr, float4);
#endif
                    float dirx = 1.0f / idirx;
                    float diry = 1.0f / idiry;
                    float dirz = 1.0f / idirz;

                    float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
                    float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
                    float t = Oz * invDz;

                    if (t > tmin && t < hitT)
                    {
                        // Compute and check barycentric u.

                        float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;
                        float Dx = dirx*v11.x + diry*v11.y + dirz*v11.z;
                        float u = Ox + t*Dx;

                        if (u >= 0.0f)
                        {
                            // Compute and check barycentric v.

#ifdef TRIANGLES_ARRAY_OF_STRUCTURES
                            float4 v22 = FETCH_GLOBAL(trisA, triAddr*4+2, float4);
#else
                            float4 v22 = FETCH_GLOBAL(trisC, triAddr, float4);
#endif
                            float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
                            float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
                            float v = Oy + t*Dy;

                            if (v >= 0.0f && u + v <= 1.0f)
                            {
                                // Record intersection.
                                // Closest intersection not required => terminate.

                                hitT = t;
                                hitIndex = triAddr;
                                if (anyHit)
                                {
                                    nodeAddr = EntrypointSentinel;
                                    break;
                                }
                            }
                        }
                    }
                } // triangle

                // Another leaf was postponed => process it as well.

                leafAddr = nodeAddr;
                if(nodeAddr<0)
                {
                    nodeAddr = lastNodeAddr;
                    lastNodeAddr = current_child;
                }
            } // leaf
// Triangle Intersection End =============================================================================================================================

        } // we returned to the root or found a hit

        if (hitIndex != -1)
            hitIndex = FETCH_TEXTURE(triIndices, hitIndex, int);
        STORE_RESULT(rayidx, hitIndex, hitT);

    } while(true); // persistent threads (always true)
}

//------------------------------------------------------------------------
