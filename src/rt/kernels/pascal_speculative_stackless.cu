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
    GF100-optimized variant of the "Speculative while-while"
    kernel used in:

    "Understanding the Efficiency of Ray Traversal on GPUs",
    Timo Aila and Samuli Laine,
    Proc. High-Performance Graphics 2009
*/

#include "CudaTracerKernels.hpp"

//------------------------------------------------------------------------

//------------------------------------------------------------------------

extern "C" __global__ void queryConfig(void)
{
    g_config.bvhLayout = BVHLayout_Compact_Stackless;
    g_config.blockWidth = 32; // One warp per row.
    g_config.blockHeight = 4; // 4*32 = 128 threads
}

//------------------------------------------------------------------------

TRACE_FUNC
{
    // Traversal stack in CUDA thread-local memory.

    // Live state during traversal, stored in registers.

    int     rayidx;                 // Ray index.
    float   origx, origy, origz;    // Ray origin.
    float   dirx, diry, dirz;       // Ray direction.
    float   tmin;                   // t-value from which the ray starts. Usually 0.
    float   idirx, idiry, idirz;    // 1 / dir
    float   oodx, oody, oodz;       // orig / dir

    int     leafAddr;               // First postponed leaf, non-negative if none.
    int     nodeAddr;               // Non-negative: current internal node, negative: second postponed leaf.
    int     lastNodeAddr;
    int     hitIndex;               // Triangle index of the closest intersection, -1 if none.
    float   hitT;                   // t-value of the closest intersection.

    // Initialize.
    {
        // Pick ray index.

        rayidx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
        if (rayidx >= numRays)
            return;

        // Fetch ray.

        float4 o = rays[rayidx * 2 + 0];
        float4 d = rays[rayidx * 2 + 1];
        origx = o.x, origy = o.y, origz = o.z;
        dirx = d.x, diry = d.y, dirz = d.z;
        tmin = o.w;

        float ooeps = exp2f(-80.0f); // Avoid div by zero.
        idirx = 1.0f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
        idiry = 1.0f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
        idirz = 1.0f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
        oodx = origx * idirx, oody = origy * idiry, oodz = origz * idirz;

        // Setup traversal.

        leafAddr = 0;   // No postponed leaf.
        nodeAddr = 0;   // Start from the root.
        lastNodeAddr = EntrypointSentinel;
        hitIndex = -1;  // No triangle intersected so far.
        hitT     = d.w; // tmax
    }

    // Traversal loop.

    #ifdef DEBUG
    float4* ptr = (float4*)((char*)nodesA + nodeAddr);
    float4 nods = ptr[3];
    //float4 nods=FETCH_TEXTURE(nodesA, nodeAddr+3, float4);

    if(nodeAddr == 0){
        //printf("children of root %i %i\n", __float_as_int(nods.x), __float_as_int(nods.y));
        int nearChild = __float_as_int(nods.x);
        int farChild = __float_as_int(nods.y);
        float4 nods_nc=((float4*)((char*)nodesA + nearChild))[3];
        float4 nods_fc=((float4*)((char*)nodesA + farChild))[3];
        printf("parent of children %i %i of root: %i %i\n", nearChild, farChild, __float_as_int(nods_nc.z),  __float_as_int(nods_fc.z));
    }
    #endif

    
    while (nodeAddr != EntrypointSentinel)
    {
        // Traverse internal nodes until all SIMD lanes have found a leaf.
        int current_child;
        bool searchingLeaf = true;
        while (nodeAddr >= 0 && nodeAddr != EntrypointSentinel)
        {
            float4* ptr = (float4*)((char*)nodesA + nodeAddr);
            float4 cnodes = ptr[3]; // (c0, c1, p, dim)
            int nearChild = __float_as_int(cnodes.x);
            int farChild = __float_as_int(cnodes.y);
            int parent = __float_as_int(cnodes.z);
            int nch_idx = 0;
            int fch_idx = 1;

                
                // get near and far child
                int dim = __float_as_int(cnodes.w);

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

                
                const float4 nxy = ptr[current_child_idx];  // (c0/1.lo.x, c0/1.hi.x, c0/1.lo.y, c0/1.hi.y)
                const float4 nz = ptr[2];  // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)

                const float c0lox = nxy.x * idirx - oodx;
                const float c0hix = nxy.y * idirx - oodx;
                const float c0loy = nxy.z * idiry - oody;
                const float c0hiy = nxy.w * idiry - oody;

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
                
                const float c0loz = nz_x * idirz - oodz;
                const float c0hiz = nz_y * idirz - oodz;
                const float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin);
                const float c0max = spanEndKepler  (c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT);
                //float c0min = max4(fminf(c0lox, c0hix), fminf(c0loy, c0hiy), fminf(c0loz, c0hiz), aux->tmin);
                //float c0max = min4(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy), fmaxf(c0loz, c0hiz), hitT);
                const int traverseCurrentChild = (c0max >= c0min);

                if(traverseCurrentChild){
                    // if we hit the BB -> go down a level
                    lastNodeAddr = nodeAddr;
                    nodeAddr = current_child;
                     #ifdef DEBUG1
                        printf("Found intersection: %i\n", nodeAddr);
                    #endif
               
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
                   // continue;
                }



            // First leaf => postpone and continue traversal.

            if (nodeAddr < 0 && leafAddr >= 0)
            {
                searchingLeaf = false;
                leafAddr = nodeAddr;
                nodeAddr = lastNodeAddr;
                lastNodeAddr = current_child;
            }

            // All SIMD lanes have found a leaf => process them.

            if (!__any(searchingLeaf))
                break;
        }

        // Process postponed leaf nodes.

        while (leafAddr < 0)
        {
            // Intersect the ray against each triangle using Sven Woop's algorithm.

            for (int triAddr = ~leafAddr;; triAddr += 3)
            {
                // Read first 16 bytes of the triangle.
                // End marker (negative zero) => all triangles processed.

                float4 v00 = tex1Dfetch(t_trisA, triAddr + 0);
                if (__float_as_int(v00.x) == 0x80000000)
                    break;

                // Compute and check intersection t-value.

                float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
                float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
                float t = Oz * invDz;

                if (t > tmin && t < hitT)
                {
                    // Compute and check barycentric u.

                    float4 v11 = tex1Dfetch(t_trisA, triAddr + 1);
                    float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;
                    float Dx = dirx*v11.x + diry*v11.y + dirz*v11.z;
                    float u = Ox + t*Dx;

                    if (u >= 0.0f && u <= 1.0f)
                    {
                        // Compute and check barycentric v.

                        float4 v22 = tex1Dfetch(t_trisA, triAddr + 2);
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
    } // traversal



    // Remap intersected triangle index, and store the result.

    if (hitIndex != -1)
        hitIndex = tex1Dfetch(t_triIndices, hitIndex);
    STORE_RESULT(rayidx, hitIndex, hitT);
}

//------------------------------------------------------------------------
