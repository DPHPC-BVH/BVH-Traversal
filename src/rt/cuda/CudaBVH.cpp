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

#include "cuda/CudaBVH.hpp"

using namespace FW;

//------------------------------------------------------------------------

CudaBVH::CudaBVH(const BVH& bvh, BVHLayout layout)
:   m_layout    (layout)
{
    FW_ASSERT(layout >= 0 && layout < BVHLayout_Max);

    if (layout == BVHLayout_Compact)
    {
        createCompact(bvh,1, false);
        return;
    }

    if (layout == BVHLayout_Compact2)
    {
        createCompact(bvh,16, false);
        return;
    }

    if(layout == BVHLayout_Compact_Stackless){
        createCompact(bvh,1, true);
        return;
    }

    if(layout == BVHLayout_Compact2_Stackless){
        createCompact(bvh,16, true);
        return;
    }

        createNodeBasic(bvh);
    createTriWoopBasic(bvh);
    createTriIndexBasic(bvh);
}

//------------------------------------------------------------------------

CudaBVH::CudaBVH(InputStream& in)
{
    in >> (S32&)m_layout >> m_nodes >> m_triWoop >> m_triIndex;
}

//------------------------------------------------------------------------

CudaBVH::~CudaBVH(void)
{
}

//------------------------------------------------------------------------

void CudaBVH::serialize(OutputStream& out)
{
    out << (S32)m_layout << m_nodes << m_triWoop << m_triIndex;
}

//------------------------------------------------------------------------

Vec2i CudaBVH::getNodeSubArray(int idx) const
{
    FW_ASSERT(idx >= 0 && idx < 4);
    S32 size = (S32)m_nodes.getSize();

    if (m_layout == BVHLayout_SOA_AOS || m_layout == BVHLayout_SOA_SOA)
        return Vec2i((size >> 2) * idx, (size >> 2));
    return Vec2i(0, size);
}

//------------------------------------------------------------------------

Vec2i CudaBVH::getTriWoopSubArray(int idx) const
{
    FW_ASSERT(idx >= 0 && idx < 4);
    S32 size = (S32)m_triWoop.getSize();

    if (m_layout == BVHLayout_AOS_SOA || m_layout == BVHLayout_SOA_SOA)
        return Vec2i((size >> 2) * idx, (size >> 2));
    return Vec2i(0, size);
}

//------------------------------------------------------------------------

CudaBVH& CudaBVH::operator=(CudaBVH& other)
{
    if (&other != this)
    {
        m_layout    = other.m_layout;
        m_nodes     = other.m_nodes;
        m_triWoop   = other.m_triWoop;
        m_triIndex  = other.m_triIndex;
    }
    return *this;
}

//------------------------------------------------------------------------


S32 CudaBVH::getChildOrderDim(const AABB* b0, const AABB* b1) {
    Vec3f center0 = b0->midPoint();
    Vec3f center1 = b1->midPoint();
    Vec3f diff = (center1 - center0).abs();

    F32 max_xy = max(diff.x, diff.y);
    F32 max_dim_xy = diff.x > diff.y ? 0 : 1;
    
    return diff.z > max_xy ? 2 : max_dim_xy;
}

void CudaBVH::createNodeBasic(const BVH& bvh)
{
    struct StackEntry
    {
        const BVHNode*  node;
        S32             parent_idx;
        S32             idx;

        StackEntry(const BVHNode* n = NULL, int i = 0, int p = 0) : node(n), idx(i), parent_idx(p) {}
        int encodeIdx(void) const { return (node->isLeaf()) ? ~idx : idx; }
    };

    const BVHNode* root = bvh.getRoot();
    m_nodes.resizeDiscard((root->getSubtreeSize(BVH_STAT_NODE_COUNT) * 64 + Align - 1) & -Align);

    int nextNodeIdx = 0;

    int root_parent = -1;

    if (m_layout == BVHLayout_Stackless2) {
        root_parent = EntrypointSentinel;
        printf("\nLAyout2!!!!!!\n");
    }

    Array<StackEntry> stack(StackEntry(root, nextNodeIdx++, root_parent));
    while (stack.getSize())
    {
        StackEntry e = stack.removeLast();
        const AABB* b0;
        const AABB* b1;
        int c0;
        int c1;

        // order of c0,c1
        S32 ord = 0;

        // Leaf?

        if (e.node->isLeaf())
        {
            const LeafNode* leaf = reinterpret_cast<const LeafNode*>(e.node);
            b0 = &leaf->m_bounds;
            b1 = &leaf->m_bounds;
            // triangle buffer index range [m_lo, m_hi)
            c0 = leaf->m_lo;
            c1 = leaf->m_hi;
        }

        // Internal node?

        else
        {
            StackEntry e0 = stack.add(StackEntry(e.node->getChildNode(0), nextNodeIdx++, e.idx));
            StackEntry e1 = stack.add(StackEntry(e.node->getChildNode(1), nextNodeIdx++, e.idx));
            b0 = &e0.node->m_bounds;
            b1 = &e1.node->m_bounds;
            c0 = e0.encodeIdx();
            c1 = e1.encodeIdx();

            ord = getChildOrderDim(b0, b1);

        }

        Vec4i extra_data;

        // if stackless -> add parent index and separation axis to unused fields
        if(m_layout == BVHLayout_Stackless || m_layout == BVHLayout_Stackless2){
            int p = e.parent_idx; // no need to encode since it has to be an internal node
            extra_data = Vec4i(c0, c1, p, ord);
        }else{
            extra_data = Vec4i(c0, c1, 0, 0);
        }

        // Write entry.
        Vec4i data[] =
        {
            Vec4i(floatToBits(b0->min().x), floatToBits(b0->max().x), floatToBits(b0->min().y), floatToBits(b0->max().y)),
            Vec4i(floatToBits(b1->min().x), floatToBits(b1->max().x), floatToBits(b1->min().y), floatToBits(b1->max().y)),
            Vec4i(floatToBits(b0->min().z), floatToBits(b0->max().z), floatToBits(b1->min().z), floatToBits(b1->max().z)),
            extra_data
        };

        switch (m_layout)
        {
        case BVHLayout_AOS_AOS:
        case BVHLayout_AOS_SOA:
        case BVHLayout_Stackless:
        case BVHLayout_Stackless2:
            memcpy(m_nodes.getMutablePtr(e.idx * 64), data, 64);
            break;

        case BVHLayout_SOA_AOS:
        case BVHLayout_SOA_SOA:
            for (int i = 0; i < 4; i++)
                memcpy(m_nodes.getMutablePtr(e.idx * 16 + (m_nodes.getSize() >> 2) * i), &data[i], 16);
            break;

        default:
            FW_ASSERT(false);
            break;
        }
    }
}

//------------------------------------------------------------------------

void CudaBVH::createTriWoopBasic(const BVH& bvh)
{
    const Array<S32>& tidx = bvh.getTriIndices();
    m_triWoop.resizeDiscard((tidx.getSize() * 64 + Align - 1) & -Align);

    for (int i = 0; i < tidx.getSize(); i++)
    {
        woopifyTri(bvh, i);

        switch (m_layout)
        {
        case BVHLayout_AOS_AOS:
        case BVHLayout_SOA_AOS:
        case BVHLayout_Stackless:
        case BVHLayout_Stackless2:
            memcpy(m_triWoop.getMutablePtr(i * 64), m_woop, 48);
            break;

        case BVHLayout_AOS_SOA:
        case BVHLayout_SOA_SOA:
            for (int j = 0; j < 3; j++)
                memcpy(m_triWoop.getMutablePtr(i * 16 + (m_triWoop.getSize() >> 2) * j), &m_woop[j], 16);
            break;

        default:
            FW_ASSERT(false);
            break;
        }
    }
}

//------------------------------------------------------------------------

void CudaBVH::createTriIndexBasic(const BVH& bvh)
{
    const Array<S32>& tidx = bvh.getTriIndices();
    m_triIndex.resizeDiscard(tidx.getSize() * 4);

    for (int i = 0; i < tidx.getSize(); i++)
        *(S32*)m_triIndex.getMutablePtr(i * 4) = tidx[i];
}

//------------------------------------------------------------------------

void CudaBVH::createCompact(const BVH& bvh, int nodeOffsetSizeDiv, bool stackless)
{
    struct StackEntry
    {
        const BVHNode*  node;
        S32             idx;
        S32             real_idx;
        S32             parent_idx;
        StackEntry(const BVHNode* n = NULL, int i = 0, int ridx=0, int p = 0) : node(n), idx(i), real_idx(ridx), parent_idx(p) {}
    };

    // Construct data.

    Array<Vec4i> nodeData(NULL, 4);
    Array<Vec4i> triWoopData;
    Array<S32> triIndexData;
    Array<StackEntry> stack(StackEntry(bvh.getRoot(), 0, 0, 0x76543210));

    

    while (stack.getSize())
    {
        StackEntry e = stack.removeLast();

        /*
        if (e.idx == 0) {
            printf("compact2 root parent: %x\n", e.parent_idx);
        }

        if (e.real_idx == 64 || e.real_idx == 128) {
            printf("compact2 root child %i (idx: %i) parent: %i\n", e.real_idx, e.idx, e.parent_idx);
        }

        if (e.parent_idx == 0) {
            printf("compact: node %i is child of root \n", e.real_idx, e.parent_idx);
        }
        */

        FW_ASSERT(e.node->getNumChildNodes() == 2);
        const AABB* cbox[2];
        int cidx[2];

        // Process children.

        for (int i = 0; i < 2; i++)
        {
            // Inner node => push to stack.

            const BVHNode* child = e.node->getChildNode(i);
            cbox[i] = &child->m_bounds;
            if (!child->isLeaf())
            {
                cidx[i] = nodeData.getNumBytes() / nodeOffsetSizeDiv;
                if (e.idx == 0) {
                    printf("compact2 root child %i (nodedata_size: %i): %i\n", i, nodeData.getSize(), cidx[i]);
                }
                
                stack.add(StackEntry(child, nodeData.getSize(), cidx[i], e.real_idx));
                nodeData.add(NULL, 4);
                continue;
            }

            // Leaf => append triangles.

            const LeafNode* leaf = reinterpret_cast<const LeafNode*>(child);
            cidx[i] = ~triWoopData.getSize();
            for (int j = leaf->m_lo; j < leaf->m_hi; j++)
            {
                woopifyTri(bvh, j);
                if (m_woop[0].x == 0.0f)
                    m_woop[0].x = 0.0f;
                triWoopData.add((Vec4i*)m_woop, 3);
                triIndexData.add(bvh.getTriIndices()[j]);
                triIndexData.add(0);
                triIndexData.add(0);
            }

            // Terminator.

            triWoopData.add(0x80000000);
            triIndexData.add(0);
        }

        Vec4i extra_data;
       
        if(stackless){
            S32 ord = getChildOrderDim(cbox[0], cbox[1]);
            int p = e.parent_idx; // no need to encode since it has to be an internal node
            extra_data = Vec4i(cidx[0], cidx[1], p, ord);
        }else{
            extra_data = Vec4i(cidx[0], cidx[1], 0, 0);
        }

        // Write entry.

        // TODO try to adjust with nodeData.getPtr

     
        if (e.idx == 0) {
            printf("children of root \n", cidx[0], cidx[1]);
        }

        if (e.real_idx == 64 || e.real_idx == 128) {
            printf("compact2 root child %i parent: %i\n", e.real_idx, e.parent_idx);
        }

        


        Vec4i* dst = nodeData.getPtr(e.idx);
        
        dst[0] = Vec4i(floatToBits(cbox[0]->min().x), floatToBits(cbox[0]->max().x), floatToBits(cbox[0]->min().y), floatToBits(cbox[0]->max().y));
        dst[1] = Vec4i(floatToBits(cbox[1]->min().x), floatToBits(cbox[1]->max().x), floatToBits(cbox[1]->min().y), floatToBits(cbox[1]->max().y));
        dst[2] = Vec4i(floatToBits(cbox[0]->min().z), floatToBits(cbox[0]->max().z), floatToBits(cbox[1]->min().z), floatToBits(cbox[1]->max().z));
        dst[3] = extra_data;
    }

    // Write to buffers.

    m_nodes.resizeDiscard(nodeData.getNumBytes());
    m_nodes.set(nodeData.getPtr(), nodeData.getNumBytes());

    m_triWoop.resizeDiscard(triWoopData.getNumBytes());
    m_triWoop.set(triWoopData.getPtr(), triWoopData.getNumBytes());

    m_triIndex.resizeDiscard(triIndexData.getNumBytes());
    m_triIndex.set(triIndexData.getPtr(), triIndexData.getNumBytes());
}

//------------------------------------------------------------------------

void CudaBVH::woopifyTri(const BVH& bvh, int idx)
{
    const Vec3i* triVtxIndex = (const Vec3i*)bvh.getScene()->getTriVtxIndexBuffer().getPtr();
    const Vec3f* vtxPos = (const Vec3f*)bvh.getScene()->getVtxPosBuffer().getPtr();
    const Vec3i& inds = triVtxIndex[bvh.getTriIndices()[idx]];
    const Vec3f& v0 = vtxPos[inds.x];
    const Vec3f& v1 = vtxPos[inds.y];
    const Vec3f& v2 = vtxPos[inds.z];

    Mat4f mtx;
    mtx.setCol(0, Vec4f(v0 - v2, 0.0f));
    mtx.setCol(1, Vec4f(v1 - v2, 0.0f));
    mtx.setCol(2, Vec4f(cross(v0 - v2, v1 - v2), 0.0f));
    mtx.setCol(3, Vec4f(v2, 1.0f));
    mtx = invert(mtx);

    m_woop[0] = Vec4f(mtx(2,0), mtx(2,1), mtx(2,2), -mtx(2,3));
    m_woop[1] = mtx.getRow(0);
    m_woop[2] = mtx.getRow(1);
}

//------------------------------------------------------------------------
