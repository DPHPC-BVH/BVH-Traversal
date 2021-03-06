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

#pragma once
#include "gpu/CudaCompiler.hpp"
#include "cuda/CudaBVH.hpp"
#include "ray/RayBuffer.hpp"

namespace FW
{
//------------------------------------------------------------------------

class CudaTracer
{
public:
                        CudaTracer              (void);
                        ~CudaTracer             (void);

    void                setMessageWindow        (Window* window)        { m_compiler.setMessageWindow(window); }
    void                setKernel               (const String& kernelName);
    BVHLayout           getDesiredBVHLayout     (void) const            { return (BVHLayout)m_kernelConfig.bvhLayout; }
    void                setBVH                  (CudaBVH* bvh)          { m_bvh = bvh; }

    F32                 traceBatch              (RayBuffer& rays); // returns launch time in seconds

private:
    CudaModule*         compileKernel           (void);

private:
                        CudaTracer              (const CudaTracer&); // forbidden
    CudaTracer&         operator=               (const CudaTracer&); // forbidden

private:
    CudaCompiler        m_compiler;
    String              m_kernelName;
    KernelConfig        m_kernelConfig;
    CudaBVH*            m_bvh;
    // For profiling.
    S32                 m_numTraced;
    S32                 m_profileInterval;
    S32                 m_numProfiled;
    S32                 m_numTotalProfile;
};

//------------------------------------------------------------------------
}
