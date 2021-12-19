## Description of all the kernels

| Kernel name          | reference kernel? | algorithm   | data structure     | cuda threads | implementation techniques                                | additional optimizations |
| -------------------- | ----------------- | ----------- | ------------------ | ------------ | -------------------------------------------------------- | ------------------------ |
| kepler_dynamic_fetch | yes               | stack-based | BVHLayout_Compact2 | persistent   | while-while, speculative traversal, dynamic ray-fetching |                          |
|tesla_persistent_package              | yes               | stack-based  | BVHLayout_AOS/SOA_AOS/SOA    | persistent     | package                                                        ||
| |  |  |  |  |  ||
|tesla_persistent_while_while               | yes               | stack-based  | BVHLayout_AOS/SOA_AOS/SOA    | persistent     | while-while                                                    ||
|tesla_persistent_if_while                | yes               | stack-based  | BVHLayout_AOS/SOA_AOS/SOA    | persistent     | if-while                                                       ||
|tesla_persistent_while_if                | yes               | stack-based  | BVHLayout_AOS/SOA_AOS/SOA    | persistent     | while-if                                                       ||
|tesla_persistent_while_if | yes | stack-based | BVHLayout_AOS/SOA_AOS/SOA | persistent | if-if ||
| |  |  |  |  |  ||
|fermi_speculative_while_while                 | yes               | stack-based  | BVHLayout_Compact            | non-persistent | while-while, speculative traversal                             ||
|tesla_persistent_speculative_while_while|  yes               | stack-based  | BVHLayout_AOS/SOA_AOS/SOA    | persistent     | while-while                                                    ||
|tesla_persistent_speculative_while_while_early_break | no                | stack-based  | BVHLayout_AOS/SOA_AOS/SOA    | persistent     | while-while                                                    |break traversal loop if more than 16 threads in a warp have found triangles or exited the loop|
| |  |  |  |  |  ||
|tesla_persistent_speculative_while_while_warp_sync | no | stack-based | BVHLayout_AOS/SOA_AOS/SOA | persistent | while-while, with new warp_sync intrinsics ||
|tesla_persistent_speculative_while_while_warp_sync_early_break | no | stack-based | BVHLayout_AOS/SOA_AOS/SOA | persistent | while-while, with new warp_sync intrinsics |break traversal loop if more than 16 threads in a warp have found triangles or exited the loop|
| |  |  |  |  |  ||
|pascal_persistent_stackless            | no                | stackless    | BVHLayout_Stackless          | persistent     | if-if                                                          ||
|pascal_persistent_stackless_opt3          | no                | stackless    | BVHLayout_Stackless          | persistent     | if-if                                                          |branch elemination (no performance difference -> probably done by compiler)|
|pascal_persistent_stackless_opt4          | no                | stackless    | BVHLayout_Stackless          | persistent     | if-if                                                          |ray-independent traversal order|
|pascal_persistent_stackless_opt5           | no                | stackless    | BVHLayout_Stackless          | persistent     | if-if                                                          |ray-independent traversal order, removed register spilling|
|pascal_stackless_opt3                   | no                | stackless    | BVHLayout_Stackless          | non-persistent | if-if                                                          |branch elemination|
|pascal_stackless_opt5                   | no                | stackless    | BVHLayout_Stackless          | non-persistent | if-if                                                          |ray-independent traversal order, removed register spilling|
| |  |  |  |  |  ||
|pascal_speculative_stackless             | no                | stackless    | BVHLayout_Compact_Stackless  | non-persistent | while-while, speculative traversal                             ||
|pascal_speculative_stackless_opt5        | no                | stackless    | BVHLayout_Compact_Stackless  | non-persistent | while-while, speculative traversal                             |ray-independent traversal order|
|pascal_speculative_stackless_tex1d        | no                | stackless    | BVHLayout_Compact2_Stackless | non-persistent | while-while, speculative traversal                             |(tex1Dfetch)|
|pascal_speculative_stackless_tex1d_2       | no                | stackless    | BVHLayout_Compact2_Stackless | non-persistent | while-while, speculative traversal                             |(tex1Dfetch), removed boolean indicator signaling that a SIMD lane has found a leaf (like in dynamic fetch)|
|pascal_speculative_stackless_tex1d_opt5    | no                | stackless    | BVHLayout_Compact2_Stackless | non-persistent | while-while, speculative traversal                             |(tex1Dfetch), rm bool, ray-independent traversal order|
| |  |  |  |  |  ||
|pascal_dynamic_fetch_stackless           | no                | stackless    | BVHLayout_Compact2_Stackless | persistent | while-while, speculative traversal, dynamic ray-fetching           |(tex1Dfetch)|
|pascal_dynamic_fetch_stackless_opt5        | no                | stackless    | BVHLayout_Compact2_Stackless | persistent | while-while, speculative traversal, dynamic ray-fetching           |(tex1Dfetch), ray-independent traversal order|
| |  |  |  |  |  ||
|                                                              |                   |             |                              |                |                                                          |                                                              |
|                                                              |                   |  |  |  |  ||

