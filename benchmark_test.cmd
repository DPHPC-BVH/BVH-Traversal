@echo off
set LOG=benchmark_test.log

set EXE=rt_x64_Release.exe

Rem KERNELS ================================================================================================

Rem set KERNELS=--kernel=kepler_dynamic_fetch --kernel=fermi_speculative_while_while --kernel=tesla_persistent_packet --kernel=tesla_persistent_speculative_while_while --kernel=tesla_persistent_while_while --kernel=pascal_persistent_stackless --kernel=pascal_persistent_stackless_opt3 --kernel=pascal_persistent_stackless_opt5 --kernel=pascal_stackless_opt3 --kernel=pascal_stackless_opt5

Rem set KERNELS=--kernel=pascal_speculative_stackless --kernel=pascal_speculative_stackless_opt5 --kernel=pascal_speculative_stackless_tex1d --kernel=pascal_stackless_opt3 --kernel=pascal_stackless_opt5
Rem set KERNELS=--kernel=pascal_speculative_stackless_tex1d --kernel=pascal_speculative_stackless_tex1d_2 --kernel=pascal_stackless_opt3
Rem set KERNELS=--kernel=pascal_speculative_stackless_tex1d --kernel=fermi_speculative_while_while
Rem set KERNELS=--kernel=pascal_speculative_stackless_tex1d  --kernel=kepler_dynamic_fetch --kernel=pascal_stackless_opt3 --kernel=pascal_stackless_opt5
Rem set KERNELS=--kernel=pascal_dynamic_fetch_stackless_opt5 --kernel=pascal_dynamic_fetch_stackless --kernel=pascal_speculative_stackless_tex1d
Rem set KERNELS=--kernel=pascal_speculative_stackless_tex1d --kernel=pascal_speculative_stackless_tex1d_2
set KERNELS=--kernel=pascal_speculative_stackless_tex1d_2 --kernel=pascal_speculative_stackless_tex1d_opt5

Rem CAMERAS ================================================================================================

Rem Conference CAMERAS
set CAMERAS=--camera="6omr/04j3200bR6Z/0/3ZEAz/x4smy19///c/05frY109Qx7w////m100" --camera="Lpmr/07k3200CS6Z/0/QqOIz1qfnsx19///c/05frY109Qx7w////m100" --camera="Y1BR00IkZd/0aA9X/0/Gy8Px1ca7Tw19///c/05frY109Qx7w////m100" --camera="XYDl00Gqv600byxY/00IQE4x/jN1jx/9///c/05frY109Qx7w////m100" --camera="w:ie00yxXX00ND1b/03TZ6qy1egt3x/9///c/05frY109Qx7w////m100"

Rem FairyForest CAMERAS
Rem set CAMERAS=--camera="cIxMx/sK/Ty/EFu3z/5m9mWx/YPA5z/8///m007toC10AnAHx///Uy200" --camera="KI/Qz/zlsUy/TTy6z13BdCZy/LRxzy/8///m007toC10AnAHx///Uy200" --camera="mF5Gz1SuO1z/ZMooz11Q0bGz/CCNxx18///m007toC10AnAHx///Uy200" --camera="vH7Jy19GSHx/YN45x//P2Wpx1MkhWy18///m007toC10AnAHx///Uy200" --camera="ViGsx/KxTFz/Ypn8/05TJTmx1ljevx18///m007toC10AnAHx///Uy200"

Rem Sibenik CAMERAS
Rem set CAMERAS=--camera="ytIa02G35kz1i:ZZ/0//iSay/5W6Ex19///c/05frY109Qx7w////m100" --camera=":Wp802ACAD/2x9OQ/0/waE8z/IOKbx/9///c/05frY109Qx7w////m100" --camera="CFtpy/s6ea/28btX/0172CFy/K5g1z/9///c/05frY109Qx7w////m100" --camera="steO/0TlN1z1tsDg/03InaMz/bqZxx/9///c/05frY109Qx7w////m100" --camera="HJv//034:Rx1S4Xh/03dpXux1BVmGw/9///c/05frY109Qx7w////m100"

set MESH=scenes/rt/conference/conference.obj
Rem set MESH=scenes/rt/fairyforest/fairyforest.obj
Rem set MESH=scenes/rt/sibenik/sibenik.obj

Rem set CAMERAS=--camera="6omr/04j3200bR6Z/0/3ZEAz/x4smy19///c/05frY109Qx7w////m100"

Rem echo %KERNELS%
Rem echo %CAMERAS%

Rem set EXTRA=--ao-radius=0.3

rmdir /s /q %~dp0\cudacache

%EXE% benchmark --log=%LOG% --mesh=%MESH% %EXTRA% %CAMERAS% %KERNELS% %KERNELS%

echo Done.