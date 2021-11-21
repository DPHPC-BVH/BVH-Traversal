@echo off
set LOG=benchmark_test.log

set EXE=rt_x64_Release.exe

rmdir /s /q %~dp0\cudacache

Rem norm opt1 opt2
Rem %EXE% benchmark --log=%LOG% --mesh=scenes/rt/conference/conference.obj --camera="6omr/04j3200bR6Z/0/3ZEAz/x4smy19///c/05frY109Qx7w////m100" --camera="Lpmr/07k3200CS6Z/0/QqOIz1qfnsx19///c/05frY109Qx7w////m100" --camera="Y1BR00IkZd/0aA9X/0/Gy8Px1ca7Tw19///c/05frY109Qx7w////m100" --camera="XYDl00Gqv600byxY/00IQE4x/jN1jx/9///c/05frY109Qx7w////m100" --camera="w:ie00yxXX00ND1b/03TZ6qy1egt3x/9///c/05frY109Qx7w////m100" --kernel=pascal_persistent_stackless --kernel=pascal_persistent_stackless_opt1 --kernel=pascal_persistent_stackless_opt2

Rem norm opt2 opt3
Rem %EXE% benchmark --log=%LOG% --mesh=scenes/rt/conference/conference.obj --camera="6omr/04j3200bR6Z/0/3ZEAz/x4smy19///c/05frY109Qx7w////m100" --camera="Lpmr/07k3200CS6Z/0/QqOIz1qfnsx19///c/05frY109Qx7w////m100" --camera="Y1BR00IkZd/0aA9X/0/Gy8Px1ca7Tw19///c/05frY109Qx7w////m100" --camera="XYDl00Gqv600byxY/00IQE4x/jN1jx/9///c/05frY109Qx7w////m100" --camera="w:ie00yxXX00ND1b/03TZ6qy1egt3x/9///c/05frY109Qx7w////m100" --kernel=pascal_persistent_stackless --kernel=pascal_persistent_stackless_opt2 --kernel=pascal_persistent_stackless_opt3

Rem norm opt1 opt2 opt3
%EXE% benchmark --log=%LOG% --mesh=scenes/rt/conference/conference.obj --camera="6omr/04j3200bR6Z/0/3ZEAz/x4smy19///c/05frY109Qx7w////m100" --camera="Lpmr/07k3200CS6Z/0/QqOIz1qfnsx19///c/05frY109Qx7w////m100" --camera="Y1BR00IkZd/0aA9X/0/Gy8Px1ca7Tw19///c/05frY109Qx7w////m100" --camera="XYDl00Gqv600byxY/00IQE4x/jN1jx/9///c/05frY109Qx7w////m100" --camera="w:ie00yxXX00ND1b/03TZ6qy1egt3x/9///c/05frY109Qx7w////m100" --kernel=pascal_persistent_stackless --kernel=pascal_persistent_stackless_opt1 --kernel=pascal_persistent_stackless_opt2 --kernel=pascal_persistent_stackless_opt3

Rem norm opt2 opt3 kepler
Rem %EXE% benchmark --log=%LOG% --mesh=scenes/rt/conference/conference.obj --camera="6omr/04j3200bR6Z/0/3ZEAz/x4smy19///c/05frY109Qx7w////m100" --camera="Lpmr/07k3200CS6Z/0/QqOIz1qfnsx19///c/05frY109Qx7w////m100" --camera="Y1BR00IkZd/0aA9X/0/Gy8Px1ca7Tw19///c/05frY109Qx7w////m100" --camera="XYDl00Gqv600byxY/00IQE4x/jN1jx/9///c/05frY109Qx7w////m100" --camera="w:ie00yxXX00ND1b/03TZ6qy1egt3x/9///c/05frY109Qx7w////m100" --kernel=pascal_persistent_stackless --kernel=pascal_persistent_stackless_opt2 --kernel=pascal_persistent_stackless_opt3 --kernel=kepler_dynamic_fetch

Rem opt2 opt3 kepler
Rem %EXE% benchmark --log=%LOG% --mesh=scenes/rt/conference/conference.obj --camera="6omr/04j3200bR6Z/0/3ZEAz/x4smy19///c/05frY109Qx7w////m100" --camera="Lpmr/07k3200CS6Z/0/QqOIz1qfnsx19///c/05frY109Qx7w////m100" --camera="Y1BR00IkZd/0aA9X/0/Gy8Px1ca7Tw19///c/05frY109Qx7w////m100" --camera="XYDl00Gqv600byxY/00IQE4x/jN1jx/9///c/05frY109Qx7w////m100" --camera="w:ie00yxXX00ND1b/03TZ6qy1egt3x/9///c/05frY109Qx7w////m100" --kernel=pascal_persistent_stackless_opt2 --kernel=pascal_persistent_stackless_opt3 --kernel=kepler_dynamic_fetch

Rem opt2 kepler opt3 
Rem %EXE% benchmark --log=%LOG% --mesh=scenes/rt/conference/conference.obj --camera="6omr/04j3200bR6Z/0/3ZEAz/x4smy19///c/05frY109Qx7w////m100" --camera="Lpmr/07k3200CS6Z/0/QqOIz1qfnsx19///c/05frY109Qx7w////m100" --camera="Y1BR00IkZd/0aA9X/0/Gy8Px1ca7Tw19///c/05frY109Qx7w////m100" --camera="XYDl00Gqv600byxY/00IQE4x/jN1jx/9///c/05frY109Qx7w////m100" --camera="w:ie00yxXX00ND1b/03TZ6qy1egt3x/9///c/05frY109Qx7w////m100" --kernel=pascal_persistent_stackless_opt2 --kernel=kepler_dynamic_fetch --kernel=pascal_persistent_stackless_opt3


Rem opt3
Rem %EXE% benchmark --log=%LOG% --mesh=scenes/rt/conference/conference.obj --camera="6omr/04j3200bR6Z/0/3ZEAz/x4smy19///c/05frY109Qx7w////m100" --camera="Lpmr/07k3200CS6Z/0/QqOIz1qfnsx19///c/05frY109Qx7w////m100" --camera="Y1BR00IkZd/0aA9X/0/Gy8Px1ca7Tw19///c/05frY109Qx7w////m100" --camera="XYDl00Gqv600byxY/00IQE4x/jN1jx/9///c/05frY109Qx7w////m100" --camera="w:ie00yxXX00ND1b/03TZ6qy1egt3x/9///c/05frY109Qx7w////m100" --kernel=pascal_persistent_stackless_opt3

Rem opt2
Rem %EXE% benchmark --log=%LOG% --mesh=scenes/rt/conference/conference.obj --camera="6omr/04j3200bR6Z/0/3ZEAz/x4smy19///c/05frY109Qx7w////m100" --camera="Lpmr/07k3200CS6Z/0/QqOIz1qfnsx19///c/05frY109Qx7w////m100" --camera="Y1BR00IkZd/0aA9X/0/Gy8Px1ca7Tw19///c/05frY109Qx7w////m100" --camera="XYDl00Gqv600byxY/00IQE4x/jN1jx/9///c/05frY109Qx7w////m100" --camera="w:ie00yxXX00ND1b/03TZ6qy1egt3x/9///c/05frY109Qx7w////m100" --kernel=pascal_persistent_stackless_opt2


Rem %EXE% benchmark --log=%LOG% --mesh=scenes/rt/conference/conference.obj --camera="6omr/04j3200bR6Z/0/3ZEAz/x4smy19///c/05frY109Qx7w////m100" --camera="Lpmr/07k3200CS6Z/0/QqOIz1qfnsx19///c/05frY109Qx7w////m100" --camera="Y1BR00IkZd/0aA9X/0/Gy8Px1ca7Tw19///c/05frY109Qx7w////m100" --camera="XYDl00Gqv600byxY/00IQE4x/jN1jx/9///c/05frY109Qx7w////m100" --camera="w:ie00yxXX00ND1b/03TZ6qy1egt3x/9///c/05frY109Qx7w////m100" --kernel=pascal_persistent_stackless --kernel=pascal_persistent_stackless_opt --kernel=pascal_persistent_stackless_opt2

Rem %EXE% benchmark --log=%LOG% --mesh=scenes/rt/conference/conference.obj --camera="6omr/04j3200bR6Z/0/3ZEAz/x4smy19///c/05frY109Qx7w////m100" --camera="Lpmr/07k3200CS6Z/0/QqOIz1qfnsx19///c/05frY109Qx7w////m100" --camera="Y1BR00IkZd/0aA9X/0/Gy8Px1ca7Tw19///c/05frY109Qx7w////m100" --camera="XYDl00Gqv600byxY/00IQE4x/jN1jx/9///c/05frY109Qx7w////m100" --camera="w:ie00yxXX00ND1b/03TZ6qy1egt3x/9///c/05frY109Qx7w////m100"

echo Done.