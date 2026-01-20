@echo off
cd /d "c:\Users\Viraj Naik\Desktop\Suvidha"
set GIT_EDITOR=true
set GIT_SEQUENCE_EDITOR=true
git commit --no-edit
git push -u origin main
echo Push completed
pause
