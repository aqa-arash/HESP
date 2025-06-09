@echo off
echo Benchmarking running.

set configs=config2k5_co_acc config2k5_co config2k5 config5k_co_acc config5k_co config5k config10k_co_acc config10k_co config10k

REM Remove old runtimes.txt if it exists
if exist runtimes.txt del runtimes.txt

for %%c in (%configs%) do (
    echo Running %%c.txt 3 times...
    for /l %%i in (1,1,3) do (
        md.exe %%c.txt
    )
)

echo Benchmarking complete. All runtimes are saved in runtimes.txt.
pause