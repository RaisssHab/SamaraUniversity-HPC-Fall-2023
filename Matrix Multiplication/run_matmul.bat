@echo off
setlocal enabledelayedexpansion
set log_file=log.txt
echo. > %log_file%
set Ns_list=100 200 100 100 300 600 300 300 800 1600 800 800
set Ms_list=100 100 200 100 300 300 600 300 800 800 1600 800
set Ks_list=100 100 100 200 300 300 300 600 800 800 800 1600
rem Define the output file


rem Determine the number of triplets
set triplet_count=0
set ii=0
for %%a in (%Ns_list%) do (
  set Ns[!ii!]=%%a
  set /a triplet_count+=1
  set /a ii+=1
)
set /a ii=0
for %%a in (%Ms_list%) do (
	set Ms[!ii!]=%%a
	set /a ii+=1
)
set /a ii=0
for %%a in (%Ks_list%) do (
	set Ks[!ii!]=%%a
	set /a ii+=1
)

set /a end_value=triplet_count-1

echo fmad=false
echo fmad=false >> %log_file%
rem Loop through each triplet
for /l %%i in (0,1,!end_value!) do (
    set N=!Ns[%%i]!
    set M=!Ms[%%i]!
    set K=!Ks[%%i]!

    rem Run HPC MatMul input data.exe
	echo "HPC MatMul input data.exe" !N! !M! !K!
    "HPC MatMul input data.exe" !N! !M! !K!

    rem Run HPC MatMul.exe and record the output to the log file
    echo N = !N! M = !M! K = !K! combination: >> %log_file%
    "HPC MatMul (fmad=false).exe" >> %log_file%
    echo. >> %log_file%
)

echo fmad=true
echo fmad=true >> %log_file%
rem Loop through each triplet
for /l %%i in (0,1,!end_value!) do (
    set N=!Ns[%%i]!
    set M=!Ms[%%i]!
    set K=!Ks[%%i]!

    rem Run HPC MatMul input data.exe
	echo "HPC MatMul input data.exe" !N! !M! !K!
    "HPC MatMul input data.exe" !N! !M! !K!

    rem Run HPC MatMul.exe and record the output to the log file
    echo N = !N! M = !M! K = !K! combination: >> %log_file%
    "HPC MatMul (fmad=true).exe" >> %log_file%
    echo. >> %log_file%
)
endlocal