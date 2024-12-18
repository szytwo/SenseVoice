SET SENSEVOICE_DEVICE=cuda:0
@echo off
chcp 65001
call venv\python.exe api.py

@echo 请按任意键继续
call pause