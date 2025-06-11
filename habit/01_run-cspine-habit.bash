#!/usr/bin/env bash
# 20250210WF - init
#
cd $(dirname "$0")
../main.py $(cat habit_filelist.txt )
