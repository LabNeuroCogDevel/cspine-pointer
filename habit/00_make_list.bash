#!/usr/bin/env bash
# build and cache list so we don't have to wait for slow filesystem access
# 20250611WF - init
cd "$(dirname "$0")" || exit 1
ls /Volumes/Hera/Projects/Habit/mr/BIDS/sub-*/ses-*/anat/sub-*_ses-*_T1w.nii.gz > habit_filelist.txt
