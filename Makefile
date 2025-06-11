cspine.db:
	sqlite3 $@ < schema.txt

habit/habit_filelist.txt: $(wildcard /Volumes/Hera/Projects/Habit/mr/BIDS/sub-*/ses-*/anat/sub-*_ses-*_T1w.nii.gz)
	habit/00_make_list.bash

guide-image-small.png: guide-image.xcf
	magick $< -layers flatten $@
