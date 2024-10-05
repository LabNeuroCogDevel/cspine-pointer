cspine.db:
	sqlite3 $@ < schema.txt

guide-image-small.png: guide-image.xcf
	magick $< -layers flatten $@
