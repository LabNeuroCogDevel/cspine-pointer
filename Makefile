cspine.db:
	sqlite3 $@ < schema.txt

guide-image-small.png: guide-image.xcf
	magick $< -layers flatten $@

.test: $(wildcard cspine/*.py *.py test/*.py)
	python3 -m pytest test/ | tee $@
