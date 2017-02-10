import csv

FEAT_FILE = 'wrd-feats.csv'
OUT_FILE = 'n-wrd-feats.csv'

new_csv = []
with open(FEAT_FILE) as featfile:
	csvreader = csv.reader(featfile, delimiter=',')
	for row in csvreader:
		if '--undefined--' not in row:
			new_csv.append(row)


with open(OUT_FILE, 'w') as featfile:
	csvwriter = csv.writer(featfile, delimiter=',')
	for row in new_csv:
		print (row)
		csvwriter.writerow(row)



