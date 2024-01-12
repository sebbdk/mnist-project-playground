import csv

def saveResult(blah):
	with open("Output.txt", "w") as text_file:
		text_file.write(f"{blah}")


def blah():
	with open('eggs.csv', 'w', newline='') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=' ',
								quotechar='|', quoting=csv.QUOTE_MINIMAL)
		spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
		spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])