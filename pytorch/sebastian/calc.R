print(getwd())
data = read.csv("pytorch/sebastian/test.csv", header=TRUE, stringsAsFactors=FALSE)

length(data)

prog = data[,1]

progList = split(prog, rep_len(1:33, length(prog)))

d = c()
for( i in progList )
	d = append(d, tail(i, n=1))

print(d)
print(mean(d))