print(getwd())
data = read.csv("pytorch/sebastian/test.csv", header=TRUE, stringsAsFactors=FALSE)

length(data)

prog = data[,1]

progList = split(prog, rep_len(1:33, length(prog)))

d = c()
for( i in progList )
	d = append(d, tail(i, n=1))

#print(d)
#print(mean(d))

data = c(0.6748, 0.7815, 0.4864, 0.5747, 0.4362, 0.7878, 0.5114, 0.718, 0.7403, 0.5678, 0.665, 0.6249, 0.7032, 0.7902, 0.4356, 0.8038, 0.4899, 0.7473, 0.4882, 0.7356, 0.7579, 0.7409, 0.6592, 0.7484, 0.3622, 0.7205, 0.7804, 0.7613, 0.7601, 0.6071, 0.7802, 0.6987, 0.7384)
mean(data)
sd(data)