#         |1 u'>=0
#f(u') =  |
#         |0 u'<0

#
# s1| s2   | t
# ---------------
# 0 |  0   |   1
# 0 |  1   |   1
# 1 |  0   |   1
# 1 |  1   |   0

hardlim <-function(x) {
  y <- ifelse(x >= 0, 1, 0)
}

W <- c(-0.7,-0.3,-0.2)
X <- matrix(
  c(-1,0,0,
    -1,0,1,
    -1,1,0,
    -1,1,1),
  nrow = 4,
  ncol = 3,
  byrow = TRUE
)

t <- c(1,1,1,0)

n <- 0.1

e <- c()

for (epoch in 1:2){
  for (q in 1:4){
    e[q] <- t[q] - hardlim(sum(X[q,]*W))
    W <- W + X[q,]*n*e[q]
  }
}

print(e)
print(W)

