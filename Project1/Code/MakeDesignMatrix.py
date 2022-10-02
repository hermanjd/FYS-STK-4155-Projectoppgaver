def makeDesignMatrix(x,y,polinomialOrder):
	designMatrix = []
	for index_x, element_x  in enumerate(x):
		designMatrixRow = []
		designMatrixRow.append(1)
		for i in range(polinomialOrder):
			designMatrixRow.append(pow(x[index_x], i+1))
			designMatrixRow.append(pow(y[index_x], i+1))
		designMatrix.append(designMatrixRow)
	return designMatrix