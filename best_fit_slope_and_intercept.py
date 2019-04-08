from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

#xs = [1,2,3,4,5,6] 
#ys = [5,4,6,5,6,7]

xs = np.array([1,2,3,4,5,6], dtype=np.float64) 
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys):
    m = ( (( mean(xs) * mean(ys)) - mean(xs*ys)) /
           ((mean(xs)*mean(xs)) - mean(xs*xs)))

    b = mean(ys) - m*mean(xs)
    return m, b

m,b = best_fit_slope_and_intercept(xs,ys)

regression_line = [(m*x)+b for x in xs]  #-- This function is same as the below function

#for x in xs:
 #   regression_line.append((m*x)+b)

print(m,b)

predict_x = 8  # These line takes a random value for x and help us in making a prediction
predict_y = (m*predict_x)+b

#plt.plot(xs,ys) -- Used to plot a line graph
plt.scatter(xs,ys)  # Used to plot a scatter plot
plt.scatter(predict_x,predict_y,color='r')
plt.plot(xs,regression_line)
plt.show()
