import numpy as np

def tanh_activation(x):
    return np.tanh(x)

def tanh_deriv(x):
    return (1 - x**2)

def feed_forward(b, x1, x2, w1, w2):
    y_in = b + x1*w1 + x2*w2
    y_out = tanh_activation(y_in)
    return y_out

def back_propagation(y_predict, w4, w3, b2, h2, h1, w22, w12, w21, w11, b12, b11, lr):
    global x1
    global x2
    global error

    error = (y_target[i]-y_predict)**2 / 2

    gradient_w4 = (-y_target[i]+y_predict) * tanh_deriv(y_predict) * h2
    gradient_w3 = (-y_target[i]+y_predict) * tanh_deriv(y_predict) * h1
    gradient_b2 = (-y_target[i]+y_predict) * tanh_deriv(y_predict)
    gradient_w22 = (-y_target[i]+y_predict) * tanh_deriv(y_predict) * w4 * tanh_deriv(h2) * x2[i]
    gradient_w12 = (-y_target[i]+y_predict) * tanh_deriv(y_predict) * w4 * tanh_deriv(h2) * x1[i]
    gradient_b12 = (-y_target[i]+y_predict) * tanh_deriv(y_predict) * w4 * tanh_deriv(h2)
    gradient_w21 = (-y_target[i]+y_predict) * tanh_deriv(y_predict) * w3 * tanh_deriv(h1) * x2[i]
    gradient_w11 = (-y_target[i]+y_predict) * tanh_deriv(y_predict) * w3 * tanh_deriv(h1) * x1[i]
    gradient_b11 = (-y_target[i]+y_predict) * tanh_deriv(y_predict) * w3 * tanh_deriv(h1)

    b11 -= lr*gradient_b11
    b12 -= lr*gradient_b12
    b2 -= lr*gradient_b2
    w11 -= lr*gradient_w11
    w12 -= lr*gradient_w12
    w21 -= lr*gradient_w21
    w22 -= lr*gradient_w22
    w3 -= lr*gradient_w3
    w4 -= lr*gradient_w4
    
    return b11, b12, b2, w11, w12, w21, w22, w3, w4

b1 = [round(np.random.randn(), 2), round(np.random.randn(), 2)]
b2 = round(np.random.randn(), 2)
w11 = round(np.random.randn(), 2)
w12 = round(np.random.randn(), 2)
w21 = round(np.random.randn(), 2)
w22 = round(np.random.randn(), 2)
w3 = round(np.random.randn(), 2)
w4 = round(np.random.randn(), 2)
lr = 0.5
x1 = [1, 1, 0, 0]
x2 = [1, 0, 1, 0]
y_target = [-1, 1, 1, -1]
epoch = 1000
start = [b1[0], b1[1], b2, w11, w12, w21, w22, w3, w4]

print('--Training Start--\n')

for a in range(epoch):
    print('Epoch', a+1)

    # Training
    for i in range(len(y_target)):
        print(f'Input Training: ({x1[i]}, {x2[i]})')
        print('Calculating...')
        # Feed Forward
        h1 = feed_forward(b1[0], x1[i], x2[i], w11, w21)
        h2 = feed_forward(b1[1], x1[i], x2[i], w12, w22)
        y_predict = feed_forward(b2, h1, h2, w3, w4)

        # Backpropagation
        b1[0], b1[1], b2, w11, w12, w21, w22, w3, w4 = back_propagation(y_predict, w4, w3, b2, h2, h1, w22, w12, w21, w11, b1[1], b1[0], lr)

        print(f'Prediction at Epoch {a+1}:')
        print('Prediction:', y_predict)
        print()
    
    print('=======================================================')

print('\n--Training End--')

print('\n--Before Training--')
print('Bias 11:', start[0])
print('Bias 12:', start[1])
print('Bias 2:', start[2])
print('Weight 11:', start[3])
print('Weight 12:', start[4])
print('Weight 21:', start[5])
print('Weight 22:', start[6])
print('Weight 3:', start[7])
print('Weight 4:', start[8])

for i in range(len(y_target)):
    h1 = feed_forward(start[0], x1[i], x2[i], start[3], start[5])
    h2 = feed_forward(start[1], x1[i], x2[i], start[4], start[6])
    y_predict = feed_forward(start[2], h1, h2, start[7], start[8])
    print(f'Prediction ({x1[i]},{x2[i]}): {y_predict}')

print('\n--After Training--')
print('Epoch:', epoch)
print('Bias 11:', round(b1[0], 2))
print('Bias 12:', round(b1[1], 2))
print('Bias 2:', round(b2, 2))
print('Weight 11:', round(w11, 2))
print('Weight 12:', round(w12, 2))
print('Weight 21:', round(w21, 2))
print('Weight 22:', round(w22, 2))
print('Weight 3:', round(w3, 2))
print('Weight 4:', round(w4, 2))

for i in range(len(y_target)):
    h1 = feed_forward(b1[0], x1[i], x2[i], w11, w21)
    h2 = feed_forward(b1[1], x1[i], x2[i], w12, w22)
    y_predict = feed_forward(b2, h1, h2, w3, w4)
    print(f'Prediction ({x1[i]},{x2[i]}): {y_predict}')