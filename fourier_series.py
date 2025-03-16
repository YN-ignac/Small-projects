import numpy as np
import matplotlib.pyplot as plt

figure = plt.figure(figsize=(8, 4.125))

pi = np.pi
x = np.linspace(-1,1,100)
y = x

a0 = 1/pi * np.trapz(y, x, dx=1/100)

y_fourier = np.zeros(len(x)) + a0/2

for n in range(1,100):

    error = np.sqrt(np.trapz(np.abs(np.add(y_fourier, -y))**2, x, dx=1/100))

    figure.clear()

    axis = figure.subplots()

    axis.plot(x, y_fourier, color='black', label='Fourier Approximation')
    axis.plot(x, y, '--', color='red', label='Function')

    axis.set_title('Evaluation f(x) = x with fourier having: {} terms.\n Mean Squarred Error: {}'.format(n+1, error))
    axis.set_xlabel('x')
    axis.set_ylabel('f(x) - Using Fourier')
    plt.xlim(1.2*min(x), 1.2*max(x))
    plt.ylim(1.2*min(y), 1.2*max(y))
    plt.grid()
    plt.legend()
    plt.draw()
    plt.pause(0.05)

    an = 1/pi * np.trapz(y*np.cos(n*x), x, dx=1/100)
    bn = 1/pi * np.trapz(y*np.sin(n*x), x, dx=1/100)

    fourier_term = an * np.cos(n*x) + bn * np.sin(n*x)

    y_fourier = np.add(fourier_term, y_fourier)

plt.show()

print('Fourier series: ', y_fourier)