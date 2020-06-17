import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time

r_min = 1
r_max = 5
m_min = -5
m_max = 5
a_min = 1
a_max = 3
b_min = 0
b_max = .5



# Generates noisy data of a specified type: linear, circular, or elliptical 
def gen_data(type, N=1000):

    if type == 'linear':
        m = np.random.uniform(m_min, m_max)
        b = np.random.uniform(b_min, b_max)
        x = np.random.randn(N) 
        y = m * x + b
        noise = 2 * np.random.rand(N, 2) - 1 # zero centered noise 
        data = np.stack((x, y), axis=1) + noise
        return data, m, b

    elif type == 'circular':
        r = np.random.uniform(r_min, r_max)
        phi = np.random.randn(N) * 2 * np.pi
        noise = 2 * np.random.rand(N) - 1
        x = (r * np.ones(N) + noise) * np.cos(phi) # elementwise multiply
        y = (r * np.ones(N) + noise) * np.sin(phi)
        data = np.stack((x, y), axis=1)

        return data, r



# Return parameters of optimal approximation of the specified data
def fit_data(type, data, N=1000):

    if type == 'linear':
        # solve for m, b given x, y to minimize norm(y - (mx + b))^2
        # we set up as: min norm(x_wiggle * beta - y)^2 where beta = [m, b]
        x = data[:,0]
        y = data[:,1]

        # x_wiggle = [x 1] 
        x_wiggle = np.stack((x, np.ones(data.shape[0])), axis=1)

        # solve least squares problem 
        beta = np.matmul(np.linalg.inv(np.matmul(x_wiggle.T, x_wiggle)), np.matmul(x_wiggle.T, y))
        m_ls = beta[0]
        b_ls = beta[1]

        return m_ls, b_ls


    elif type == 'circular':
        # solve for r using cvxpy
        r = cp.Variable() # in x^2 + y^2 = r^2
        obj_terms = []
        for i in range(N):
            obj_terms.append(cp.square(r - np.sqrt(data[i,0] ** 2 + data[i,1] ** 2)))
        obj = cp.Minimize(cp.sum(obj_terms))
        constraints = [r >= 0]
        prob = cp.Problem(obj, constraints)
        prob.solve()

        return r.value, obj.value, prob.status




def main():

    ########### Generate and Fit Linear Data ###########
    N = 500
    data, m, b = gen_data('linear', N)
    m_est, b_est = fit_data('linear', data, N)
    fig = plt.figure(1, figsize=[6,6])

    print("Actual m: ", m)
    print("Actual b: ", b)
    print("Estimated m: ", m_est)
    print("Estimated b: ", b_est)

    x = range(-5, 5)
    plt.plot(x, m_est * x + b_est, 'r')

    plt.scatter(data[:,0], data[:,1])
    plt.title("Modeling linear data")
    plt.show()


    print()

    # ########### Generate and Fit Circular Data ###########
    data, r = gen_data('circular', N)
    r_est, error, status = fit_data('circular', data, N)
    fig = plt.figure(1, figsize=[6,6])
    if status != 'optimal':
        print("Optimization failed.")
    else:
        print("Actual r: ", r)
        print("Estimated r: ", r_est)

        phi = np.linspace(0, 2 * np.pi, 200)
        x_est = r_est * np.cos(phi)
        y_est = r_est * np.sin(phi)
        plt.plot(x_est, y_est, 'r')

    plt.scatter(data[:,0], data[:,1])
    plt.title("Modeling circular data with cvxpy")
    plt.show()




if __name__ == "__main__":
    main()










