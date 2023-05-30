import streamlit as st



import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import gamma as gamma_func
from scipy.stats import invgamma
from scipy.special import kv
from scipy.linalg import expm
import time as t
import copy
from numpy.matlib import repmat
from matplotlib.colors import LogNorm
import pandas as pd



class GammaDistr:
    def __init__(self, alpha, beta, rng=np.random.default_rng(10)):
        self.alpha = alpha
        self.beta = beta
        self.distribution = 'Gamma'
        # self.rng = rng

        self.T = None
        self.start_time = None
        self.sample_size = None
        self.rate = None

    def set_process_conditions(self, t0, T, END, sample_size, TEST=False):
        self.start_time = t0
        # t0 = t_i
        # T = t_i+1
        self.T = T
        self.sample_size = sample_size
        self.rate = 1/(T-t0) # set 1.0 to T
        if TEST:
            print('dif rate')
            self.rate=1/(T-t0)
        # self.rate = T / (t_i+1 - t_i)

        gsamps = int(10. / self.beta)
        if gsamps < 50:
            gsamps = 50
        elif gsamps > 10000:
            gsamps = 10000
            print('Warning ---> beta too low for a good approximation')
        # print('gsamps = {}'.format(gsamps))

        if (T-t0) > 1:
            gsamps = gsamps * int(2/3 * (T-t0))


        self.sample_size = gsamps # override sample size

        # self.sample_size = sample_size
        # print('sample size = {}, dt = {}'.format(self.sample_size, T-t0))


class NormalDistr:
    def __init__(self, mean, std, secondary_distr=None, rng=np.random.default_rng(10)):
        self.mean = mean
        self.std = std
        self.distribution = 'Normal'
        # self.rng = rng

        self.secondary_distr = secondary_distr

    def NormalGammaPDF(self, x):
        gamma_distr = self.secondary_distr

        t = self.secondary_distr.T

        t1 = 2*np.exp(self.mean*x/self.std**2)

        delta = 2*self.std**2/gamma_distr.beta + self.mean**2
        tau = t/gamma_distr.beta - 0.5

        t2 = np.power(gamma_distr.beta, t/gamma_distr.beta)*np.sqrt(2*np.pi*(self.std**2))*gamma_func(t/gamma_distr.beta)
        t3 = (np.abs(x)/(self.std**2)) * np.sqrt(delta)
        t4 = (1/gamma_distr.beta) - 0.5
        t5 = (1./self.std**2) * np.sqrt(self.mean**2 + (2*(self.std**2)/gamma_distr.beta)*np.abs(x))


        return (t1 / t2) * (x**2/delta)**(tau/2) * kv(tau,t3)


class DistributionSimulator:

    def __init__(self, DistributionObject): #*OtherObject
        self.distribution = DistributionObject.distribution
        self.distr_obj = DistributionObject
        # self.rng = DistributionObject.rng
        # self.secondary_distr = OtherObject

        self.sorted_process_set = None
        self.time_arr = None
        self.process_path = None
        self.NG_jump_series = None
        self.task = None

    def plot_simulation_distribution(self, process_endpoints, T):

        plt.rcParams["mathtext.fontset"] = "cm"  # Set MathText font to "cm"
        plt.rcParams['figure.dpi'] = 300

        fig, ax = plt.subplots()
        xx, bins, p = ax.hist(process_endpoints, bins=110, density=True)

        if self.distribution == 'Gamma':
            T = self.distr_obj.T
            x = np.linspace(0, 2*T, 20000)
            shape = (self.distr_obj.alpha ** 2) * T / self.distr_obj.beta
            rate = self.distr_obj.beta / self.distr_obj.alpha
            y = stats.gamma.pdf(x, a=shape, loc=0, scale=rate)

        else:  # self.distribution == 'Normal':
            x = np.linspace(-2*T, 2*T, 30000)
            # y2 = norm.pdf(x, loc=0, scale=1)
            y = self.distr_obj.NormalGammaPDF(x)
            # y = np.exp(-0.5 * x ** 2) / np.sqrt(
            #     2 * np.pi)  # Normal distribution PDF with mean 0 and standard deviation 1

        # ax.plot(x, y2, label='Standard N(0,1)')
        ax.plot(x, y, linewidth=3)

        # title = 'Endpoints For VG Process Simulation: '
        # # formatted_title = r'$\mathrm{' + title.replace(' ', r'\ ') + r' \alpha = ' + str(self.distr_obj.alpha) + r', \beta = ' + str(
        # #     self.distr_obj.beta) + r'}$'

        # formatted_title = r'$\mathrm{' + title.replace(' ', r'\ ') + r'\alpha = ' + str(1) + r', \beta = ' + str(
        #     0.001) + r', \mu = ' + str(0) + r', \sigma^2 = ' + str(1) + r'}$'

        # ax.set_xlabel(r'$\mathrm{\mathcal{VG}}$', fontsize=14)  # LaTeX formatting for x-axis label
        ax.set_ylabel(r'$\mathrm{PDF}$', fontsize=14)  # LaTeX formatting for y-axis label
        # ax.set_title(formatted_title, fontsize=14)  # LaTeX formatting for title

        ax.tick_params(axis='x', labelsize=10)  # Set x-axis tick label font size
        ax.tick_params(axis='y', labelsize=10)

        ax.grid(True, linewidth=0.5)  # Add grid
        plt.legend()
        plt.show()
        # st.pyplot(plt)

        return fig, ax

    def tractable_inverse_gamma_tail(self, alpha, beta, x):
        return 1 / ((alpha / beta) * (np.exp(beta * x / alpha ** 2) - 1))

    def acceptance_probability(self, alpha, beta, x):
        return (1 + alpha * x / beta) * np.exp(-alpha * x / beta)

    def perform_acc_rej(self, jumps, probabilities):

        unif = np.random.random(1_000)[:jumps.shape[0]]
        accepted_values = np.where(probabilities > unif, jumps, 0.)

        accepted_values = accepted_values[accepted_values>0.]

        return accepted_values

    def generate_jump_times(self, num_acceps, start_time, end_time):

        times = (end_time - start_time) * np.random.random(1_000)[:num_acceps] + start_time

        return times

    def process_simulation(self, *prev_sim_data):

        DistributionObject = self.distr_obj
        # rng = DistributionObject.rng

        if self.distribution == 'Gamma':

            T = DistributionObject.T
            t0 = DistributionObject.start_time
            sample_size = DistributionObject.sample_size
            alpha = DistributionObject.alpha
            beta = DistributionObject.beta

            exp_rvs = np.random.exponential(scale=DistributionObject.rate, size=1_000)[:sample_size]
            poisson_epochs = np.cumsum(exp_rvs)

            jump_sizes = self.tractable_inverse_gamma_tail(alpha, beta, poisson_epochs)
            acceptance_probabilities = self.acceptance_probability(alpha, beta, jump_sizes)

            jump_sizes = self.perform_acc_rej(jump_sizes, acceptance_probabilities)

            jump_times = self.generate_jump_times(jump_sizes.shape[0], t0, T)

            jumps_and_times = zip(jump_sizes, jump_times)
            self.sorted_process_set = sorted(jumps_and_times, key=lambda x: x[1])
            self.process_path = np.cumsum([jump_time_set[0] for jump_time_set in self.sorted_process_set])
            self.time_arr = [jump_time_set[1] for jump_time_set in self.sorted_process_set]

            return self.process_path, self.time_arr, self.sorted_process_set

        if self.distribution == 'Normal': # Normal Gamma process

            if self.distr_obj.secondary_distr == None:
                raise ValueError('Need to create a gamma distribution to use in our NG simulation')

            # DO NORMAL DISTR FUNCTIONS
            mean = DistributionObject.mean
            std = DistributionObject.std



            # Create a GammaDistr Object (with necessary params). Use this to make DistrSim Object. Run the Gamma Sim, then return the sorted process set
            hidden_gamma_distr = DistributionObject.secondary_distr
            hidden_gamma_sim = DistributionSimulator(hidden_gamma_distr)
            hidden_gamma_sim.process_simulation() #hidden_gamma_distr) # call .process_simulation to generate jumps and times
            hidden_gamma_process_set = hidden_gamma_sim.sorted_process_set

            #TODO: Plot the hidden sparse gamma simulation
            # if self.task != 'SS_SIM':
            #     plt = plotter(hidden_gamma_sim.time_arr, hidden_gamma_sim.process_path, 'Hidden Gamma Sim (Of NG)', 'Time', 'Value')

            # make the gamma process set the sorted process set of the hidden_gamma_distr

            # process_set = prev_sim_data # this is the sorted process set of the gamma which we just ran


            normal_gamma_jump_series = []
            jump_time = list(zip(*hidden_gamma_process_set))

            normal_gamma_jump_series.append(np.random.normal(loc=mean * np.array(jump_time[0]), scale=(std) * np.sqrt(np.array(jump_time[0]))))
            self.NG_jump_series = normal_gamma_jump_series

            self.process_path = np.cumsum(normal_gamma_jump_series)
            self.time_arr = [tuple[1] for tuple in hidden_gamma_process_set]

            jumps_and_times = zip(list(normal_gamma_jump_series[0]), list(self.time_arr))

            self.sorted_process_set = sorted(jumps_and_times, key=lambda x: x[1]) #TODO: return same 3 params for both normal and gamma sims

            return self.process_path, self.time_arr #, self.sorted_process_set

    def process_endpoint_sampler(self, iterations, DistributionObject, **kwargs):

        process_endpoints = []
        for i in range(0, iterations):

            self.process_simulation(DistributionObject)

            process_endpoints.append(self.process_path[-1])
        # print('Elapsed time for Endpoint Sampling: {}'.format(t.time() - start))
        return process_endpoints



class ExtendedStateSpaceModel:

    def __init__(self, beta, kv, sigmasq, kmu, p, initial_state, flatterned_A):

        self.rng = np.random.default_rng(150)

        # self.SS_obs_rate = 1.0 / (T - t0)
        # self.sorted_obs_times = np.insert(np.cumsum(self.rng.exponential(scale=self.SS_obs_rate, size=num_obs)), 0, 0)
        #
        # self.epochs = np.insert(np.cumsum(self.rng.exponential(scale=self.SS_obs_rate, size=num_obs)), 0, 0)
        #
        # self.random_observation_times = np.cumsum(self.rng.exponential(scale=1 / 10, size=num_obs+1))

        self.X = np.array([ [initial_state[0]],
                            [initial_state[1]],
                            [initial_state[2]] ])
        self.jump_times = []
        self.jump_sizes = []

        # self.theta = theta
        self.beta = beta
        self.kv = kv
        self.sigmasq = sigmasq
        self.kmu = kmu
        self.p = p

        # predefining model matrix structure
        self.langevin_A = np.zeros((2, 2))
        self.B = np.eye(3)
        self.h = np.array([[1, 0, 0]]) # observation matrix

        # define necessary closed-form skew matrices
        self.M = None
        self.P = None
        self.Lambda = None
        self.L = None
        self.e_state_cov = None

        # storing real state evolutions and noisy_obsevrations
        self.noisy_obs = [np.array([0])]
        self.x0 = [self.X[0][0]]
        self.x1 = [self.X[1][0]]
        self.x2 = [self.X[2][0]]

        self.state_sequence = [self.X]

        self.define_langevin_A(flatterned_A)

        self.theta = self.langevin_A[1][1]

    def define_langevin_A(self, flatterned_A):
        self.langevin_A[0][0] = flatterned_A[0]
        self.langevin_A[0][1] = flatterned_A[1]
        self.langevin_A[1][0] = flatterned_A[2]
        self.langevin_A[1][1] = flatterned_A[3]

    def compute_caligA(self, end_time, start_time):
        caligA = np.block(
            [[expm(self.langevin_A * (end_time - start_time)), self.M @ np.ones(((np.shape(self.M))[1], 1))],
             [np.zeros((1, 2)), 1.]])
        return caligA

    def produce_I_matrices(self, start_time, end_time, jump_time_set):

        if len(jump_time_set) == 0:
            jump_time_set = [(0, start_time), (0, end_time)]
        # self.calculate_M_and_P_matrix(jump_time_set, end_time)
        self.calculate_P_and_M_matrices_2(jump_time_set, end_time)

        self.produce_skew_matrices(start_time, end_time, jump_time_set)

    def calculate_P_and_M_matrices_2(self, jump_time_set, end_time):

        jumps, times = np.array(jump_time_set).T
        times = times + [end_time]
        h = np.array([[0], [1]])
        # sigma_w = self.var
        sigma_w = 1  # marginalised form?

        M = np.array([[], []])
        P = np.array([[], []])

        jt_set = jump_time_set + [(0, end_time)]
        for jump_time_tuple in jt_set:
            jump = jump_time_tuple[0]
            time = jump_time_tuple[1]
            expA_times_h = expm(self.langevin_A * (end_time - time)) @ h

            M_entry = expA_times_h * jump

            M = np.append(M, M_entry, axis=1)

            P_entry = expA_times_h * np.sqrt(jump * (self.sigmasq ** 2))

            P = np.append(P, P_entry, axis=1)
        self.M = M
        self.P = P

    def produce_skew_matrices(self, start_time, end_time, jump_time_set):

        jtimes = [start_time] + [pair[1] for pair in jump_time_set] + [end_time]
        tdiffs = np.diff(jtimes)

        self.calculate_L_matrix(tdiffs)
        self.calculate_Lambda_matrix(tdiffs)

    def calculate_L_matrix(self, tdiffs):
        self.L = np.tri(len(tdiffs), len(tdiffs))

    def calculate_Lambda_matrix(self, tdiffs):
        # self.Lambda = self.sigma_mu_sq * np.diag(tdiffs)

        sigma_mu_sq = self.kmu * self.sigmasq  # the 1 should be self.var however we say var = 1 as we marginalised it
        self.Lambda = sigma_mu_sq * np.diag(tdiffs)

    def compute_noise_vector_dynamic_skew(self):
        # alpha_t = A alpha_s + B e_state

        # noise vector = e_state ~ N(0, S)
        # S = C1 + C2 : C1 = k_mu_BM * sigma_w_sq * [ [  ] [   ] [  ] ] , C2 = sigma_w_sq * [ [P I P^T 0] [0 0 0] ]
        # S = sigma_w_sq * [ C1 + C2 ]

        # 1/ compute C1
        L_lambda_L = self.L @ self.Lambda @ self.L.T
        ML_lambda_L = self.M @ L_lambda_L
        L_lambda_LM = L_lambda_L @ self.M.T

        # NOTE - got rid of additional k_mu scaling here
        C1 = np.block([[ML_lambda_L @ self.M.T, ML_lambda_L[:, [-1]]],
                                      [L_lambda_LM[-1, :], L_lambda_L[-1, -1]]])
        # 2/ compute C2
        C2 = np.block([[self.P @ np.eye(np.shape(self.P)[1]) @ self.P.T, np.zeros((2, 1))], [np.zeros((1, 3))]])

        # 3/ add them and times by sigma_w_s1 !

        # var = self.var
        var = 1
        S = var * (C1 + C2)

        self.e_state_cov = S

        # e = np.random.multivariate_normal([0,0,0],S)
        # e = np.reshape(e, (3,1))

        if self.kmu == 0: # cholesky fails as if kmu is 0, we have a 2x2 with zero padding
            try:
                cov_chol = np.linalg.cholesky(np.block([self.P @ np.eye(np.shape(self.P)[1]) @ self.P.T]))
                e = cov_chol @ np.column_stack([self.rng.normal(size=2)]) + np.zeros(
                    (2, 1))  # used to have + mean here - but now zeros
            except np.linalg.LinAlgError:
            # truncate innovation to zero if the increment is too small for Cholesky decomposition
            # print('Chol Truncated.')
                e = np.zeros((2, 1))
            e = np.append(e, 0).reshape(3,1)
        else:
            try:
                cov_chol = np.linalg.cholesky(S)
                e = cov_chol @ np.column_stack([self.rng.normal(size=3)]) + np.zeros(
                    (3, 1))  # used to have + mean here - but now zeros
            except np.linalg.LinAlgError:
                # truncate innovation to zero if the increment is too small for Cholesky decomposition
                # print('Chol Truncated.')
                e = np.zeros((3, 1))

        # e = np.random.multivariate_normal([0,0,0], S)

        return e, S

    def propagate(self, start_time, end_time):

        step_gamma_obj = GammaDistr(alpha=1, beta=self.beta)
        step_gamma_obj.set_process_conditions(t0=start_time, T=end_time, END=None, sample_size=450)
        step_gamma_sim = DistributionSimulator(step_gamma_obj)
        step_gamma_paths, step_gamma_time, step_gamma_jump_time_set = step_gamma_sim.process_simulation()

        self.produce_I_matrices(start_time, end_time, step_gamma_jump_time_set) #INVESTIGATE if step_gamma is a PATH or JUMPS

        caligA = self.compute_caligA(end_time, start_time)

        e, S = self.compute_noise_vector_dynamic_skew()

        self.X = caligA @ self.X + self.B @ e

        noisy_observation = self.h @ self.X + np.sqrt(self.sigmasq * self.kv) * self.rng.normal()

        self.noisy_obs.append(noisy_observation.flatten())
        self.x0.append(self.X[0][0])
        self.x1.append(self.X[1][0])
        self.x2.append(self.X[2][0])

        self.state_sequence.append(self.X)

    def simulate_state_space_model(self, num_obs):
        self.random_observation_times = np.cumsum(self.rng.exponential(scale=1 / 10, size=num_obs+1))

        for i in range(len(self.random_observation_times)-1):

            start_time = self.random_observation_times[i]
            end_time = self.random_observation_times[i+1]

            self.propagate(start_time, end_time)

        return self.state_sequence, self.random_observation_times, self.noisy_obs

    def apply_kalman_filtering(self):
        # APPLY KALMAN FILTERING, IF I WANTED TO SHOW IN REPORT? ie if PF not work!
        pass

    def show_plots(self):
        obs_times = self.random_observation_times

        # plt.rcParams["text.usetex"] = True  # Enable LaTeX rendering
        # plt.rcParams["font.family"] = "serif"  # Set font family to serif (for LaTeX compatibility)
        plt.rcParams["mathtext.fontset"] = "cm"  # Set MathText font to "cm"
        plt.rcParams['figure.dpi'] = 300

        # Adjust the font sizes
        plt.rcParams.update({
            'font.size': 14,  # Set axis label and title font size
            'xtick.labelsize': 11.5,  # Set x-axis tick label size
            'ytick.labelsize': 11.5  # Set y-axis tick label size
        })

        # Create a single figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 12), sharex=True)

        # Plot 1
        ax1.scatter(obs_times, self.x0, color='r', s=4, zorder=2)
        ax1.plot(obs_times, self.x0, zorder=1, linestyle='--')
        ax1.set_ylabel('$X_0, \mathrm{Position}$')
        ax1.grid(True, linewidth=0.5)

        # Plot 2
        ax2.scatter(obs_times, self.x1, color='r', s=4, zorder=2)
        ax2.plot(obs_times, self.x1, zorder=1, linestyle='--')
        ax2.set_ylabel('$X_1, \mathrm{Velocity}$')
        ax2.grid(True, linewidth=0.5)
        # ax2.set_xlabel(r'$\mathrm{Time, t}$')


        # Plot 3
        ax3.scatter(obs_times, self.x2, color='r', s=4, zorder=2)
        ax3.plot(obs_times, self.x2, zorder=1, linestyle='--')
        ax3.set_xlabel(r'$\mathrm{Time, t}$')
        ax3.set_ylabel('$X_2, \mathrm{Skew}$')
        ax3.grid(True, linewidth=0.5)
        ax3.set_xlabel(r'$\mathrm{Time, t}$')

        # Add a shared title with parameter values
        # fig.suptitle(r'$\mathrm{Evolution} $\mathrm{of}$ $X_0$, $X_1$, $\mathrm{and}$ $X_2$'+
        #              '\n' +
        #              r'$\beta$: {:.2f}, $\kappa_\mu$: {:.2f}, $\theta$: {:.2f}'.format(
        #     self.beta, self.kmu, self.theta))

        fig.suptitle(r'$\mathrm{Evolution}$ $\mathrm{of}$ $X_0$, $X_1$ $\mathrm{and}$ $X_2$'  +
                     '\n' +
                     r'$\beta$: {:.2f}, $\kappa_\mu$: {:.4f}, $\sigma^2$: {:.2f}, $\theta$: {:.2f}'.format(self.beta, self.kmu, self.sigmasq, self.theta),
                     y=0.95)

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.05)

        # plt.show()
        st.pyplot(plt)





def run_sss_sim(theta, beta, kmu, num_obs, init_skew):
    ssmodel = ExtendedStateSpaceModel(beta=beta, kv=0.01, kmu=kmu, sigmasq=1, p=1,
                                      initial_state=[0, 0, init_skew], flatterned_A=[0, 1, 0, theta])
    true_states, times, noisy_data = ssmodel.simulate_state_space_model(num_obs=num_obs)
    ssmodel.show_plots()



header = st.container()

parameter_selector = st.container()


# AMatrixSelector = st.container()
#
# HMatrixSelector = st.container()

num_obs_selector = st.container()

theta_selector = st.container()

gamma_simulation_engine = st.container()

NormalGamma_p_selector = st.container()


dynamic_skew_selector = st.container()

normal_gamma_simulaton_engine = st.container()



st.markdown(
    """
    <style>
    .main{
    background-color: #F5F5F5;
    }
    <style>
    """,
    unsafe_allow_html=True
)


with header:
    st.title('VG Langevin State Space Evolution')

# with AMatrixSelector:
#     st.markdown('**Choose A Matrix:**')
#     st.markdown('-- > Empty for Now. Feature to be built later')
#
#
# with HMatrixSelector:
#     st.markdown('**Choose h Vector:**')
#     st.markdown('-- > Empty for Now. Feature to be built later')

with num_obs_selector:
    st.markdown('**Choose Number of Observations:**')
    num_obs = st.slider('Number of Observations?', min_value=int(100), max_value=int(500), step=10)

# with theta_selector:
#     st.markdown('**Select Decay Strength in A Matrix**')
#
#     theta = st.slider('Theta Value?', min_value=float(-3), max_value=float(3), step= 0.01)

with parameter_selector:
    st.markdown('**Choose Beta and Theta value:**')

    beta_col, theta_col = st.columns(2)

    # alpha = alpha_col.slider('Alpha Value?', min_value=float(0.01), max_value=float(5), step=0.01)
    beta = beta_col.slider('Beta Value?', min_value=float(0.01), max_value=float(5), step=0.01)
    theta = theta_col.slider('Decay Rate Theta Value?', min_value=-3.000, max_value=-0.001)

with NormalGamma_p_selector:
    st.markdown('**Choose Initial Skew:**')

    # mean_col = st.columns(1)

    init_skew = st.slider('Skew (X2) Value?', min_value=float(0.0), max_value=float(5), step=0.01)
    # var = var_col.slider('Variance Value?', min_value=float(0.0), max_value=float(5), step=0.01)

with dynamic_skew_selector:
    st.markdown('**Choose dynamic skew parameter, k_mu:**')

    kmu = st.slider('Dynamic Skew Coefficient Value?', min_value=float(0), max_value=float(1), step= 0.0001)


with normal_gamma_simulaton_engine:
    if st.button('Click for SDE Forward Simulation'):


        run_sss_sim(theta, beta, kmu, num_obs, init_skew)
        # RunStateSpaceSimulation(beta, samples, mean, var, theta, obs)


        # ssmodel = ExtendedStateSpaceModel(beta=beta, kv=0.01, kmu=kmu, sigmasq=1, p=1,
        #                                   initial_state=[0, 0, 0], flatterned_A=[0, 1, 0, theta])
        # true_states, times, noisy_data = ssmodel.simulate_state_space_model(num_obs=105)

