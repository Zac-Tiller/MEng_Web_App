import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
from scipy.special import gamma as gamma_func

from scipy.special import kv

import streamlit as st

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
            y2 = norm.pdf(x, loc=0, scale=1)
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
        st.pyplot(plt)

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

import matplotlib.pyplot as plt

def plotter(times, paths, title, x_lbl, y_lbl, alpha, beta):
    plt.rcParams["mathtext.fontset"] = "cm"  # Set MathText font to "cm"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['figure.dpi'] = 300

    plt.figure(99)
    for x, y in zip(times, paths):
        plt.step(x, y)

    formatted_title = r'$\mathrm{' + title.replace(' ', r'\ ') + r' \alpha = ' + str(alpha) + r', \beta = ' + str(
        beta) + r'}$'

    plt.title(formatted_title, fontsize=14)  # Set title font size with spaces

    plt.xlabel(r'$\mathrm{' + x_lbl + '}$', fontsize=14)  # Re-format x-axis label as LaTeX expression
    plt.ylabel(r'$\mathrm{' + y_lbl + '}$', fontsize=14)  # Re-format y-axis label as LaTeX expression
    plt.grid(True, linewidth=0.5)  # Add grid

    plt.tick_params(axis='x', labelsize=10)  # Set x-axis tick label font size
    plt.tick_params(axis='y', labelsize=10)

    plt.show()
    return plt


def ngplotter(times, paths, title, x_lbl, y_lbl, alpha, beta, mu, sigmasq):
    plt.rcParams["mathtext.fontset"] = "cm"  # Set MathText font to "cm"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['figure.dpi'] = 300

    plt.figure(99)
    for x, y in zip(times, paths):
        plt.step(x, y)

    formatted_title = r'$\mathrm{' + title.replace(' ', r'\ ') + r', \alpha = ' + str(alpha) + r', \beta = ' + str(beta) + r', \mu = ' + str(mu) + r', \sigma^2 = ' + str(sigmasq) + r'}$'


    plt.title(formatted_title, fontsize=14)  # Set title font size with spaces

    plt.xlabel(r'$\mathrm{' + x_lbl + '}$', fontsize=14)  # Re-format x-axis label as LaTeX expression
    plt.ylabel(r'$\mathrm{' + y_lbl + '}$', fontsize=14)  # Re-format y-axis label as LaTeX expression
    plt.grid(True, linewidth=0.5)  # Add grid

    plt.tick_params(axis='x', labelsize=10)  # Set x-axis tick label font size
    plt.tick_params(axis='y', labelsize=10)

    plt.show()
    return plt












# The longer the simulation, the more samples we need !
# JJs method of gsamps works well for intervals t = 0 -> 1, but, when T gets large, divergence occurs so need to increase
# sample size (which is why i x by 2/3 T in set_process_conditions)


def gamma_paths(alpha, beta, T):

    gamma_obj = GammaDistr(alpha=alpha, beta=beta)
    gamma_obj.set_process_conditions(t0=0, T=T, END=None, sample_size=300)  # sample_size refers to number of random times we generate

    gamma_sim = DistributionSimulator(gamma_obj) # create our simulation
    path, time, jump_time_set = gamma_sim.process_simulation() #gamma_obj) # run the simulation. after this comment, the gamma_sim object has the jump_time_sets (sorted process set) as an attribute
    path2, time2, jump_time_set2 = gamma_sim.process_simulation()
    path3, time3, jump_time_set3 = gamma_sim.process_simulation()
    path4, time4, jump_time_set4 = gamma_sim.process_simulation()
    path5, time5, jump_time_set5 = gamma_sim.process_simulation()
    path6, time6, jump_time_set6 = gamma_sim.process_simulation()
    path7, time7, jump_time_set7 = gamma_sim.process_simulation()
    path8, time8, jump_time_set8 = gamma_sim.process_simulation()

    times = [time, time2, time3, time4, time5, time6, time7, time8]
    paths = [path, path2, path3, path4, path5, path6, path7, path8]

    plt = plotter(times, paths, 'Gamma Process Path: ', 'Time, t', '\Gamma_t', alpha, beta)
    st.pyplot(plt)


def gamma_hist(alpha, beta, T):
    gamma_obj = GammaDistr(alpha=alpha, beta=beta)
    gamma_obj.set_process_conditions(t0=0, T=T, END=None,
                                     sample_size=300)  # sample_size refers to number of random times we generate

    gamma_sim = DistributionSimulator(gamma_obj)

    fig, ax = gamma_sim.plot_simulation_distribution(gamma_sim.process_endpoint_sampler(10_000, gamma_obj), T)


def ng_path(alpha, beta, mu, sigmasq, T):

    gamma_for_ng_obj = GammaDistr(alpha=alpha, beta=beta)
    gamma_for_ng_obj.set_process_conditions(t0=0, T=T, END=None, sample_size=300)

    normal_obj = NormalDistr(mean=mu, std=np.sqrt(sigmasq), secondary_distr=gamma_for_ng_obj) # We MUST define the secondary distribution of this NG sim ie. define the gamma distr to use

    normal_gamma_sim = DistributionSimulator(normal_obj)
    path, time = normal_gamma_sim.process_simulation()
    path2, time2 = normal_gamma_sim.process_simulation()
    path3, time3 = normal_gamma_sim.process_simulation()
    path4, time4 = normal_gamma_sim.process_simulation()
    path5, time5 = normal_gamma_sim.process_simulation()
    path6, time6 = normal_gamma_sim.process_simulation()
    path7, time7 = normal_gamma_sim.process_simulation()
    path8, time8 = normal_gamma_sim.process_simulation()

    times = [time, time2, time3, time4, time5, time6, time7, time8]
    paths = [path, path2, path3, path4, path5, path6, path7, path8]

    plt = ngplotter(times, paths, 'VG Process Path', 'Time, t', '\mathcal{VG}', alpha, beta, mu, sigmasq)
    st.pyplot(plt)


def ng_hist(alpha, beta, mu, sigmasq, T):
    gamma_for_ng_obj = GammaDistr(alpha=alpha, beta=beta)
    gamma_for_ng_obj.set_process_conditions(t0=0, T=T, END=None, sample_size=300)

    normal_obj = NormalDistr(mean=mu, std=np.sqrt(sigmasq),
                             secondary_distr=gamma_for_ng_obj)  # We MUST define the secondary distribution of this NG sim ie. define the gamma distr to use

    normal_gamma_sim = DistributionSimulator(normal_obj)

    fig, ax = normal_gamma_sim.plot_simulation_distribution(normal_gamma_sim.process_endpoint_sampler(10_000, normal_obj), T)




header = st.container()

parameter_selector = st.container()

gamma_simulation_engine = st.container()

NormalGamma_param_selector = st.container()

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
    st.title('Interactive Simulation of a Levy Process!')

with parameter_selector:
    st.header('Gamma Process')

    st.markdown('**Choose parameters of the Gamma distribution, alpha and beta, and time:**')

    alpha_col, beta_col, T_col = st.columns(3)

    alpha = alpha_col.slider('Alpha Value?', min_value=float(0.01), max_value=float(5), step=0.01)
    beta = beta_col.slider('Beta Value?', min_value=float(0.01), max_value=float(5), step=0.01)
    T = T_col.slider('Time?', min_value=int(10), max_value=int(30), step=int(1))
    # samples = sample_col.slider('Number of Samples?', min_value=1, max_value=1000)

    # gamma_obj = GammaDistr(alpha, beta)
    # gamma_obj.set_process_conditions(T=1, sample_size=samples)
    # gamma_sim = DistributionSimulator(gamma_obj)

with gamma_simulation_engine:

    show_path_col, show_samples_col = st.columns(2)

    with show_path_col:
        if st.button('Show Process Path', key="GProcess"):
            gamma_paths(alpha, beta, T)



            # fig, ax = gamma_sim.plot_simulation_distribution(gamma_sim.process_endpoint_sampler(10000, gamma_obj))
            #
            # # process_endpoints = sample_gamma_process(alpha, beta, samples, iterations=10000)
            # # fig, ax = plot_process_endpoints_samples(process_endpoints, gamma_distribution=False)
            #
            # st.pyplot(fig)
            # st.pyplot(ax)

    with show_samples_col:
        text_container = st.container
        if st.button('Show Samples', key="GHist"):
            gamma_hist(alpha, beta, T)

            # path, times, jumps = gamma_process_sim(alpha, beta, samples, T=1)
            # fig, ax = streamlit_plotter(times, path) #, 'Gamma Process Simulation', 'Time', 'Path')


            # path, time, jump_time_set = gamma_sim.process_simulation(gamma_obj)  # run the simulation. after this comment, the gamma_sim object has the jump_time_sets (sorted process set) as an attribute
            # fig = plotter(time, path, 'Gamma Process Simulation TEST', 'Time', 'Value')
            # st.pyplot(fig)

with NormalGamma_param_selector:
    st.header('Normal-Gamma (VG) Process')
    st.markdown('**Choose Extra parameters for the Normal-Gamma, mean and var:**')

    mean_col, var_col = st.columns(2)

    mean = mean_col.slider('Mean Value?', min_value=float(0.0), max_value=float(5), step=0.01)
    var = var_col.slider('Variance Value?', min_value=float(0.0), max_value=float(5), step=0.01)

with normal_gamma_simulaton_engine:
    show_path_col, show_samples_col = st.columns(2)

    with show_path_col:
        if st.button('Show Process Path', key="NGProcess"):
            alpha = float(1) # we want the gamma subordinator process to measure 1 as the rate of passing of time
            ng_path(alpha, beta, mean, var, T)

    with show_samples_col:
        if st.button('Show Samples', key="NGHist"):
            alpha = float(1)
            ng_hist(alpha, beta, mean, var, T)


