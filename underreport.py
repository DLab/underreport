import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from scipy import special
from scipy.stats import lognorm
import gpflow
from gpflow.ci_utils import ci_niter
from gpflow import set_trainable
import requests
import json
import sys, getopt
import datetime

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
gpflow.config.set_default_summary_fmt("notebook")
# convert to float64 for tfp to play nicely with gpflow in 64
f64 = gpflow.utilities.to_default_float
tf.random.set_seed(123)

def number_to_date(number):
    if isinstance(number,int):
            number = [number]
    if len(number) > 1:
        date_list = []

        for num in number:
            starting_date =  datetime.date(2020, 1, 1)
            date = starting_date + datetime.timedelta(days=num)
            date_list.append(date.strftime("%Y-%m-%d"))
        date = date_list

    else:
        starting_date =  datetime.date(2020, 1, 1)
        date = starting_date + datetime.timedelta(days=number[0])
        date = date.strftime("%Y-%m-%d")
    return date

def date_to_number(dates):

    if len(dates) == 1:
        yyyy, mm, dd = dates.split("-")
        yyyy = int(yyyy)
        mm = int(mm)
        dd = int(dd)

        starting_date =  datetime.date(2020, 1, 1)
        query_date = datetime.date(yyyy, mm, dd)
        days_passed = query_date-starting_date
        days_passed = int(days_passed.days)

    else:
        days_list = []

        for date in dates:
            yyyy, mm, dd = date.split("-")
            yyyy = int(yyyy)
            mm = int(mm)
            dd = int(dd)

            starting_date =  datetime.date(2020, 1, 1)
            query_date = datetime.date(yyyy, mm, dd)
            days_passed = query_date-starting_date
            days_passed = int(days_passed.days)
            days_list.append(days_passed)
        days_passed = days_list

    return days_passed

def underreport_estimate(cases, deaths):
    pass
def get_regional_deaths(region):
    endpointm = requests.get('http://192.168.2.223:5006/getDeathsByState?state='+str(region))
    deaths = json.loads(endpointm.text)
    deaths = pd.DataFrame(deaths)
    deaths.index = pd.to_datetime(deaths.dates)
    deaths.drop(columns = 'dates', inplace= True)
    deaths.index = [x.strftime("%Y-%m-%d") for x in deaths.index]
    deaths['total'] = deaths['confirmed'] +deaths['suspected']
    deaths['total'] = deaths['total']
    return deaths

def get_nacional_deaths():
    endpointm = requests.get('http://192.168.2.223:5006/getDeathsByState?state=16')
    deaths = json.loads(endpointm.text)
    deaths = pd.DataFrame(deaths)
    deaths.index = pd.to_datetime(deaths.dates)
    deaths.drop(columns = 'dates', inplace= True)
    deaths.index = [x.strftime("%Y-%m-%d") for x in deaths.index]

    for i in range(1,16):
        endpointm = requests.get('http://192.168.2.223:5006/getDeathsByState?state='+str(i))
        deaths2 = json.loads(endpointm.text)
        deaths2 = pd.DataFrame(deaths2)
        deaths2.index = pd.to_datetime(deaths2.dates)
        deaths2.drop(columns = 'dates', inplace= True)
        deaths2.index = [x.strftime("%Y-%m-%d") for x in deaths2.index]
        deaths = deaths+ deaths2
    deaths['total'] = deaths['confirmed'] +deaths['suspected']
    deaths['total'] = deaths['total']
    return deaths

def delay_correction():
    mu, sigma = 13, 12.7  #?
    mean = np.log(mu**2 / np.sqrt(mu**2 + sigma**2) )
    std =  np.sqrt(np.log(1 + sigma**2/mu**2)  )
    f = lognorm(s = std, scale = np.exp(mean))
    days = 15

    pass

def sissor(x):
    return x[:10]

def args_parser(argv):
   from_region = ''
   to_region = ''
   try:
      opts, args = getopt.getopt(argv,"hf:t:n:",["from=","to=", "gpun"])
   except getopt.GetoptError:
      print( 'underreport.py -f <regionnumber> -t <regionnumber> -n <gpunumber>')
      sys.exit(2)
   for opt, arg in opts:
        if opt == '-h':
            print( 'underreport.py -f <regionnumber> -t <regionnumber> -n <gpunumber> ')

            sys.exit()
        elif opt in ("-f", "--ifile"):
            from_region = arg
        elif opt in ("-t", "--ofile"):
            to_region = arg
        elif opt in ("-n", "--ofile"):
            ngpu = arg

   return from_region, to_region, ngpu


endpointnew = requests.get('http://192.168.2.223:5006/getNewCasesAllStates')
actives = json.loads(endpointnew.text)
dates = pd.to_datetime(actives['dates'])
dates = [x.strftime("%Y-%m-%d") for x in dates]
# Parsing args
from_region, to_region, gpun =  args_parser(sys.argv[1:])


ssess =tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))


with tf.device('/gpu:'+gpun):


    for current_region in range(int(from_region),int(to_region)+1):

            if current_region == 17:
                endpointnew = requests.get('http://192.168.2.223:5006/getNationalNewCases')
                actives = json.loads(endpointnew.text)
                dates = pd.to_datetime(actives['dates'])
                dates = [x.strftime("%Y-%m-%d") for x in dates]
                reg_active = pd.DataFrame(data = {'cases': actives['cases']}, index = dates)


                endpointm = requests.get('http://192.168.2.223:5006/getDeathsByState?state=16')
                deaths = json.loads(endpointm.text)
                deaths = pd.DataFrame(deaths)
                deaths.index = pd.to_datetime(deaths.dates)
                deaths.drop(columns = 'dates', inplace= True)
                deaths.index = [x.strftime("%Y-%m-%d") for x in deaths.index]

                for i in range(1,16):
                    endpointm = requests.get('http://192.168.2.223:5006/getDeathsByState?state='+str(i))
                    deaths2 = json.loads(endpointm.text)
                    deaths2 = pd.DataFrame(deaths2)
                    deaths2.index = pd.to_datetime(deaths2.dates)
                    deaths2.drop(columns = 'dates', inplace= True)
                    deaths2.index = [x.strftime("%Y-%m-%d") for x in deaths2.index]
                    deaths = deaths+ deaths2
                deaths['total'] = deaths['confirmed'] +deaths['suspected']
                deaths = deaths.query("total > 0")
            else:
                padded_region = '{:02d}'.format(current_region)
                reg_active = pd.DataFrame(data = {'cases': actives['data'][padded_region]}, index = dates)


                endpointm = requests.get('http://192.168.2.223:5006/getDeathsByState?state='+str(current_region))
                deaths = json.loads(endpointm.text)
                deaths = pd.DataFrame(deaths)



                deaths.index = pd.to_datetime(deaths.dates)
                deaths.drop(columns = 'dates', inplace= True)
                deaths.index = [x.strftime("%Y-%m-%d") for x in deaths.index]

                deaths['total'] = deaths['confirmed'] +deaths['suspected']
                deaths = deaths.query("total > 0")


            common = list(set(deaths.index.to_list()).intersection(reg_active.index.to_list()))
            # min. nummber of datapoints to compute
            if len(common) <30:
                continue

            common = sorted(common)
            mu, sigma = 13, 12.7  #?
            mean = np.log(mu**2 / np.sqrt(mu**2 + sigma**2) )
            std =  np.sqrt(np.log(1 + sigma**2/mu**2)  )
            f = lognorm(s = std, scale = np.exp(mean))
            days = 15

            RM_ac = reg_active
            RM_ac = RM_ac.loc[common]
            RM_deaths = deaths['total'].loc[common]

            d_cases = np.empty((RM_ac.shape[0],1))
            for i in range(RM_ac.shape[0]):
                until_t_data = RM_ac.values[:i+1]
                reversed_arr = until_t_data[::-1].reshape(i+1)
                d_cases[i] = np.sum(f.pdf(np.linspace(0,i,i+1)) * reversed_arr)


            dcfr = RM_deaths.values/d_cases.reshape(len(common))
            estimator_a = pd.read_csv('adjusted_cfr.csv').iloc[current_region-1]['cfr_mid']/(dcfr*100)
            estimator_a = estimator_a[:-1]
            common = common[:-1]

            pro_a = special.expit(estimator_a) #logit
            numeric_common = date_to_number(common)
            X = np.asarray(numeric_common)
            X = tf.convert_to_tensor(X.reshape(estimator_a.shape[0],-1), dtype=tf.float64)
            pro_a = pro_a
            pro_a = tf.convert_to_tensor(pro_a.reshape(estimator_a.shape[0],-1))
            data = (X, pro_a)

            kernel = gpflow.kernels.SquaredExponential()
            mean_function = gpflow.mean_functions.Constant(f64(0.5))#c=1 , mean_function
            model = gpflow.models.GPR(data, kernel, mean_function = mean_function, noise_variance=0.01)
            optimizer = gpflow.optimizers.Scipy()
            optimizer.minimize(model.training_loss, model.trainable_variables)

            model.kernel.variance.prior  = tfd.LogNormal(f64(1.0), f64(1.0))
            model.kernel.lengthscales.prior  = tfd.LogNormal(f64(4.0), f64(0.5))

            model.likelihood.variance.prior  = tfd.HalfNormal( f64(0.5))
            model.mean_function.c.prior  = tfd.Uniform(f64(0.0), f64(1.0))

            num_burnin_steps = ci_niter(1000)
            num_samples = ci_niter(10000)

            hmc_helper = gpflow.optimizers.SamplingHelper(
                model.log_posterior_density, model.trainable_parameters
            )

            hmc = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, step_size=.01
            )
            adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
                hmc, num_adaptation_steps=10, target_accept_prob=f64(0.75), adaptation_rate=0.1
            )


            @tf.function
            def run_chain_fn():
                return tfp.mcmc.sample_chain(
                    num_results=num_samples,
                    num_burnin_steps=num_burnin_steps,
                    current_state=hmc_helper.current_state,
                    kernel=adaptive_hmc,#hmc
                    trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
                )


            samples, traces = run_chain_fn()
            parameter_samples = hmc_helper.convert_to_constrained_values(samples)

            param_to_name = {param: name for name, param in gpflow.utilities.parameter_dict(model).items()}
            xx2 = np.linspace(numeric_common[0], numeric_common[-1], numeric_common[-1]-numeric_common[0]+1)[:, None]
            posterior_samples = []

            for i in range(0, 10000):#num_samples
                for var, var_samples in zip(hmc_helper.current_state, samples):
                    var.assign(var_samples[i])
                f = model.predict_f_samples(xx2, 1)
                posterior_samples.append( f[0, :, :])

            posterior_samples = np.hstack(posterior_samples)
            posterior_samples = posterior_samples.T

            mean = special.logit(np.mean(posterior_samples, 0))
            low = np.percentile(special.logit(posterior_samples), 0.13, axis=0)
            high = np.percentile(special.logit(posterior_samples), 99.87, axis=0)

            final_data = pd.DataFrame(index = number_to_date([i for i in range(numeric_common[0],numeric_common[-1]+1)]))
            final_data['mean'] = mean
            final_data['low'] = low
            final_data['high'] = high
            final_data.to_csv('output/'+str(current_region)+'.csv')
