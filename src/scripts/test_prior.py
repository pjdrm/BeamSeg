'''
Created on Apr 12, 2018

@author: pjdrm
'''
import numpy as np
import toyplot
import toyplot.browser
import toyplot.pdf
import toyplot.png
import toyplot.svg
import scipy.special as scipy
from click.termui import style

def draw_theta_topics(beta_t, theta_t_mean, K):
    theta_t = []
    for k in range(K):
        theta_t_k = np.random.dirichlet(beta_t[k]*theta_t_mean[k])
        theta_t.append(theta_t_k)
    return theta_t

def draw_words(K, W, topic_draws, theta_t):
    N_t_k_w = np.zeros((K, W))
    for k in range(K):
        for z in range(topic_draws[k]):
            word = np.random.multinomial(1, theta_t[k])
            wi = np.nonzero(word)[0][0] #w is a vocabulary index
            N_t_k_w[k, wi] += 1.0
           
    return N_t_k_w

def update_alpha(phi_t_mean, topic_draws, alpha_t):
    n_draws = np.sum(topic_draws)
    num_alpha = np.sum(phi_t_mean*(scipy.digamma(topic_draws+alpha_t*phi_t_mean)-scipy.digamma(alpha_t*phi_t_mean)), axis=0)
    denom_alpha = scipy.digamma(n_draws+alpha_t)-scipy.digamma(alpha_t)
    alpha_t_update = alpha_t*(num_alpha/denom_alpha)
    return alpha_t_update

def update_phi(topic_draws, alpha_t, phi_t_mean):
    n_draws = np.sum(topic_draws)
    num_phi_t_mean = topic_draws+alpha_t*phi_t_mean
    denom_phi_t_mean = n_draws+alpha_t
    phi_t_mean_update = num_phi_t_mean/denom_phi_t_mean
    return phi_t_mean_update

def update_beta(theta_t_mean_k, N_t_k_w_k, beta):
    num_beta = np.sum(theta_t_mean_k*(scipy.digamma(N_t_k_w_k+beta*theta_t_mean_k)-scipy.digamma(beta*theta_t_mean_k)), axis=0)
    denom_beta = scipy.digamma(np.sum(N_t_k_w[k])+beta)-scipy.digamma(beta)
    if num_beta == 0.0 or denom_beta == 0.0:
        return beta
    beta_t_update = beta*(num_beta/denom_beta)
    return beta_t_update

def update_theta(N_t_k_w_k, beta_t_k, theta_t_mean_k):
    num_theta_t_mean = N_t_k_w_k+beta_t_k*theta_t_mean_k
    denom_theta_t_mean = np.sum(N_t_k_w_k)+beta_t_k
    theta_t_mean_update = num_theta_t_mean/denom_theta_t_mean
    return theta_t_mean_update
    
use_seed = False
seed = 232#84
if use_seed:
    np.random.seed(seed)
        
n_draws = 100
W = 10
K = 5
n_chunks = 10
base_prior = 10.0

alpha_base = 10
alpha_t = [alpha_base]*n_chunks

phi_t_init = np.random.dirichlet([base_prior]*K)
topic_draws = np.random.multinomial(n_draws, phi_t_init, size=1)[0]
alpha_smooth = 0.001
#phi_t_mean = (topic_draws+alpha_smooth)/(np.sum(topic_draws*1.0)+alpha_smooth*K)

#phi_t_mean = np.random.dirichlet([base_prior]*K)
phi_t_mean = np.array([base_prior]*K)/alpha_t[0]
phi_t = phi_t_mean

beta_t = [base_prior]*K
theta_t_mean = [np.array([1.0]*W) for k in range(K)]
theta_t = draw_theta_topics(beta_t, theta_t_mean, K)

all_topic_prop_draws = []
for t in range(n_chunks):
    #print(phi_t_mean)
    #print("alpha: " + str(alpha_t))
    
    all_topic_prop_draws.append(phi_t)
    
    theta_t = draw_theta_topics(beta_t, theta_t_mean, K)
    N_t_k_w = draw_words(K, W, topic_draws, theta_t)
    
    #alpha_t = update_alpha(phi_t_mean, topic_draws, alpha_t)
    phi_t_mean = update_phi(topic_draws, alpha_t[t], phi_t_mean)
    
    phi_t = np.random.dirichlet(alpha_t[t]*phi_t_mean)
    topic_draws = np.random.multinomial(n_draws, phi_t, size=1)[0]
    
    for k, beta in enumerate(beta_t):
        beta_t[k] = update_beta(theta_t_mean[k], N_t_k_w[k], beta)
        theta_t_mean[k] = update_theta(N_t_k_w[k], beta_t[k], theta_t_mean[k])
    
rnd_topics = []
for t in range(n_chunks):
    rnd_topics.append(np.random.dirichlet([base_prior]*K))
    
canvas = toyplot.Canvas(width=350, height=250)
axes = canvas.cartesian(xlabel="Topics", ylabel="Word Probabilities")
axes.bars(rnd_topics)
#toyplot.browser.show(canvas)
toyplot.svg.render(canvas, "independent_prior.svg")

canvas = toyplot.Canvas(width=350, height=250)
axes = canvas.cartesian(xlabel="Topics", ylabel="Word Probabilities")
axes.bars(all_topic_prop_draws[1:])
#toyplot.browser.show(canvas)
toyplot.pdf.render(canvas, "dynamic_prior.pdf")
