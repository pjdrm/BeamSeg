'''
Created on Apr 12, 2018

@author: pjdrm
'''
import numpy as np
import toyplot
import toyplot.browser
import scipy.special as scipy

def draw_theta_topics(beta_t, theta_t_mean, K):
    theta_t = []
    for k in range(K):
        theta_t_k = np.random.dirichlet(beta_t[k]*theta_t_mean[k])
        theta_t.append(theta_t_k)
    return theta_t

def draw_words(K, W, topic_draws, theta_t):
    N_t_k_w = np.zeros((K, W))
    word_counts_t = np.zeros(W)
    for k in range(K):
        for z in range(topic_draws[k]):
            word = np.random.multinomial(1, theta_t[k])
            wi = np.nonzero(word)[0][0] #w is a vocabulary index
            word_counts_t[wi] += 1.0
            N_t_k_w[k, wi] += 1.0
           
    return N_t_k_w

def update_alpha(phi_t_mean, topic_draws, alpha_t):
    n_draws = np.sum(topic_draws)
    num_alpha = np.sum(phi_t_mean*(scipy.digamma(topic_draws+alpha_t*phi_t_mean)+scipy.digamma(alpha_t*phi_t_mean)), axis=0)
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
    
n_draws = 100
W = 10
K = 5
n_chunks = 15
base_prior = 0.2
alpha_t = base_prior
phi_t_mean = np.array([1.0]*K)

beta_t = [base_prior]*K
theta_t_mean = [np.array([1.0]*W) for k in range(K)]
theta_t = draw_theta_topics(beta_t, theta_t_mean, K)

all_topic_prop_draws = []
for t in range(n_chunks):
    phi_t = np.random.dirichlet(alpha_t*phi_t_mean)
    all_topic_prop_draws.append(phi_t)
    topic_draws = np.random.multinomial(n_draws, phi_t, size=1)[0]
    
    theta_t = draw_theta_topics(beta_t, theta_t_mean, K)
    N_t_k_w = draw_words(K, W, topic_draws, theta_t)
    
    alpha_t = update_alpha(phi_t_mean, topic_draws, alpha_t)
    phi_t_mean = update_phi(topic_draws, alpha_t, phi_t_mean)


    for k, beta in enumerate(beta_t):
        beta_t[k] = update_beta(theta_t_mean[k], N_t_k_w[k], beta)
        theta_t_mean[k] = update_theta(N_t_k_w[k], beta_t[k], theta_t_mean[k])
    
rnd_topics = []
for t in range(n_chunks):
    rnd_topics.append(np.random.dirichlet([base_prior]*K))
    
canvas = toyplot.Canvas(width=1500, height=500)
axes = canvas.cartesian(label="Random Topics", margin=100)
axes.bars(rnd_topics)
toyplot.browser.show(canvas)

canvas = toyplot.Canvas(width=1500, height=500)
axes = canvas.cartesian(label="Dynamic Topics", margin=100)
axes.bars(all_topic_prop_draws)
toyplot.browser.show(canvas)