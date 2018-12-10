def calc_likelihood(simulated_flat, hists_flat):
    simulated = unflatten_data(simulated_flat, shape)
    hists = unflatten_data(hists_flat, shape)
    tot_score = 0
    for sim, hist in zip(simulated, hists):
        kde = KernelDensity(bandwidth=bw)
        kde.fit(sim)
        tot_score += kde.score(hist)
    return tot_score

def get_gid_data(gids):
    return {pseudo:{gid:pseudo_data_correct[pseudo][gid] for gid in gids} for pseudo in pseudo_data_correct}

def flatten_data(hists):
    shape_vec = []
    hists_vec = np.array([])
    for hist in hists:
        shape_vec.append(hist.shape)
        hists_vec = np.append(hists_vec, hist.flatten())
    return(hists_vec,shape_vec)

def unflatten_data(hists_vec, shape_vec):
    start = 0
    end = 0
    hists = []
    for shape in shape_vec:
        end += shape[0]*shape[1]
        hists.append(np.reshape(hists_vec[start:end],shape))
        start += shape[0]*shape[1]
    return hists

def plot_models(history):
    for i in history.alive_models(history.max_t):
        df, w = history.get_distribution(m=i)
        df_copy = df.copy()
        model = model_names[i]
        for param in αβ_params[model]:
            a, b = αβ_lims[model][param]
            df[param] = df_copy.apply(lambda x: st.beta.mean(x[param], x[param+"_sd"])*(b-a) + a, axis=1)
            df[param+"_sd"] = df_copy.apply(lambda x: st.beta.std(x[param], x[param+"_sd"])*(b-a), axis=1)
        plot = plot_kde_matrix(df, w, limits=αβ_param_spaces[model_names[i]])
        plot.savefig("fig/SSE/abc-beta" + str(n_particles) + "-" + str(max_pops)+ "-bw" + str(bw) + "-"  + pseduo_pop  + "-" + model_names[i] + ".png")
    plt.close("all")


αβ_lims = dict()
αβ_lims["LK"] = {"τ":(0,4), "p":(0,1)}
αβ_lims["LLK"] = {"τ":(0,4), "p":(0,1), "λ":(0,10)}
αβ_lims["PCHM"] = {"τ":(0,4), "p":(0,1)}
αβ_lims["LPCHM"] = {"τ":(0,4), "p":(0,1), "λ":(0,10)}
αβ_lims["JPCHM"] = {"τ":(0.,4.), "p":(0,1)}
αβ_lims["LJPCHM"] = {"τ":(0,4.), "p":(0,1), "λ":(0,10)}
αβ_lims["L1PCHM"] = {"τ":(0,4.), "p":(0,1), "λ":(0,10)}
αβ_lims["BR"] = {"ε":(0,1), "p":(0,1)}
αβ_lims["LBR"] = {"ε":(0,1), "p":(0,1), "λ":(0,10)}
αβ_lims["EWA"] = {"p":(0,1), "λ":(0,10), "φ":(0,1), "ρ":(0,1), "δ":(0,1)}


def LK_model(parameters):
    p = parameters["p"]
    τ = parameters["τ"]
    p_sd = parameters["p_sd"]
    τ_sd = parameters["τ_sd"]
    if "init_τ" in parameters:
        init_params = np.array([parameters["init_τ"], parameters["init_λ"]])
    else:
         init_params=np.array([1.5,1.])
    hists = []
    for gid in gids:
        pop_LK = Population(games[gid][0], games[gid][1], rounds, p1_size,  p2_size, init_params=init_params, params_vec=[p,τ], σ_vec=[p_sd, τ_sd], lower_vec=[0.,0.], upper_vec=[1.,4.], random_params=True)
        hists.append(flatten_h(pop_LK.mul_runs_LK(n_runs)))
    flat_hists, shape = flatten_data(hists)
    return {"data": flat_hists, "shape": shape}


def LLK_model(parameters):
    p = parameters["p"]
    τ = parameters["τ"]
    λ = parameters["λ"]
    p_sd = parameters["p_sd"]
    τ_sd = parameters["τ_sd"]
    λ_sd = parameters["λ_sd"]
    hists = []
    for gid in gids:
        pop_LLK = Population(games[gid][0], games[gid][1], rounds, p1_size,  p2_size, params_vec=[p,τ, λ], σ_vec=[p_sd, τ_sd, λ_sd], lower_vec=[0.,0., 0.], upper_vec=[1.,4., 10.], random_params=True)
        hists.append(flatten_h(pop_LLK.mul_runs_LLK(n_runs)))
    flat_hists, shape = flatten_data(hists)
    return {"data": flat_hists, "shape":shape}

def PCHM_model(parameters):
    p = parameters["p"]
    τ = parameters["τ"]
    p_sd = parameters["p_sd"]
    τ_sd = parameters["τ_sd"]
    if "init_τ" in parameters:
        init_params = np.array([parameters["init_τ"], parameters["init_λ"]])
    else:
         init_params=np.array([1.5,1.])
    hists = []
    for gid in gids:
        pop_PCHM = Population(games[gid][0], games[gid][1], rounds, p1_size,  p2_size,  init_params=init_params, params_vec=[p,τ], σ_vec=[p_sd, τ_sd], lower_vec=[0.,0.], upper_vec=[1.,4.], random_params=True)
        hists.append(flatten_h(pop_PCHM.mul_runs_PCHM(n_runs)))
    flat_hists, shape = flatten_data(hists)
    return {"data": flat_hists, "shape": shape}

def JPCHM_model(parameters):
    p = parameters["p"]
    τ = parameters["τ"]
    p_sd = parameters["p_sd"]
    τ_sd = parameters["τ_sd"]
    if "init_τ" in parameters:
        init_params = np.array([parameters["init_τ"], parameters["init_λ"]])
    else:
         init_params=np.array([1.5,1.])
    hists = []
    for gid in gids:
        pop_JPCHM = Population(games[gid][0], games[gid][1], rounds, p1_size,  p2_size,  init_params=init_params, params_vec=[p,τ], σ_vec=[p_sd, τ_sd], lower_vec=[0.,0.], upper_vec=[1.,αβ_lims["JPCHM"]["τ"][1]], random_params=True)
        hists.append(flatten_h(pop_JPCHM.mul_runs_JPCHM(n_runs)))
    flat_hists, shape = flatten_data(hists)
    return {"data": flat_hists, "shape": shape}


def LPCHM_model(parameters):
#     p = parameters["p"]
    τ = parameters["τ"]
    λ = parameters["λ"]
#     p_sd = parameters["p_sd"]
    τ_sd = parameters["τ_sd"]
    λ_sd = parameters["λ_sd"]
    if "init_τ" in parameters:
        init_params = np.array([parameters["init_τ"], parameters["init_λ"]])
    else:
         init_params=np.array([1.5,1.])
    hists = []
    for gid in gids:
        pop_LPCHM = Population(games[gid][0], games[gid][1], rounds, p1_size,  p2_size,  init_params=init_params, params_vec=[τ, λ], σ_vec=[τ_sd, λ_sd], lower_vec=[0., 0.], upper_vec=[4., 10.], random_params=True)
        hists.append(flatten_h(pop_LPCHM.mul_runs_LPCHM(n_runs)))
    flat_hists, shape = flatten_data(hists)
    return {"data": flat_hists, "shape": shape}


def LJPCHM_model(parameters):
    p = parameters["p"]
    τ = parameters["τ"]
    λ = parameters["λ"]
    p_sd = parameters["p_sd"]
    τ_sd = parameters["τ_sd"]
    λ_sd = parameters["λ_sd"]
    if "init_τ" in parameters:
        init_params = np.array([parameters["init_τ"], parameters["init_λ"]])
    else:
         init_params=np.array([1.5,1.])
    hists = []
    for gid in gids:
        pop_LJPCHM = Population(games[gid][0], games[gid][1], rounds, p1_size,  p2_size,  init_params=init_params, params_vec=[p, τ, λ], σ_vec=[p_sd, τ_sd, λ_sd], lower_vec=[0.,0., 0.], upper_vec=[1.,4., 10.], random_params=True)
        hists.append(flatten_h(pop_LJPCHM.mul_runs_LJPCHM(n_runs)))
    flat_hists, shape = flatten_data(hists)
    return {"data": flat_hists, "shape": shape}



def L1PCHM_model(parameters):
    p = parameters["p"]
    τ = parameters["τ"]
    λ = parameters["λ"]
    p_sd = parameters["p_sd"]
    τ_sd = parameters["τ_sd"]
    λ_sd = parameters["λ_sd"]
    if "init_τ" in parameters:
        init_params = np.array([parameters["init_τ"], parameters["init_λ"]])
    else:
         init_params=np.array([1.5,1.])
    hists = []
    for gid in gids:
        pop_L1PCHM = Population(games[gid][0], games[gid][1], rounds, p1_size,  p2_size,  init_params=init_params, params_vec=[p, τ, λ], σ_vec=[p_sd, τ_sd, λ_sd], lower_vec=[0.,0., 0.], upper_vec=[1.,4., 10.], random_params=True)
        hists.append(flatten_h(pop_L1PCHM.mul_runs_L1PCHM(n_runs)))
    flat_hists, shape = flatten_data(hists)
    return {"data": flat_hists, "shape": shape}


def BR_model(parameters):
    p = parameters["p"]
    ε = parameters["ε"]
    p_sd = parameters["p_sd"]
    ε_sd = parameters["ε_sd"]
    if "init_τ" in parameters:
        init_params = np.array([parameters["init_τ"], parameters["init_λ"]])
    else:
         init_params=np.array([1.5,1.])
    hists = []
    for gid in gids:
        pop_BR = Population(games[gid][0], games[gid][1], rounds, p1_size,  p2_size,  init_params=init_params, params_vec=[p,ε], σ_vec=[p_sd, ε_sd], lower_vec=[0.,0.], upper_vec=[1.,1.], random_params=True)
        hists.append(flatten_h(pop_BR.mul_runs_BR(n_runs)))
    flat_hists, shape = flatten_data(hists)
    return {"data": flat_hists, "shape": shape}


def LBR_model(parameters):
    p = parameters["p"]
#     ε = parameters["ε"]
    λ = parameters["λ"]
    p_sd = parameters["p_sd"]
#     ε_sd = parameters["ε_sd"]
    λ_sd = parameters["λ_sd"]
    if "init_τ" in parameters:
        init_params = np.array([parameters["init_τ"], parameters["init_λ"]])
    else:
         init_params=np.array([1.5,1.])
    hists = []
    for gid in gids:
        pop_LBR = Population(games[gid][0], games[gid][1], rounds, p1_size,  p2_size,  init_params=init_params, params_vec=[p, λ], σ_vec=[p_sd, λ_sd], lower_vec=[0., 0.], upper_vec=[1., 10.], random_params=True)
        hists.append(flatten_h(pop_LBR.mul_runs_LBR(n_runs)))
    flat_hists, shape = flatten_data(hists)
    return {"data": flat_hists, "shape": shape}

def EWA_model(parameters):
    p = parameters["p"]
    p_sd = parameters["p_sd"]
    λ = parameters["λ"]
    λ_sd = parameters["λ_sd"]
    φ = parameters["φ"]
    φ_sd = parameters["φ_sd"]
    ρ = parameters["ρ"]
    ρ_sd = parameters["ρ_sd"]
    δ = parameters["δ"]
    δ_sd = parameters["δ_sd"]
    if "init_τ" in parameters:
        init_params = np.array([parameters["init_τ"], parameters["init_λ"]])
    else:
         init_params=np.array([1.5,1.])
    hists = []
    for gid in gids:
        pop_EWA = Population(games[gid][0], games[gid][1], rounds, p1_size,  p2_size, init_params=init_params, params_vec=[p, λ, φ, ρ, δ], σ_vec=[p_sd, λ_sd, φ_sd, ρ_sd, δ_sd], lower_vec=[0.,0., 0., 0., 0.], upper_vec=[1.,10., 1., 1., 1.], random_params=True)
        hists.append(flatten_h(pop_EWA.mul_runs_EWA(n_runs)))
    flat_hists, shape = flatten_data(hists)
    return {"data": flat_hists, "shape": shape}

# def distance(x, y):
#     return 1 - calc_likelihood(x["data"], y["data"])/max_like

def distance(x,y):
    return max_like - calc_likelihood(x["data"], y["data"])

def euclidean_distance(x,y):
    simulated = unflatten_data(x["data"], x["shape"])
    hists = unflatten_data(y["data"], y["shape"])
    tot_distance = 0
    for sim, hist in zip(simulated, hists):
        tot_distance += scp.spatial.distance.euclidean(sim,hist)
    return tot_distance
#     return scp.spatial.distance.euclidean(x["data"], y["data"])

# param_spaces = dict()
# param_spaces["LK"] = {"τ":(0.6, 3.), "τ_sd":(0,0.5), "p":(0., 1.), "p_sd":(0,0.5)}
# param_spaces["PCHM"] = {"τ":(0.6, 3.), "τ_sd":(0,0.5), "p":(0., 1.), "p_sd":(0,0.5)}
# param_spaces["BR"] = {"ε":(0,0.3), "ε_sd":(0,0.5), "p":(0.,1.), "p_sd":(0,0.5)}
# param_spaces["EWA"] = {"λ":(0,10), "λ_sd":(0,5), "p":(0., 1), "p_sd":(0,0.5), "φ":(0, 1), "φ_sd":(0,0.5), "ρ":(0,1), "ρ_sd":(0,0.5) , "δ":(0,1), "δ_sd":(0,0.5)}

# init_param_spaces = dict()
# init_param_spaces["LK"] = {"τ":(0.6, 3.), "τ_sd":(0,0.5), "p":(0., 1.), "p_sd":(0,0.5), "init_τ":(0,4), "init_λ":(0,10)}
# init_param_spaces["PCHM"] = {"τ":(0.6, 3.), "τ_sd":(0,0.5), "p":(0., 1.), "p_sd":(0,0.5), "init_τ":(0,4), "init_λ":(0,10)}
# init_param_spaces["BR"] = {"ε":(0,0.3), "ε_sd":(0,0.5), "p":(0.,1.), "p_sd":(0,0.5), "init_τ":(0,4), "init_λ":(0,10)}
# init_param_spaces["EWA"] = {"λ":(0,10), "λ_sd":(0,5), "p":(0., 1), "p_sd":(0,0.5), "φ":(0, 1), "φ_sd":(0,0.5), "ρ":(0,1), "ρ_sd":(0,0.5) , "δ":(0,1), "δ_sd":(0,0.5), "init_τ":(0,4), "init_λ":(0,10)}



# αβ_param_spaces = dict()
# αβ_param_spaces["LK"] = {"τ":(0., 100.), "τ_sd":(0,100.), "p":(0., 100.), "p_sd":(0,100.), "init_τ":(0,4), "init_λ":(0,10)}
# αβ_param_spaces["LLK"] = {"τ":(0., 100.), "τ_sd":(0,100.), "p":(0., 100.), "p_sd":(0,100.), "λ":(0,100.), "λ_sd":(0,100.), "init_τ":(0,4), "init_λ":(0,10)}
# αβ_param_spaces["PCHM"] = {"τ":(0., 100.), "τ_sd":(0,100.), "p":(0., 100.), "p_sd":(0,100.), "init_τ":(0,4), "init_λ":(0,10)}
# αβ_param_spaces["LPCHM"] = {"τ":(0., 100.), "τ_sd":(0,100.), "p":(0., 100.), "p_sd":(0,100.), "λ":(0,100.), "λ_sd":(0,100.), "init_τ":(0,4), "init_λ":(0,10)}
# αβ_param_spaces["BR"] = {"ε":(0,100.), "ε_sd":(0,100.), "p":(0.,100.), "p_sd":(0,100.), "init_τ":(0,4), "init_λ":(0,10)}
# αβ_param_spaces["EWA"] = {"λ":(0,10), "λ_sd":(0,100.), "p":(0., 100.), "p_sd":(0,100.), "φ":(0, 100.), "φ_sd":(0,100.), "ρ":(0, 100.), "ρ_sd":(0,100.) , "δ":(0,100.), "δ_sd":(0,100.), "init_τ":(0,4), "init_λ":(0,10)}


αβ_param_spaces = dict()
αβ_param_spaces["LK"] = {"τ":(0., 100.), "τ_sd":(0,100.), "p":(0., 100.), "p_sd":(0,100.), "init_τ":(0,4), "init_λ":(0,10)}
αβ_param_spaces["LLK"] = {"τ":(0., 100.), "τ_sd":(0,100.), "p":(0., 100.), "p_sd":(0,100.), "λ":(0,100.), "λ_sd":(0,100.), "init_τ":(0,4), "init_λ":(0,10)}
αβ_param_spaces["PCHM"] = {"τ":(0., 100.), "τ_sd":(0,100.), "p":(0., 100.), "p_sd":(0,100.), "init_τ":(0,4), "init_λ":(0,10)}
αβ_param_spaces["JPCHM"] = {"τ":(0., 100.), "τ_sd":(0,100.), "p":(0., 100.), "p_sd":(0,100.), "init_τ":(0,4), "init_λ":(0,10)}
αβ_param_spaces["LPCHM"] = {"τ":(0., 100.), "τ_sd":(0,100.), "p":(0., 100.), "p_sd":(0,100.), "λ":(0,100.), "λ_sd":(0,100.), "init_τ":(0,4), "init_λ":(0,10)}
αβ_param_spaces["LJPCHM"] = {"τ":(0., 100.), "τ_sd":(0,100.), "p":(0., 100.), "p_sd":(0,100.), "λ":(0,100.), "λ_sd":(0,100.), "init_τ":(0,4), "init_λ":(0,10)}
αβ_param_spaces["BR"] = {"ε":(0,100.), "ε_sd":(0,100.), "p":(0.,100.), "p_sd":(0,100.), "init_τ":(0,4), "init_λ":(0,10)}
αβ_param_spaces["LBR"] = {"ε":(0,100.), "ε_sd":(0,100.), "p":(0.,100.), "p_sd":(0,100.), "λ":(0,100.), "λ_sd":(0,100.),  "init_τ":(0,4), "init_λ":(0,10)}
αβ_param_spaces["EWA"] = {"λ":(0,10), "λ_sd":(0,100.), "p":(0., 100.), "p_sd":(0,100.), "φ":(0, 100.), "φ_sd":(0,100.), "ρ":(0, 100.), "ρ_sd":(0,100.) , "δ":(0,100.), "δ_sd":(0,100.), "init_τ":(0,4), "init_λ":(0,10)}


## With init_params sampling
# param_spaces = dict()
# param_spaces["LLK"] = {"τ":(0., 4.), "τ_sd":(0,0.1), "p":(0., 1.), "p_sd":(0,0.1), "λ":(0.,10.), "λ_sd":(0,1.), "init_τ":(0,4), "init_λ":(0,10)}
# param_spaces["LJPCHM"] = {"τ":(0., 4.), "τ_sd":(0,0.1), "p":(0., 1.), "p_sd":(0,0.1), "λ":(0.,10.), "λ_sd":(0,1.), "init_τ":(0,4), "init_λ":(0,10)}
# param_spaces["L1PCHM"] = {"τ":(0., 4.), "τ_sd":(0,0.1), "p":(0., 1.), "p_sd":(0,0.1), "λ":(0.,10.), "λ_sd":(0,1.), "init_τ":(0,4), "init_λ":(0,10)}
# param_spaces["LPCHM"] = {"τ":(0., 4.), "τ_sd":(0,0.1), "p":(0., 1.), "p_sd":(0,0.1), "λ":(0.,10.), "λ_sd":(0,1.), "init_τ":(0,4), "init_λ":(0,10)}
# param_spaces["LBR"] = {"p":(0.,1.), "p_sd":(0,0.1), "λ":(0.,10.), "λ_sd":(0,1.), "init_τ":(0,4), "init_λ":(0,10)}
# param_spaces["EWA"] = {"λ":(0,10), "λ_sd":(0,1.), "p":(0., 1), "p_sd":(0,0.1), "φ":(0, 1), "φ_sd":(0,0.1), "ρ":(0,1), "ρ_sd":(0,0.1) , "δ":(0,1), "δ_sd":(0,0.1), "init_τ":(0,4), "init_λ":(0,10)}




param_spaces = dict()
param_spaces["LLK"] = {"τ":(0., 4.), "τ_sd":(0,0.1), "p":(0., 1.), "p_sd":(0,0.1), "λ":(0.,10.), "λ_sd":(0,1.)}
param_spaces["LJPCHM"] = {"τ":(0., 4.), "τ_sd":(0,0.1), "p":(0., 1.), "p_sd":(0,0.1), "λ":(0.,10.), "λ_sd":(0,1.)}
param_spaces["L1PCHM"] = {"τ":(0., 3.), "τ_sd":(0,0.4), "p":(0., 1.), "p_sd":(0,0.2), "λ":(0.,10.), "λ_sd":(0,1.)}
param_spaces["LPCHM"] = {"τ":(0., 2.), "τ_sd":(0,0.3), "λ":(0.,10.), "λ_sd":(0,1.)}
param_spaces["LBR"] = {"p":(0.,1.), "p_sd":(0,0.1), "λ":(0.,10.), "λ_sd":(0,1.)}
param_spaces["EWA"] = {"λ":(0,10), "λ_sd":(0,1.), "p":(0., 1), "p_sd":(0,0.2), "φ":(0, 1), "φ_sd":(0,0.2), "ρ":(0,1), "ρ_sd":(0,0.2) , "δ":(0,1), "δ_sd":(0,0.2)}



αβ_params = dict()
αβ_params["LK"] = ["τ", "p"]
αβ_params["LLK"] = ["τ", "p", "λ"]
αβ_params["PCHM"] = ["τ", "p"]
αβ_params["JPCHM"] = ["τ", "p"]
αβ_params["LPCHM"] = ["τ", "p", "λ"]
αβ_params["LJPCHM"] = ["τ", "p", "λ"]
αβ_params["L1PCHM"] = ["τ", "p", "λ"]

αβ_params["BR"] = ["ε", "p"]
αβ_params["LBR"] = ["ε", "p", "λ"]
αβ_params["EWA"] = ["p", "λ", "φ", "ρ", "δ"]


αβ_plot_lims = copy.deepcopy(αβ_param_spaces)
for model in αβ_plot_lims:
    for param in αβ_params[model]:
        a,b = αβ_lims[model][param]
        αβ_plot_lims[model][param] = αβ_lims[model][param]
        αβ_plot_lims[model][param+"_sd"] = αβ_lims[model][param]

# model_names = ["LK", "LLK", "PCHM", "LPCHM", "JPCHM", "LJPCHM", "BR", "LBR", "EWA"]
# models = [LK_model, LLK_model, PCHM_model, LPCHM_model, JPCHM_model, LJPCHM_model, BR_model, LBR_model, EWA_model]

# model_names = ["LLK", "LPCHM", "LJPCHM", "LBR", "EWA"]
# models = [LLK_model, LPCHM_model, LJPCHM_model, LBR_model, EWA_model]

# model_names = ["LPCHM", "LJPCHM"]
# models = [LPCHM_model, LJPCHM_model]


# model_names = ["LLK", "LJPCHM", "LBR", "EWA"]
# models = [LLK_model, LJPCHM_model, LBR_model, EWA_model]

# model_names = ["L1PCHM", "LJPCHM", "LPCHM", "EWA"]
# models = [L1PCHM_model, LJPCHM_model, LPCHM_model, EWA_model]

model_names = ["LPCHM", "EWA"]
models = [LPCHM_model, EWA_model]

param_names = dict()
param_names["EWA"] = ["p", "λ", "φ", "ρ", "δ"]
param_names["L1PCHM"] = ["τ", "p", "λ"]
param_names["LPCHM"] = ["τ", "λ"]


# priors = [Distribution(**{key: RV("uniform", a, b - a)
#                         for key, (a,b) in param_spaces[mod].items()}) for mod in ["LK", "PCHM", "BR", "EWA"]]
# init_priors = [Distribution(**{key: RV("uniform", a, b - a)
#                         for key, (a,b) in init_param_spaces[mod].items()}) for mod in ["LK", "PCHM", "BR", "EWA"]]

# αβ_priors = [Distribution(**{key: RV("uniform", a, b - a)
#                         for key, (a,b) in αβ_param_spaces[mod].items()}) for mod in ["LK", "LLK", "PCHM", "LPCHM", "BR", "EWA"]]



priors = [Distribution(**{key: RV("uniform", a, b - a)
                        for key, (a,b) in param_spaces[mod].items()}) for mod in model_names]

# αβ_priors = [Distribution(**{key: RV("uniform", a, b - a)
#                         for key, (a,b) in αβ_param_spaces[mod].items()}) for mod in model_names]
