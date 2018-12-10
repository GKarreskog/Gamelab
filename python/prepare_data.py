#%%
import numpy as np
import pandas as pd
import pickle as pickle
import matplotlib.pyplot as plt
import matplotlib
import builtins

def rational_to_double(rat):
    """ Takes a string of typ 4/4 or 4 and returns the corresponding float"""
    tmp = rat.split("/")
    if len(tmp) > 1:
        num = float(tmp[0])
        den = float(tmp[1])
        return num/den
    else:
        return float(tmp[0])

def string_to_list(lista):
    """ Takes a string of rationals on format 1/2 and returns a list of floats """
    splitted = lista.split()
    return [rational_to_double(x) for x in splitted]

def get_payoff_mats(payoffs, p1_n_strats):
    """ Takes a string of payoffs and return an array with the two payoff mats """
    payoff_list = [int(x) for x in payoffs.split()]
    p2_n_strats = int(len(payoff_list)/(2*p1_n_strats))

    p1_payoffs = np.array([payoff_list[i] for i in range(0,len(payoff_list)) if i % 2 == 0])
    p2_payoffs = np.array([payoff_list[i] for i in range(0,len(payoff_list)) if (i+1) % 2 == 0])

    p1_payoff_mat = np.reshape(p1_payoffs, (p1_n_strats, p2_n_strats))
    p2_payoff_mat = np.reshape(p2_payoffs, (p2_n_strats, p1_n_strats), order='F')

    return [p1_payoff_mat, p2_payoff_mat]


all_choices = pd.read_csv("../DATA_2018-11-19/choice.csv")
all_players = pd.read_csv("../DATA_2018-11-19/player.csv")
all_games = pd.read_csv("../DATA_2018-11-19/game.csv")
all_payoffs = pd.read_csv("../DATA_2018-11-19/payoff.csv")
all_pasts = pd.read_csv("../DATA_2018-11-19/past.csv")
all_gameplays = pd.read_csv("../DATA_2018-11-19/gameplay.csv")


all_choices["strats"] = all_choices["strats"].apply(string_to_list)
all_payoffs["payoff"] = all_payoffs["payoff"].apply(rational_to_double)
all_pasts["currentsp1"] = all_pasts["currentsp1"].apply(string_to_list)
all_pasts["currentsp2"] = all_pasts["currentsp2"].apply(string_to_list)
all_games["payoffs"] = all_games.apply(lambda row: get_payoff_mats(row["payoffs"], row["p1"]), axis=1)


init_strats_dict = dict()
init_p1_dict = dict()
init_p2_dict = dict()
payoffs_dict = dict()
role_dict = dict()
id_dict = dict()
actual_plays_dict = dict()
gid = 1
for gid in range(1,4):
    strats = all_choices[["strats", "playerid"]][(all_choices["round"] == 0) & (all_choices["gameid"] == gid)].copy()
    roles = all_players[["id", "role"]].copy()
    roles["playerid"] = roles["id"]
    pasts = all_pasts[["currentsp1", "currentsp2", "gameid"]][(all_pasts["round"] == 0) & (all_pasts["gameid"] == gid)].copy()

    init_p1 = all_pasts["currentsp1"][(all_pasts["round"] == 0) & (all_pasts["gameid"] == gid)].copy()
    init_p2 = all_pasts["currentsp2"][(all_pasts["round"] == 0) & (all_pasts["gameid"] == gid)].copy()

    data_mat = strats.set_index("playerid").join(roles.set_index("id"), how='outer').copy()

    # # Replace NaN with role average
    data_mat["strats"][(data_mat["role"] == 0) & pd.isnull(data_mat["strats"])] = np.repeat(np.array(init_p1), sum((data_mat["role"] == 0) & pd.isnull(data_mat["strats"])))
    data_mat["strats"][(data_mat["role"] == 1) & pd.isnull(data_mat["strats"])] = np.repeat(np.array(init_p2), sum((data_mat["role"] == 1) & pd.isnull(data_mat["strats"])))

    plays_vec = [np.array(all_pasts["currentsp1"][all_pasts["gameid"] == gid]), np.array(all_pasts["currentsp2"][all_pasts["gameid"] == gid])]
    plays_vec[0] = np.array([np.array(v) for v in plays_vec[0]])
    plays_vec[1] = np.array([np.array(v) for v in plays_vec[1]])

    init_strats_dict[gid] = np.array(data_mat["strats"])
    init_p1_dict[gid] = init_p1
    init_p2_dict[gid] = init_p2
    payoffs_dict[gid] = all_games["payoffs"][all_games["id"]==gid]
    role_dict[gid] = np.array(data_mat["role"].copy())
    id_dict[gid] = np.array(data_mat["playerid"].copy())
    actual_plays_dict[gid] = plays_vec

#%%

save_dict = {"init_strats_dict": init_strats_dict, "init_p1_dict":init_p1_dict,
"init_p2_dict": init_p2_dict, "payoffs_dict":payoffs_dict,
"role_dict":role_dict, "id_dict":id_dict, "actual_plays_dict":actual_plays_dict}

## Save dictionary

with open('data_dicts.pkl', 'wb') as output:
    pickle.dump(save_dict, output, pickle.HIGHEST_PROTOCOL)

#%%
with open('data_dicts.pkl', 'rb') as input:
    load_dict = pickle.load(input)
    init_strats_dict = load_dict["init_strats_dict"]
    init_p1_dict = load_dict["init_p1_dict"]
    init_p2_dict = load_dict["init_p2_dict"]
    payoffs_dict = load_dict["payoffs_dict"]
    role_dict = load_dict["role_dict"]
    id_dict = load_dict["id_dict"]
    actual_plays_dict = load_dict["actual_plays_dict"]

#%%
rgb_cols = [(228/255, 26/255, 28/255), (55/255, 126/255, 184/255), (77/255, 175/255, 74/255), (152/255, 78/255, 163/255), (255/255, 127/255, 0/255)]
# ['rgb(228,26,28)','rgb(55,126,184)','rgb(77,175,74)','rgb(152,78,163)','rgb(255,127,0)']
matplotlib.colors.is_color_like(rgb_cols[0])


def plot_h(h, save=False):
    plt.figure(figsize=(24,8))
    for role in range(2):
        ax = plt.subplot(1,2,(role + 1))
        ax.set_ylim([-0.01,1.01])
        ax.set_yticks([0,0.25,0.5,0.75,1])
        print(h[role][1].shape)
        n_s = h[role].shape[1]
        for s in range(n_s):
            plt.plot(h[role][:,s], color=rgb_cols[s], ls="-", label="Strat "+str(s))
        ax.legend()
    if save:
        plt.savefig(save, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

plot_h(actual_plays_dict[3], "game3.png")

#%%

same_plays_vec = []
for pid in pd.Series.unique(all_choices.playerid):
    if len(pd.Series.unique(all_choices["gameid"][((all_choices.playerid == pid))])) == 3:
        dict = {"pid": pid}
        rounds = all_choices["round"][(all_choices.playerid == pid )]
        rounds = pd.Series.unique(rounds)
        dict["n_rounds"] = len(rounds)
        dict["first_round"] = min(rounds)
        dict["last_round"] = max(rounds)
        for gid in range(1,4):
            choices = list(all_choices["strats"][((all_choices.playerid == pid ) & (all_choices.gameid == gid))])
            same = 0
            diff = 0
            prev = choices[0]
            for choice in choices[1:]:
                if choice == prev:
                    same += 1
                prev = choice
            dict[str(gid) + "_n_same"] = same
            dict[str(gid) + "_p_same"] = same/len(rounds)
            dict["n_rounds_" + str(gid)] = len(choices)
        same_plays_vec.append(dict)

same_plays_df = pd.DataFrame(same_plays_vec)
same_plays_df
gid_1_same = np.mean(np.array(same_plays_df["1_n_same"]))/np.mean(same_plays_df["n_rounds_1"] -1)
gid_2_same = np.mean(np.array(same_plays_df["2_n_same"]))/np.mean(same_plays_df["n_rounds_2"] -1)
gid_3_same = np.mean(np.array(same_plays_df["3_n_same"]))/np.mean(same_plays_df["n_rounds_3"] -1)

gid_3_same

np.mean(same_plays_df["n_rounds"])

#%%

def best_reply(payoff_mat, opp_s):
    n_strats = payoff_mat.shape[0]
    avg_payoff = payoff_mat @ opp_s
    best_rep = np.zeros(n_strats)
    if (avg_payoff == avg_payoff.max()).sum() == 1:
        best_rep[avg_payoff == avg_payoff.max()] = 1
    else:
        sample = np.zeros((avg_payoff == avg_payoff.max()).sum())
        sample_simplex(sample)
        best_rep[avg_payoff == avg_payoff.max()] = sample
    # best_rep[avg_payoff == avg_payoff.max()] = 1
    # best_rep = best_rep/best_rep.sum()
    return best_rep

p1_payoff = 0
p1_best_payoff = 0
p2_payoff = 0
p2_best_payoff = 0
gid = 3
for round in range(0, 30):
    p1_play = actual_plays_dict[gid][0][round]
    p2_play = actual_plays_dict[gid][1][round]
    p1_payoff += p1_play @ payoffs_dict[gid][gid-1][0] @ p2_play
    p1_best = best_reply(payoffs_dict[gid][gid-1][0], p2_play)
    p1_best_payoff += p1_best @ payoffs_dict[gid][gid-1][0] @ p2_play
    p2_payoff += p2_play @ payoffs_dict[gid][gid-1][1] @ p1_play
    p2_best = best_reply(payoffs_dict[gid][gid-1][1], p1_play)
    p2_best_payoff += p2_best @ payoffs_dict[gid][gid-1][1] @ p1_play


p1_payoff
p1_best_payoff
p2_payoff
p2_best_payoff


# %% Game tables

gid = 3
game_info_vec = []
for pid in pd.Series.unique(all_choices.playerid):
    if len(pd.Series.unique(all_choices["gameid"][((all_choices.playerid == pid))])) == 3:
        dict = {"pid": pid}
        rounds = all_choices["round"][(all_choices.playerid == pid) & (all_choices.gameid == gid)]
        rounds = pd.Series.unique(rounds)
        dict["n rounds"] = len(rounds)
        dict["first"] = min(rounds)
        dict["last"] = max(rounds)
        choices = list(all_choices["strats"][((all_choices.playerid == pid ) & (all_choices.gameid == gid))])
        same = 0
        diff = 0
        prev = choices[0]
        mixed = 0
        for choice in choices[1:]:
            if choice == prev:
                same += 1
            prev = choice
            if sum(map(lambda x: x > 0, choice)) > 1:
                mixed += 1
        dict["n same"] = same
        dict["% same"] = builtins.round(same/len(rounds),2)
        dict["n mix"] = mixed
        dict["% mix"] = builtins.round(mixed/len(rounds),2)
        player_payoff = 0
        player_opt_payoff = 0
        mixed_payoff = 0
        role = int(all_players["role"][all_players.id == pid])
        dict["role"] = role
        opp_role = (role + 1) % 2
        round = 3
        gid
        for round in rounds:
            choice = list(all_choices["strats"][((all_choices.playerid == pid ) & (all_choices.gameid == gid) & (all_choices["round"] == round))])[0]
            mixed = np.ones(len(choice))/len(choice)
            player_payoff += choice @ payoffs_dict[gid][gid-1][role] @ actual_plays_dict[gid][opp_role][round]
            best_play = best_reply(payoffs_dict[gid][gid-1][role], actual_plays_dict[gid][opp_role][round])
            player_opt_payoff += best_play @ payoffs_dict[gid][gid-1][role] @ actual_plays_dict[gid][opp_role][round]
            mixed_payoff += mixed @ payoffs_dict[gid][gid-1][role] @ actual_plays_dict[gid][opp_role][round]
        dict["PO"] = builtins.round(player_payoff, 2)
        dict["mix PO"] = mixed_payoff
        dict["max PO"] = builtins.round(player_opt_payoff, 2)
        dict["% PO"] = builtins.round(player_payoff/player_opt_payoff, 2)
        game_info_vec.append(dict)


builtins.round(0.3333333, 2)
game_info_df = pd.DataFrame(game_info_vec)
game_info_df = game_info_df.set_index("pid")

game_info_df = game_info_df[["role", "n rounds", "first", "last", "n same", "% same", "n mix", "% mix", "mix PO", "PO", "max PO", "% PO"]]



avg_dict = {"pid": "all"}
for key in game_info_vec[0]:
    if key != "pid":
        avg_dict[key] = np.mean(game_info_df[key])
avg_df = pd.DataFrame([avg_dict])
avg_df = avg_df.set_index("pid")


game_info_df = game_info_df.round(2)
print(pd.DataFrame.to_latex(game_info_df))
game_info_df = game_info_df.append(avg_df, sort=False)
game_info_df = game_info_df.round(2)
print(pd.DataFrame.to_latex(game_info_df))

all_choices["strats"][((all_choices.playerid == pid ) & (all_choices.gameid == gid) & (all_choices["round"] == 3))]
all_choices[(all_choices.playerid == pid) & (all_choices["round"] == 3)]
same_plays_vec.append(dict)

same_plays_df = pd.DataFrame(same_plays_vec)

sum(map(lambda x: x > 0, [1,1,0]))

#%% Are changes improvements?

change_in_play_vec= []
for gid in [1,2,3]:
    for pid in pd.Series.unique(all_choices.playerid):
        if len(pd.Series.unique(all_choices["gameid"][((all_choices.playerid == pid))])) == 3:
            rounds = all_choices["round"][(all_choices.playerid == pid) & (all_choices.gameid == gid)]
            rounds = pd.Series.unique(rounds)
            choices = list(all_choices["strats"][((all_choices.playerid == pid ) & (all_choices.gameid == gid))])
            same = 0
            diff = 0
            prev = choices[0]
            mixed = 0
            role = int(all_players["role"][all_players.id == pid])
            dict["role"] = role
            opp_role = (role + 1) % 2
            # change_vec = []
            # regret_vec = []
            # improvement_vec = []
            for i in range(1, len(rounds) - 1):
                dict = {"pid": pid, "gid":gid}
                choice = list(all_choices["strats"][((all_choices.playerid == pid ) & (all_choices.gameid == gid) & (all_choices["round"] == rounds[i]))])[0]
                prev_choice = list(all_choices["strats"][((all_choices.playerid == pid ) & (all_choices.gameid == gid) & (all_choices["round"] == rounds[i-1]))])[0]
                next_choice = list(all_choices["strats"][((all_choices.playerid == pid ) & (all_choices.gameid == gid) & (all_choices["round"] == rounds[i+1]))])[0]
                prev_payoff = prev_choice @ payoffs_dict[gid][gid-1][role] @ actual_plays_dict[gid][opp_role][rounds[i]]
                player_payoff = choice @ payoffs_dict[gid][gid-1][role] @ actual_plays_dict[gid][opp_role][rounds[i]]
                next_payoff = next_choice @ payoffs_dict[gid][gid-1][role] @ actual_plays_dict[gid][opp_role][rounds[i]]
                best_play = best_reply(payoffs_dict[gid][gid-1][role], actual_plays_dict[gid][opp_role][rounds[i]])
                player_opt_payoff = best_play @ payoffs_dict[gid][gid-1][role] @ actual_plays_dict[gid][opp_role][rounds[i]]

                prev_round_payoff =  prev_choice @ payoffs_dict[gid][gid-1][role] @ actual_plays_dict[gid][opp_role][rounds[i-1]]
                prev_best_play = best_reply(payoffs_dict[gid][gid-1][role], actual_plays_dict[gid][opp_role][rounds[i-1]])
                prev_player_opt_payoff = best_play @ payoffs_dict[gid][gid-1][role] @ actual_plays_dict[gid][opp_role][rounds[i-1]]
                dict["changed"] = (choice != prev_choice)
                dict["changing"] = (choice != next_choice)
                dict["improvement"] = player_payoff - prev_payoff
                dict["bad change"] = (player_payoff < prev_payoff)
                dict["improvement current"] = next_payoff - player_payoff
                dict["regret"] = player_opt_payoff - player_payoff
                dict["prev regret"] = prev_player_opt_payoff - prev_round_payoff
                dict["gid"] = gid
                dict["round"] = rounds[i]
                change_in_play_vec.append(dict)

change_in_play_df = pd.DataFrame(change_in_play_vec)

change_in_play_df

change_in_play_df[["regret", "changing"]].cov()
change_in_play_df[["prev regret", "improvement"]][change_in_play_df.changed == True].corr()
change_in_play_df["improvement"][change_in_play_df.changed == True].mean()
change_in_play_df["improvement current"][change_in_play_df.changing == True].mean()
sum(change_in_play_df["improvement current"] > 0)

sum(change_in_play_df["improvement"] > 0)


sum(change_in_play_df["changed"])
sum(change_in_play_df["bad change"])

change_in_play_df["improvement"][change_in_play_df["bad change"] == True].mean()
change_in_play_df["improvement"][(change_in_play_df["changed"] == True) & (change_in_play_df["bad change"] != True)].mean()


avg_dict = {"pid": "all", "gid":"all"}
for key in change_in_play_vec[0]:
    if (key != "pid") & (key != "gid"):
        avg_dict[key] = np.mean(change_in_play_df[key])
avg_df = pd.DataFrame([avg_dict])
change_in_play_df = change_in_play_df.append(avg_df, sort=False)

## Save df to print
summary_dict = []
ele_dict = {"name":"corr regret change"}
ele_dict["all"] = change_in_play_df[["regret", "changing"]].corr().iloc[1][0]
for gid in [1,2,3]:
    ele_dict["Game "+str(gid)] = change_in_play_df[["regret", "changing"]][change_in_play_df.gid == gid].corr().iloc[1][0]
summary_dict.append(ele_dict)

ele_dict = {"name":"improvement on change"}
ele_dict["all"] = change_in_play_df["improvement"][change_in_play_df.changed == True].mean()
for gid in [1,2,3]:
    ele_dict["Game "+str(gid)] = change_in_play_df["improvement"][(change_in_play_df.changed == True) & (change_in_play_df.gid == gid)].mean()
summary_dict.append(ele_dict)

ele_dict = {"name":"within round improvement"}
ele_dict["all"] = change_in_play_df["improvement current"][change_in_play_df.changing == True].mean()
for gid in [1,2,3]:
    ele_dict["Game "+str(gid)] = change_in_play_df["improvement current"][(change_in_play_df.changing == True) & (change_in_play_df.gid == gid)].mean()
summary_dict.append(ele_dict)


ele_dict = {"name": "n changes"}
ele_dict["all"] = change_in_play_df["changed"].sum()
for gid in [1,2,3]:
    ele_dict["Game "+str(gid)] = change_in_play_df["changed"][(change_in_play_df.gid == gid)].sum()
summary_dict.append(ele_dict)

ele_dict = {"name": "bad changes"}
ele_dict["all"] = change_in_play_df["bad change"].sum()
for gid in [1,2,3]:
    ele_dict["Game "+str(gid)] = change_in_play_df["bad change"][(change_in_play_df.gid == gid)].sum()
summary_dict.append(ele_dict)


summary_df = pd.DataFrame(summary_dict)
summary_df = summary_df.set_index("name")
summary_df.round(2)
