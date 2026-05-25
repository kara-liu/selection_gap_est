import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import pandas as pd 
import numpy as np 
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker


METHOD_NAME_DICT ={
        'ippw_Utilde': r'Naive IPPW of $\tilde{U}$',
        'ub_trueUstar': r'UB (true $U^*$)',
        'ub_predUstar':r'UB (heuristic $U^*$)',
        'baseline_calib': 'Calibration',
        'baseline_eb': 'Ent. balancing',
        'baseline_raking': 'Raking'
    }


DECOMP_PALETTE = {"$\hat{R}_Q - R_Q$": '#264653',
                 r"$\Delta_{TBE}$": '#2a9d8f',
                r"$\Delta_{CI}$": '#E9C772',
                r"$\Delta_{CS}$": '#F39B53'}

sns.set_theme(style="whitegrid", rc={
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 16,
    "legend.fontsize": 13,
    "font.family": "serif",
    "text.usetex": False,          
    "font.family": "serif",      
    "mathtext.fontset": "dejavuserif" 
})

def get_error_stats(df,eps=-1e-2,methods=[
        'baseline_raking_logloss_vs_RQ',
            'baseline_calib_logloss_vs_RQ',
            'baseline_eb_logloss_vs_RQ',
            'ippw_Utilde_logloss_vs_RQ',
            'ub_trueUstar_logloss_vs_RQ',
            'ub_predUstar_logloss_vs_RQ'
            ]):
    
    """
    Plot stats of the error RQhat - RQ for each method, where we average across seeds first 
    and then report the average among the different tasks and different values of Utilde_cols.
    We report : mu \pm std; perc_valid where RQhat - RQ > eps; the 5th and 9th percentiles,
     the design effect of the weights used to compute RQhat
    """
    stats = pd.DataFrame({})
    
    for c in methods:
        if c not in df.columns:
            continue
        method_name = '_'.join(c.split('_')[:2])
        agg_error_df = df[~df[c].isna()].groupby(by=['Utilde_cols','task'])[c].mean().reset_index()
        agg_de_df = df[~df[c].isna()].groupby(by=['Utilde_cols','task'])[method_name+'_de'].mean().reset_index()

        agg_error_df['valid_upper_bound'] = agg_error_df[c]>eps

        stats = pd.concat([stats,pd.DataFrame({'method': METHOD_NAME_DICT[method_name],
                                            'mean_error': round(agg_error_df[c].mean(),6),
                                            'std_error': round(agg_error_df[c].std(),6),
                                            'perc_valid_bound': round(agg_error_df['valid_upper_bound'].mean(),6), 
                                            '5perc': round(np.percentile(agg_error_df[c],5),2),
                                            '95perc': round(np.percentile(agg_error_df[c],95),2),
                                            'design effect': round(agg_de_df[method_name+'_de'].mean(),1),
                                            },index=[0])])
    return stats


def plot_error_by_udim(df, max_udim=None, methods=[
            'baseline_calib_logloss_vs_RQ',
            'baseline_eb_logloss_vs_RQ',
            'baseline_raking_logloss_vs_RQ',
            'ippw_Utilde_logloss_vs_RQ',
            'ub_trueUstar_logloss_vs_RQ',
            'ub_predUstar_logloss_vs_RQ'
            ]):
    """
    Plot RQhat - RQ error stratified by Utilde dim observed 
    Because baseline are not affected by Utilde dim observed, they will be shown 
    under Constant 

    max_udim = dim(U)
    """
    df = df.copy()
    if max_udim is None:
        max_udim = df.Utilde_dim.max()
        print("Setting max_udim to ", max_udim)

    df = df[df.Utilde_dim < max_udim] # When Utilde_dim == U_dim, we have fully observed selection 
    df['Utilde_dim_name'] = df.Utilde_dim.apply(lambda x: r'd($\tilde{U}$)='+str(x)+'\n'+r'd($U^*$)='+str(max_udim-x))

    for c in methods:
        if c not in df.columns:
            print(f"Method {c} not found!! Setting to 0")
            df[c] = 0

    df = df[['Utilde_dim_name', 'task'] + methods].melt(
        id_vars=['Utilde_dim_name', 'task'],
        value_vars=methods,
        var_name='method',
        value_name='gap_logloss'
    )
    df = df.groupby(["task", "method", "Utilde_dim_name"]).agg(gap_logloss=("gap_logloss", "mean"),).reset_index()

    df['method'] = df['method'].apply(lambda m: METHOD_NAME_DICT['_'.join(m.split('_')[:2])])
    # Because baseline are not affected by Utilde dim observed, they will be shown 
    # under Constant 
    df.loc[df.method.isin(['Calibration', 'Ent. balancing', 'Raking']), 'Utilde_dim_name'] = 'Constant'
        
    # Create two sub-dataframes: one for varying Ustar_dim, one for constant Ustar_dim
    varyUdim_subdf = df.loc[df['Utilde_dim_name'] != 'Constant']
    varyUdim_subdf["method"] = varyUdim_subdf["method"].astype('category').cat.reorder_categories(
        [
            "UB (true $U^*$)",
            "UB (heuristic $U^*$)",
            "Naive IPPW of $\\tilde{U}$",
        ],
        ordered=True,
    )
    constUdim_subdf = df.loc[df['Utilde_dim_name'] == 'Constant']
    constUdim_subdf["method"] = constUdim_subdf["method"].astype('category').cat.reorder_categories(
        [
            "Raking",
            "Calibration",
            "Ent. balancing",
        ],
        ordered=True,
    )
    custom_palette_baselines= sns.color_palette(["#e76f51",'#A23216','#1C0221'])
    custom_palette_varyUtildedim = sns.color_palette(["#2a9d8f", "#E9C772","#f4a261"])
    # Create two suplots, one for varying Ustar_dim, one for constant Ustar_dim
    # such that they look like one single, seamless plot
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(8.5, 4),
        sharey=True,
        sharex=False,
        gridspec_kw={"width_ratios": [4, 1]},
    )
    plt.subplots_adjust(wspace=0)  
    axs[1].spines['left'].set_visible(False)


    for ax, df, palette in zip(axs, [varyUdim_subdf, constUdim_subdf], 
                               [custom_palette_varyUtildedim, custom_palette_baselines]):
        sns.pointplot(
            data=df,
            x="Utilde_dim_name",
            y="gap_logloss",
            hue="method",
            palette=palette,
            dodge=0.25,
            markers='o',
            markersize=9,
            markerfacecolor='white',
            markeredgewidth=2,
            errorbar='sd',
            linestyle='',
            err_kws={'linewidth':2},
            capsize=0.05,
            ax=ax
        )
        sns.pointplot(
            data=df,
            x="Utilde_dim_name",
            y="gap_logloss",
            hue="method",
            palette=palette,
            dodge=0.25,
            markers='o',
            markersize=9,
            markerfacecolor='white',
            markeredgewidth=2,
            errorbar=None,
            linestyle='',
            capsize=0.05,
            ax=ax
        )

    for ax in axs:
        ax.axhline(y=0, linewidth=1, color='grey')
        ax.set_xlabel("")
        ax.set_ylabel(r"$\hat{R}_Q - {R}_Q$")


    # Get handles/labels from both axes and dedupe
    handles1, labels1 = axs[0].get_legend_handles_labels()
    handles2, labels2 = axs[1].get_legend_handles_labels()
    pairs = []
    seen = set()
    for h, lbl in list(zip(handles1, labels1)) + list(zip(handles2, labels2)):
        if lbl not in seen and lbl != "":
            pairs.append((h, lbl))
            seen.add(lbl)

    # Remove individual legends
    if axs[0].get_legend() is not None:
        axs[0].get_legend().remove()
    if axs[1].get_legend() is not None:
        axs[1].get_legend().remove()

    # Create custom filled-circle legend (bigger markers)
    legend_markersize = 8  # adjust as desired
    custom_handles, custom_labels = [], []
    for h, lbl in pairs:
        color = h.get_color()
       
        custom_handles.append(
            Line2D([], [], marker='o', linestyle='',
                markersize=legend_markersize,
                markeredgewidth=3,
                markerfacecolor='white',      # FILLED in legend
                markeredgecolor=color)
        )
        custom_labels.append(lbl)

    legend = axs[1].legend(
        custom_handles, custom_labels,
        title=None, loc="upper right", bbox_to_anchor=(3, 1.1), borderaxespad=0.0
    )

    frame = legend.get_frame()
    frame.set_facecolor("white")  # light grey
    frame.set_edgecolor("none")     # no border
    frame.set_alpha(0.8)  

    for ax in axs:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v:.2f}"))


def plot_error_decomp(
    df,
    xlabel=None,
    figsize=(6.2, 4.2),
):
    if xlabel is None:
        xlabel='Averaged over all tasks'
        df[xlabel] = 0

    cols = [
            "ub_trueUstar_logloss_vs_RQ",
            "ub_trueUstar_vs_ippw_margXYgUtilde_logloss",
            "ippw_margXYgUtilde_vs_ippw_XYUtilde_logloss",
            "ippw_XYUtilde_logloss_vs_RQ",
        ]
    df_long = df[[xlabel] + cols].melt(
        id_vars=[xlabel],
        value_vars=cols,
        var_name="decomp",
        value_name="gap_logloss",
    )
    df_long["decomp"] = df_long["decomp"].map({
            "ub_trueUstar_logloss_vs_RQ": r"$\hat{R}_Q - R_Q$",
            "ub_trueUstar_vs_ippw_margXYgUtilde_logloss": r"$\Delta_{TBE}$",
            "ippw_margXYgUtilde_vs_ippw_XYUtilde_logloss": r"$\Delta_{CI}$",
            "ippw_XYUtilde_logloss_vs_RQ": r"$\Delta_{CS}$",
        })

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    ax.set_axisbelow(True)
    ax.grid(True, axis="y", zorder=0)

    sns.barplot(
        data=df_long,
        x=xlabel, y="gap_logloss", hue="decomp",
        hue_order=list(DECOMP_PALETTE.keys()),
        palette=DECOMP_PALETTE, 
        ax=ax, width=0.7,
        edgecolor="black", linewidth=.4,
        errwidth=.7, capsize=0.3, dodge=True, saturation=0.95,
        zorder=3,
    )

    # Larger labels & ticks
    ax.set_ylabel("Logloss")
    ax.tick_params(axis='both')
    ax.grid(axis="y", color="#d0d0d0", linewidth=1, alpha=0.6)
    ax.axhline(0, color="#8e8e8e", linewidth=1.3)
    ax.legend(loc='upper right',fontsize=15,bbox_to_anchor=(1.4,1),frameon=False)
    ax.spines["top"].set_visible(False)
    import matplotlib as mpl

    # Bars already plotted
    for line in ax.lines:                    # whiskers/caps (Line2D)
        line.set_zorder(6)

    for coll in ax.collections:              # caps can be LineCollection on some versions
        if isinstance(coll, mpl.collections.LineCollection):
            coll.set_zorder(6)


def plot_gen_gap(df, methods=[
            'baseline_calib_logloss_vs_RP',
            'baseline_eb_logloss_vs_RP',
            'baseline_raking_logloss_vs_RP',
            'ippw_Utilde_logloss_vs_RP',
            'ub_trueUstar_logloss_vs_RP',
            'ub_predUstar_logloss_vs_RP'
            ]):
    PALETTE = sns.color_palette(["#264653", "#2a9d8f", "#E9C772", "#f4a261", "#e76f51",'#A23216']) #1C0221

    df = df.copy()
    if 'task' not in df.columns:
        df['task'] = '0'
    
    for c in methods:
        if c not in df.columns:
            print(f"Method {c} not found!! Setting to 0")
            df[c] = 0

    df = df[['task'] + methods].melt(
        id_vars=['task'],
        value_vars=methods,
        var_name='method',
        value_name='gap_logloss'
    )
    df['method'] = df['method'].apply(lambda m: METHOD_NAME_DICT['_'.join(m.split('_')[:2])])
    df['method'] = pd.Categorical(df['method'].astype(str), categories=[
                    r'UB (heuristic $U^*$)', r'Naive IPPW of $\tilde{U}$',
                    'Raking', 
                    'Calibration', 'Ent. balancing'], ordered=True)
    df = df.sort_values(['method'])

    plt.figure(figsize=(5, 4), dpi=300)
    ax = plt.gca()
    sns.barplot(
        data=df,
        y='gap_logloss',
        x='task',
        palette=PALETTE,
        hue='method',
        saturation=0.9,
        linewidth=.1,
        edgecolor='black',
        ax=ax,
        capsize=0.05

    )
    for line in ax.lines:
        line.set_linewidth(0.7)

    ax.set_xlabel("")
    ax.set_ylabel(r"$\hat{R}_Q - R_P$", fontsize=20, labelpad=8)
    ax.axhline(y=0, linewidth=1.5, color='grey')
    legend = ax.legend(title='', fontsize=14,bbox_to_anchor=(1.6,1.01))
    frame = legend.get_frame()
    frame.set_facecolor("white")  # light grey
    frame.set_edgecolor("none")     # no border
    frame.set_alpha(0.8)  
    ax.tick_params(axis='both', which='major', length=6)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=14)
    sns.despine(ax=ax)