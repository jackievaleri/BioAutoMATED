import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob as glob
import scipy.stats as sp

import matplotlib
from matplotlib import rc
font = {'size'   : 16}
matplotlib.rc('font', **font)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

# change font
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
sns.set_style("whitegrid")

############## PART 1: STATISTICS FUNCTIONS ##############

def calculate_stats(df, additional_cols = None):
    for col in df.columns:
        if col == 'scr':
            continue
        if additional_cols is not None:
            if col in additional_cols:
                continue
        vals = df[col]
        orig = list(vals[df['scr'] == 'Original'])
        scr = list(vals[df['scr'] == 'Scrambled'])
        t = sp.ttest_ind(orig, scr)
        print('T-test for ' + col + ' model : ' + str(np.round(t[0], 3)) + ' with p-val ' + str(np.round(t[1],5))) 

def calculate_model_vs_model_stats(df):
    ak = list(df['AutoKeras'])
    ds = list(df['DeepSwarm'])
    tpot = list(df['TPOT'])
    print('Comparing DeepSwarm with AK')
    t = sp.ttest_ind(ds, ak)
    print('T-test: ' + str(np.round(t[0], 3)) + ' with p-val ' + str(np.round(t[1],5))) 
    print('Comparing DeepSwarm with TPOT')
    t = sp.ttest_ind(ds, tpot)
    print('T-test: ' + str(np.round(t[0], 3)) + ' with p-val ' + str(np.round(t[1],5))) 
    print('Comparing AK with TPOT')
    t = sp.ttest_ind(ak, tpot)
    print('T-test: ' + str(np.round(t[0], 3)) + ' with p-val ' + str(np.round(t[1],5))) 
    
def calculate_comp_method_stats(df, model_names = ['AutoKeras', 'DeepSwarm', 'TPOT']):
    orig = df[df['scr'] == 'Original']
    
    none = orig[orig['Class']=='none']
    rev = orig[orig['Class']=='rev']
    comp = orig[orig['Class']=='comp']
    both = orig[orig['Class']=='both']

    for model in model_names:
        print('\nComparing ' + model)
        print('Rev compared to none')
        t = sp.ttest_ind(list(rev[model]), list(none[model]))
        print('T-test: ' + str(np.round(t[0], 3)) + ' with p-val ' + str(np.round(t[1],5))) 
        print('Comp compared to none')
        t = sp.ttest_ind(list(comp[model]), list(none[model]))
        print('T-test: ' + str(np.round(t[0], 3)) + ' with p-val ' + str(np.round(t[1],5))) 
        print('Both compared to none')
        t = sp.ttest_ind(list(both[model]), list(none[model]))
        print('T-test: ' + str(np.round(t[0], 3)) + ' with p-val ' + str(np.round(t[1],5))) 
        print('Rev compared to comp')
        t = sp.ttest_ind(list(rev[model]), list(comp[model]))
        print('T-test: ' + str(np.round(t[0], 3)) + ' with p-val ' + str(np.round(t[1],5))) 
        print('Both compared to comp')
        t = sp.ttest_ind(list(both[model]), list(comp[model]))
        print('T-test: ' + str(np.round(t[0], 3)) + ' with p-val ' + str(np.round(t[1],5))) 
        print('Both compared to rev')
        t = sp.ttest_ind(list(both[model]), list(rev[model]))
        print('T-test: ' + str(np.round(t[0], 3)) + ' with p-val ' + str(np.round(t[1],5))) 
    return

############## PART 2: DATA PROCESSING FUNCTIONS ##############

def strip_output_text(folder, seq, model_type, model_names, file_name, val_name):
    valdf = pd.DataFrame()
    for model in model_names:
        df = pd.read_csv(folder + seq + 'outputs/' + model + '/' + model_type + '/' + file_name, sep = '\t', header = None)
        # manually correct so they have appropriate capitalization for later plotting
        model = model.replace('deepswarm', 'DeepSwarm')
        model = model.replace('autokeras', 'AutoKeras')
        model = model.replace('tpot', 'TPOT')
        vals = df.values
        for i in range(len(vals)):
            val = vals[i] # grab current line
            if val_name in val[0]:
                new = val[0].split(": ")[1]
                new = new.strip("[")
                new = new.strip("]")

                new = [s for s in new.split(' ')]
                new = [s for s in new if s != '']
                new = [0 if s == '0.' else s for s in new]
                new = [float(s) for s in new]
                valdf[model] = new
    return(valdf)

def calc_val(folder, seq, val_name, model_type, model_names = ['autokeras', 'deepswarm', 'tpot']):
    valdf = strip_output_text(folder, seq, model_type, model_names, 'all_results.txt', val_name)
    valdf['scr'] = ['Original'] * len(valdf)
    
    scr_valdf = strip_output_text(folder, seq, model_type, model_names, 'scrambled/all_scrambled_control_results.txt', val_name)
    scr_valdf['scr'] = ['Scrambled'] * len(scr_valdf)
    valdf = pd.concat([valdf, scr_valdf])
   
    print('Running statistics now...')
    calculate_stats(valdf)

    return(valdf)

def calc_robustness_vals(folder, seq, val_name, model_type, max_len, model_names = ['autokeras', 'deepswarm', 'tpot']):
    # initialize with basic vals
    print('Testing Scrambled vs Original Models: ')
    print('\nTesting Dataset Size: ' + str(max_len))
    valdf = calc_val(folder, seq, val_name, model_type, model_names = model_names)
    valdf['datasize'] = [float(max_len)] * len(valdf)
    
    already_assayed = []
    for model in model_names:
        for filename in glob.glob(folder + seq + 'outputs/' + model + '/' + model_type +  '/robustness/*' +  '_results.txt'):
            clean_filename = filename.split('robustness/')[1]
            if clean_filename in already_assayed:
                continue # skip those already counted in one of the model folders (else df will be 3x repeated - messes up stats)
            if 'scrambled' in filename:
                scrambled = 'Scrambled'
            else:
                scrambled = 'Original'
            
            curr_valdf = strip_output_text(folder, seq, model_type, model_names, 'robustness/' + clean_filename, val_name)
            datasize = clean_filename.split('_')[0]
            curr_valdf['scr'] = scrambled
            curr_valdf['datasize'] = [float(datasize)] * len(curr_valdf)
            valdf = pd.concat([valdf, curr_valdf])
            already_assayed.append(clean_filename)
            
    # now do stats
    valdf = valdf.sort_values('datasize', ascending = False)
    for i, smalldf in valdf.groupby('datasize', sort = False):
        if i == max_len:
            continue
        print('\nTesting Dataset Size: ' + str(i))
        calculate_stats(smalldf, ['datasize'])
        
    print('\nTesting Different Models at Each Value: ')
    # now do stats
    valdf = valdf.sort_values('datasize', ascending = False)
    for i, smalldf in valdf.groupby('datasize', sort = False):
        smalldf = smalldf[smalldf['scr'] == 'Original'] # just with original model
        print('\nTesting Dataset Size: ' + str(i))
        calculate_model_vs_model_stats(smalldf)    
    return(valdf)

def reshape_for_plotting(df, additional_cols = None):
    new = pd.DataFrame()
    new['model_type'] = []
    index = 0
    for name, col in df.iteritems():
        if 'scr' in name:
            continue
        if additional_cols is not None:
            if name in additional_cols:
                continue
        new_pd = pd.DataFrame(col)
        new_pd.columns = ['values']
        new_pd['scr'] = list(df['scr'])
        new_pd['model_type'] = [name] * len(new_pd)
        if additional_cols is not None:
            for add_col in additional_cols:
                new_pd[add_col] = list(df[add_col])
        new = pd.concat([new, new_pd])
        index = index + 1
    return(new)

def calc_time(folder, seq, model_type, task = 'autoML search', model_names = ['autokeras', 'deepswarm', 'tpot']):
    folds = ['outputs/autokeras/binary_classification/', 'outputs/deepswarm/binary_classification/', 'outputs/tpot/binary_classification/']
    time = pd.DataFrame()
    for model in model_names:
        df = pd.read_csv(folder + seq + 'outputs/' + model + '/' + model_type + '/' + 'runtime_statistics.txt', sep = '\t', header = None)
        model = model.replace('deepswarm', 'DeepSwarm')
        model = model.replace('autokeras', 'AutoKeras')
        model = model.replace('tpot', 'TPOT')
        vals = df.values
        for i in range(len(vals)):
            val = vals[i]
            if task not in val[0]:
                continue
            new = val[0].split("Elapsed time for " + task + " : ")[1].split(" minutes")[0]
            new = float(new)
            time[model] = [new]
    return(time)

############## PART 3: PLOTTING FUNCTIONS ##############

def plot_barplot(valdf, savepath, yaxis):
    valdf = reshape_for_plotting(valdf)  

    plt.figure(figsize=(5,4), dpi=300)
    my_pal = {"Original": "cornflowerblue", "Scrambled": "lightgrey"}
    ax = sns.barplot(x = 'model_type', y = 'values', hue = 'scr', data=valdf, edgecolor='black', alpha = 1, linewidth = 1, palette = my_pal, ci = 'sd', errcolor = 'black', errwidth = 1.5, capsize = 0.2, saturation = 0.6)
    sns.swarmplot(x = 'model_type', y = 'values', hue = 'scr', data=valdf, dodge = True, edgecolor='black', alpha = 1, linewidth = 1, s = 6, palette = my_pal)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles = handles[2:4], labels = labels[2:4], title='Data', loc = 'upper left', bbox_to_anchor=(1.1, 1.05))    

    plt.xlabel('')
    plt.xticks(rotation = 80)
    plt.ylabel(yaxis)
    plt.ylim([0, 1.1])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()
    
def plot_multicolumn_barplot(valdf, savepath, yaxis = None, multiclass = False, regression = False, add_rows = False, barplot = True):
    
    if multiclass:
        add_cols = ['Class']
    if regression:
        add_cols = ['Metric']
    if add_rows:
        add_cols = ['Class', 'Length']
    valdf = reshape_for_plotting(valdf, add_cols)  
    my_pal = {"Original": "cornflowerblue", "Scrambled": "lightgrey"}
    valdf['Model'] = valdf['model_type']
    xaxis = add_cols[0]
    if yaxis == None:
        yaxis = 'Value'
    valdf[yaxis] = valdf['values']
    if not add_rows:
        g = sns.FacetGrid(valdf, col="Model", height=6, aspect=1, margin_titles=True)
    else:
        g = sns.FacetGrid(valdf, col="Length", row = "Model", height=6, aspect=1, margin_titles=True)
    if barplot:
        g.map(sns.barplot, xaxis, yaxis, 'scr', dodge = True, edgecolor='black', alpha = 1, linewidth = 1, palette = my_pal, ci = 'sd', errcolor = 'black', errwidth = 1.5, capsize = 0.2, saturation = 0.6)
    g.map(sns.swarmplot, xaxis, yaxis, 'scr', dodge = True, edgecolor='black', alpha = 1, linewidth = 1, s = 10, palette = my_pal)
    for ax in g.axes_dict.values():
        handles, labels = ax.get_legend_handles_labels()
        if barplot:
            ax.legend(handles[2:4], labels[2:4], title='Data', loc = 'upper right')            
        else:
            ax.legend(loc = 'upper right')
        plt.xlabel('')
        plt.xticks(rotation = 80)
        plt.ylabel(yaxis)
        plt.ylim([0, 1.1])
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()
    
def plot_lineplot(valdf, savepath, yaxis):
    valdf = reshape_for_plotting(valdf, additional_cols = ['datasize'])
    valdf['$\it{Model}$'] = valdf['model_type']
    valdf['$\it{Data}$'] = valdf['scr']
    plt.figure(figsize=(7,4), dpi=300)
    my_pal = {"AutoKeras": "cornflowerblue", "DeepSwarm": "sandybrown", "TPOT": "grey"}
    ax = sns.lineplot(x = 'datasize', y = 'values', hue = '$\it{Model}$', style = '$\it{Data}$', data=valdf,  alpha = 1, linewidth = 2,  ci = 'sd', palette = my_pal)
    sns.scatterplot(x = 'datasize', y = 'values', hue = '$\it{Model}$', data=valdf, edgecolor='black', alpha = 1, linewidth = 0.4, s = 12, palette = my_pal)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:7], labels[:7], bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.xlabel('Dataset Size')
    plt.xticks(rotation = 60)
    plt.ylabel(yaxis)
    ax.set_xscale('log')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()
    
def plot_one_element_time(valdf, savepath, yaxis):
    plt.figure(figsize=(3,4), dpi=300)
    my_pal = {"AutoKeras": "cornflowerblue", "DeepSwarm": "sandybrown", "TPOT": "grey"}
    ax = sns.barplot(data=valdf, edgecolor='black', alpha = 1, linewidth = 1, palette = my_pal)
    plt.xlabel('')
    plt.xticks(rotation = 80)
    plt.ylabel(yaxis)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()
    
def plot_all_time(valdf, savepath, yaxis):
    fig, ax = plt.subplots(figsize=(6,4), dpi=300)
    colors = ['navy', 'cornflowerblue', 'sandybrown', 'rosybrown', 'grey']
    valdf.plot(kind='bar', stacked=True, color = colors, ax = ax, edgecolor='black', alpha = 1, linewidth = 1)
    plt.xlabel('')
    plt.xticks(rotation = 80)
    plt.ylabel(yaxis)
    ax.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()
############## END OF FILE ##############