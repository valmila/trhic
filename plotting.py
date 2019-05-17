import argparse
from scipy import stats
import numpy as np
import pandas as pd
from cometml_api import api
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ipdb
sns.set()
sns.set_palette("colorblind")


def plot_from_comet(experiment_id='98Hhyb58cThYVpxaOvbL3Yu8S',
                    metric_name='trH1Cte',
                    metric_name_x=None,
                    plot_name=''):

    os.makedirs("plots/" + plot_name, exist_ok=True)

    #experiment = Experiment(api_key='98Hhyb58cThYVpxaOvbL3Yu8S', project_name='trhic', workspace='valthom')
    experiments = api.get_experiments(experiment_id)

    metric_names = api.get_metrics(experiments[0]["experiment_key"]).keys()

    metricsx = []
    metricsy = []
    #metric_namex = 'trace/H1Ctr'
    metric_namex = 'trace/H1Cte'
    metric_namey = 'loss/te'
    #metric_namey = 'loss/te'
    for i in range(len(experiments)):
        fig = plt.figure()
        param = api.get_params(experiments[i]["experiment_key"])
        metric = api.get_metrics(experiments[i]["experiment_key"])

        metricx = np.array(metric[metric_namex]['value'])
        metricy = np.array(metric[metric_namey]['value'])
        steps = np.array(metric[metric_namey]['step'])
        metricx = metricx[:len(steps)]
        min_len = min([len(metricx), len(metricy), len(steps)])
        metricx = metricx[:min_len]
        metricy = metricy[:min_len]
        steps = steps[:min_len]
        name = f"{metric_namex.replace('/', '_')}-{metric_namey.replace('/', '_')}_{param['model']}_bs={param['batch_size']}_lr={param['lr']}_datasize={param['dataset_size']}"
        if min_len > 10:
            df = pd.DataFrame({metric_namex: metricx, metric_namey: metricy, 'step':
                steps})
            sns.scatterplot(data=df, x=metric_namex, y=metric_namey, hue='step')
            #sns.jointplot(data=df, x=metric_namex, y=metric_namey)
            # g = sns.JointGrid(data=df, x=metric_namex, y=metric_namey)
            # g = g.plot_joint(plt.scatter, color="b", s=40, edgecolor="white")
            # g = g.plot_marginals(sns.distplot, kde=False, color="g")
            # rsquare = lambda a, b: stats.pearsonr(a, b)[0] ** 2
            # g = g.annotate(rsquare, template="{stat}: {val:.2f}", stat="$R^2$", loc="upper left", fontsize=12)
            fig.savefig(f"./plots/{name}.png")
        plt.close('all')
        print(f'Done {name}')
    return
        # do the scatter plot

        #sns.scatterplot(np.array(metric[metric_namex]['value'], np.array(metric[metric_namex]['value'])
    #     metricsx.append(
    #     metricsy.append(
    # dfx = metricsx[0]
    # for dfx_ in metricsx[1:]:
    #     dfx = dfx.append(dfx_)
    # dfy = metricsy[0]
    # for dfy_ in metricsy[1:]:
    #     dfy = dfy.append(dfy_)
    # ipdb.set_trace()
    #if metric_name is not 'ess_0':
        #metric_name += env_name
    for metric_name in metric_names:
        fig = plt.figure()
        for i in range(len(experiments)):
            ipdb.set_trace()
            param = api.get_params(experiments[i]["experiment_key"])
            metric = api.get_metrics(experiments[i]["experiment_key"])
            try:
                sns.lineplot(x="step", y="value", data=metric[metric_name])
            except:
                pass
        fig.savefig(f"./plots/{metric_name.replace('/', '_')}.png")
        plt.close('all')
        print(f'Done {metric_name}')


    # df = metrics[0]
    # for df_ in metrics[1:]:
    #     df = df.append(df_)
    #
    # df['category'] = df[param_names].apply(lambda x: '-'.join(x), axis=1).astype(str)
    # for group in param_names:
    #     os.makedirs("plots/" + plot_name + '/' + env_name + '/' + group, exist_ok=True)
    #
    #     for subgroup in set(df[group]):
    #         df_sub = df.loc[df[group] == subgroup]
    #         df_sub = df_sub.reset_index()
    #         sns.lineplot(x="step", y="value", hue="category", data=df_sub)
    #         plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #         plt.savefig('plots/' + plot_name + '/' + env_name + '/' + group + '/' + subgroup + '.pdf', bbox_inches='tight')
    #         plt.close()


if __name__ == '__main__':
    # project_ids = api.get_project_names_and_ids('alexpiche')
    parser = argparse.ArgumentParser(description='plotting')
    parser.add_argument('--metric_name', type=str, default='loss/gap',
                        help='')
    parser.add_argument('--params_name', default=['algo', 'model'], nargs='+')
    parser.add_argument('--plot_name', type=str, default='test3',
                        help='')

    args = parser.parse_args()

    trhic = 'f5ffcac825df402c991de3e3f1d5b676'
    trhic3 ='fa53a63863c54742b1b0b4781cf3a2cb'
    exp_id = trhic
    metric_name = 'trace/H1Cte'



    plot_from_comet(experiment_id=exp_id, metric_name=metric_name,
                        plot_name='')
