import json
import functools
import matplotlib.pyplot as plt


class ExpAnalyzer:
    def __init__(self):
        plt.style.use('_mpl-gallery')
        plt.rcParams['figure.figsize'] = [1.5, 1.5]
        # self.metrics = ['ndcg_5', 'ndcg_10', 'ndcg_30', 'ndcg_60',
        #                 'mrr_5', 'mrr_10', 'mrr_30', 'mrr_60',
        #                 'dcg_5', 'dcg_10', 'dcg_30', 'dcg_60']


        self.metrics = ['ndcg_5',
                        'mrr_5',
                        'dcg_5']
        self.filename = 'experiment_result.json'
        self.dataprefix = './approxndcg'
        self.loss_name = 'approxndcg'
        self.postfix = '-lr0001-fold'
        self.exp_num = 1
        self.append_name = '/results/test_run_'
        self.alphas = [0.5, 1, 2, 3, 4, 5, 10]
        self.result = {}
        for alpha in self.alphas:
            self.result[(alpha, 'web10')] = []
            for i in range(1, self.exp_num + 1):
                filepath = f'{self.dataprefix}/{self.loss_name}{self.postfix}{i}-{alpha}{self.append_name}{alpha}/{self.filename}'
                f = open(
                    filepath,
                    'r')
                self.result[(alpha, 'web10')].append(json.load(f))

    def get_result(self, dataset='web10', alpha=0.5, metric_name='ndcg_5'):
        return functools.reduce(lambda x, y: x + y[f'val_metrics\\{metric_name}'], self.result[(alpha, dataset)],
                                0) / len(self.result[(alpha, dataset)])

    def plot_bar_chart(self, dataset='web10', metric_name='ndcg_5'):
        bar_data = []
        alpha_values = []
        for alpha in self.alphas:
            alpha_values.append(alpha)
            bar_data.append(self.get_result(dataset, alpha, metric_name))
        fig, ax = plt.subplots()
        alpha_values_string = [str(x) for x in alpha_values]
        # plot everything horizontally
        ax.barh(alpha_values_string, bar_data, color='b', edgecolor="white", linewidth=0.7)
        # ax.set_ylabel()
        ax.set_title(f'{dataset} {metric_name}')
        fig.savefig(f'imgs/{dataset}_{metric_name}.png')
        plt.show()


if __name__ == '__main__':
    exp = ExpAnalyzer()
    for num in range(5):
        print(exp.result[(0.5, 'web10')][num]['val_metrics\\ndcg_5'])
    print(exp.get_result('web10', 0.5, 'ndcg_5'))
    exp.plot_bar_chart('web10', 'ndcg_5')
