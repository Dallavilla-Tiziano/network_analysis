import pandas as pd
from reactome2py import analysis, content, utils
import mygene
import urllib.parse
import urllib.request
from joblib import Parallel, delayed
from pathlib import Path
import subprocess
import glob
import networkx as nx
import statistics
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import scipy.stats as st
from statsmodels.stats.multitest import fdrcorrection


class NetworkSearchAnalysis(object):

    def __init__(self, graphml_path, biopax_path, metabolites_path, genes_path):

        self.cores = 20
        self.mg = mygene.MyGeneInfo()
        self.graphml_models_path = graphml_path
        self.biopax_models_path = biopax_path

        # Load gene list into 2 dictionaries (key:uniprot values:ensmblid)
        # and (key:uniprot values:symbol)
        self.gene_list = pd.read_csv(genes_path, names=['ensmbl'])
        self.gene_list = list(self.gene_list['ensmbl'])
        # self.uniprot2ensmbl, self.uniprot2symbol, self.symbol2uniprot, self.missing = self.ensmbl2uniprot(self.gene_list)

        # Load metabolites into dictionary (key:chebi values:chemical names)
        self.metabolites_path = metabolites_path

    def map_genes2pathways(self, gene_list, scopes):
        query = self.mg.querymany(gene_list,
                                  scopes=scopes,
                                  species=9606,
                                  fields='uniprot,symbol,pathway',
                                  returnall=True,
                                  as_dataframe=True)

        gene2pathways = {}
        uniprot2ensmbl = {}
        reactome2pathway_names = {}
        hits = query['out'].dropna(subset=['pathway.reactome'])
        for index, row in hits.iterrows():
            pathways = []
            for d_ in row['pathway.reactome']:
                pathways.append(d_['id'])
                reactome2pathway_names[d_['id']] = d_['name']
            if isinstance(row['uniprot.Swiss-Prot'], list):
                for id_ in row['uniprot.Swiss-Prot']:
                    gene2pathways[id_] = pathways
                    uniprot2ensmbl[id_] = index
            else:
                gene2pathways[row['uniprot.Swiss-Prot']] = pathways
                uniprot2ensmbl[row['uniprot.Swiss-Prot']] = index
        return gene2pathways, uniprot2ensmbl, reactome2pathway_names

    def ensmbl2uniprot(self, gene_list):
        '''Convert a list of ensmbl ids to uniprot ids.'''
        uniprot2ensmbl = {}
        uniprot2symbol = {}
        symbol2uniprot = {}
        query = self.mg.querymany(gene_list,
                                  scopes='ensemblgene',
                                  species=9606,
                                  fields='uniprot,symbol',
                                  returnall=True)
        # Keep list of ensmbl ids that couldn't be associated with uniprot id:
        missing = query['missing']
        for entry in query['out']:
            if ('uniprot' in entry):
                if 'symbol' in entry:
                    symbol = entry['symbol']
                else:
                    symbol = entry['query']
                if ('Swiss-Prot' in entry['uniprot']):
                    ids = entry['uniprot']['Swiss-Prot']
                elif ('TrEMBL' in entry['uniprot']):
                    ids = entry['uniprot']['TrEMBL']

                if isinstance(ids, list):
                    for gene_id in ids:
                        uniprot2ensmbl[gene_id] = entry['query']
                        uniprot2symbol[gene_id] = symbol
                        symbol2uniprot[symbol] = gene_id
                else:
                    uniprot2ensmbl[ids] = entry['query']
                    uniprot2symbol[ids] = symbol
                    symbol2uniprot[symbol] = ids
        return uniprot2ensmbl, uniprot2symbol, symbol2uniprot, missing

    @staticmethod
    def get_reactome_pathway_from_uniprot(uniprot_id):
        '''Query Reactome for low level pathways associated with a given uniprot id'''
        gene2pathways = {}
        reactome2pathway_names = {}
        reactome_ids = []
        query = content.interactors_acc_pathways(uniprot_id)
        if query:
            for pathway in query:
                reactome_ids.append(pathway['stId'])
                reactome2pathway_names[pathway['stId']] = pathway['displayName']
        gene2pathways[uniprot_id] = reactome_ids
        return gene2pathways, reactome2pathway_names

    def map_genes_to_pathway(self, uniprot_ids):
        '''Query a list of uniprot ids for associated low level Reactom pathways'''
        results = Parallel(n_jobs=self.cores)(delayed(self.get_reactome_pathway_from_uniprot)
                                             (key) for key in uniprot_ids)
        gene2pathways = {}
        reactome2pathway_names = {}
        for result in results:
            gene2pathways.update(result[0])
            reactome2pathway_names.update(result[1])
        return gene2pathways, reactome2pathway_names

    def make_table(self, pathway):
        pathway_table = pd.DataFrame()
        i=0
        for gene in pathway['genes']:
            pathway_table.loc[i,'gene'] = gene
            pathway_table.loc[i,'pathway'] = pathway['stId']
            i=i+1
        return pathway_table

    def map_genes_to_pathway_alternative(self, symbol2uniprot):
        '''Query a list of uniprot ids for associated low level Reactom pathways'''
        pathways = utils.gene_mappings()
        results = Parallel(n_jobs=self.cores, require='sharedmem')(delayed(self.make_table)(pathway) for pathway in pathways)
        pathway_table = pd.DataFrame()
        for result in results:
            pathway_table = pd.concat([pathway_table, result])
        gene_symbol2reactome = {}
        for name, group in pathway_table.groupby('gene'):
            gene_symbol2reactome[name] = list(group['pathway'])
        gene2pathway = {}
        for gene in symbol2uniprot:
            if gene in gene_symbol2reactome:
                gene2pathway[gene] = gene_symbol2reactome[gene]
        return gene2pathway

    def load_pathways(self, pathways_list):
        '''Load a list of pathway as NetworkX undirected graph'''
        pathways_graphs = {}
        g_directed = {}
        for pathway in pathways_list:
            try:
                graph = nx.read_graphml(f'{self.graphml_models_path}/{pathway}.graphml')
                pathways_graphs[pathway] = graph.to_undirected()
                g_directed[pathway] = graph
            except Exception as e:
                print(f'pathway {pathway} not found!')
        return pathways_graphs, g_directed

    def map_chebi_to_pathway(self, pathways_graphs):

        metabolites_list = pd.read_csv(self.metabolites_path)
        self.metabolites_dict = {}
        for index, row in metabolites_list.iterrows():
            self.metabolites_dict[row['ChEBI']] = row['CHEMICAL_NAME']

        reactome2chebi = {}
        reactome2nodeid_chebi = {}
        for pathway in pathways_graphs:
            nodes = dict(pathways_graphs[pathway].nodes(data=True))
            chebi_in_pathway = []
            chebi_id_map = []
            for node_id in nodes:
                node = nodes[node_id]
                if 'chebi' in node:
                    chebis = node['chebi'].split(';')
                    for chebi in chebis:
                        chebi_in_pathway.append(chebi.split(':')[-1])
                        chebi_id_map.append((node_id, chebi.split(':')[-1]))
            chebi_in_pathway = set(chebi_in_pathway)
            reactome2chebi[pathway] = list(chebi_in_pathway)
            reactome2nodeid_chebi[pathway] = chebi_id_map
        return reactome2chebi, reactome2nodeid_chebi

    def findTargetChebiInPathways(self, reactome2chebi, chebi_ids):
        target_chebis2pathways = {}
        for pathway_id in reactome2chebi:
            chebi_list = reactome2chebi[pathway_id]
            target_chebis = []
            for chebi in chebi_ids:
                chebi_id = chebi.split(':')[-1]
                if chebi_id in chebi_list:
                    target_chebis.append(chebi_id)
            target_chebis2pathways[pathway_id] = target_chebis
        return target_chebis2pathways

    def mapIdToAnalyze(self, target_chebis2pathways, reactome2nodeid_chebi):
        ids_to_analyze = {}
        for pathway in target_chebis2pathways:
            chebis = target_chebis2pathways[pathway]
            target_ids = []
            tuples = reactome2nodeid_chebi[pathway]
            if chebis:
                for chebi in chebis:
                    for element in tuples:
                        if chebi in element:
                            target_ids.append(element[0])
            if target_ids:
                ids_to_analyze[pathway] = target_ids
        return ids_to_analyze

    def calculate_random_t_stats(self, ids_to_analyze, pathways_graphs):
        random_t_stats = {}
        for pathway in ids_to_analyze:
            graph = pathways_graphs[pathway]
            graph_nodes = dict(graph.nodes(data=True))
            graph_degrees=dict(graph.degree())
            # mean_degree = statistics.mean(graph_degrees.values())
            #determine the size of the sample to calculate test statistics
            sample_size = len(ids_to_analyze[pathway])
            #find metabolites ids in the pathway network
            pathway_metabolites = []
            for node in list(graph.nodes(data=True)):
                if 'chebi' in node[1]:
                    pathway_metabolites.append(node[0])
            #calculate N random test statistics
            random_test_statistics = []
            for i in range(0, 500):
                #get gene nodes with distance 3 or less from a metabolite of interest
                test_statistic = 0
                if len(pathway_metabolites)>=sample_size:
                    for node in random.sample(pathway_metabolites, sample_size):
                        distances = nx.shortest_path_length(graph, source = node)
                        valid_distances = {key:val for key, val in distances.items() if val <= 3}
                        #remove nodes with degree higher tha 3 time the average degree in the pathway network
                        valid_distances_filtered = valid_distances.copy()
                        # for key in valid_distances:
                            # if graph_degrees[key]/mean_degree > 3
                            #     valid_distances_filtered.pop(key)
                        #remove nodes that are not genes
                        valid_distances_uniprot = valid_distances_filtered.copy()
                        valid_genes = []
                        for key in valid_distances_filtered:
                            if ('uniprot' in graph_nodes[key]):
                                uniprot_ids = graph_nodes[key]['uniprot'].split(';')
                                gene_of_interest = [id_ in list(self.uniprot2ensmbl.keys()) for id_ in uniprot_ids]
                                if not any([id_ in list(self.uniprot2ensmbl.keys()) for id_ in uniprot_ids]):
                                    valid_distances_uniprot.pop(key)
                                else:
                                    valid_genes = valid_genes + list(np.array(uniprot_ids)[gene_of_interest])
                            else:
                                valid_distances_uniprot.pop(key)
                        # test_statistic = test_statistic + len(valid_distances_uniprot)
                        test_statistic = test_statistic + len(valid_genes)
                random_test_statistics.append(test_statistic)
            if len(pd.Series(random_test_statistics).value_counts()) > 5:
                random_t_stats[pathway] = random_test_statistics
        return random_t_stats

    # def calculate_b_centrality(self, ids_to_analyze, pathways_graphs):


    def calculate_t_stats(self, ids_to_analyze, pathways_graphs):
        pathways_distances_uniprot = {}
        pathway_test_statistics = {}
        for pathway in ids_to_analyze:
            graph = pathways_graphs[pathway]
            graph_nodes = dict(graph.nodes(data=True))
            graph_degrees=dict(graph.degree())
            mean_degree = statistics.mean(graph_degrees.values())
            #calculate test statistics for each observed metabolite
            test_statistic = 0
            for metabolite in ids_to_analyze[pathway]:
                chebiid = pathways_graphs[pathway].nodes(data=True)[metabolite]['chebi']
                distances = nx.shortest_path_length(pathways_graphs[pathway], source = metabolite)
                valid_distances = {key:val for key, val in distances.items() if val <= 3}
                #remove nodes with degree higher tha 3 time the average degree in the pathway network
                valid_distances_filtered = valid_distances.copy()
                # for key in valid_distances:
                #     if graph_degrees[key]/mean_degree > 3:
                #         valid_distances_filtered.pop(key)
                #remove nodes that are not genes
                valid_genes = []
                valid_distances_uniprot = valid_distances_filtered.copy()
                for key in valid_distances_filtered:
                    if ('uniprot' in graph_nodes[key]):
                        uniprot_ids = graph_nodes[key]['uniprot'].split(';')
                        gene_of_interest = [id_ in list(self.uniprot2ensmbl.keys()) for id_ in uniprot_ids]
                        if not gene_of_interest:
                            valid_distances_uniprot.pop(key)
                        else:
                            valid_genes = valid_genes + list(np.array(uniprot_ids)[gene_of_interest])
                    else:
                        valid_distances_uniprot.pop(key)
                pathways_distances_uniprot[pathway,chebiid] = list(set(valid_genes))
                # test_statistic = test_statistic + len(valid_distances_uniprot)
                test_statistic = test_statistic + len(valid_genes)
            pathway_test_statistics[pathway] = test_statistic
        return pathways_distances_uniprot, pathway_test_statistics

    def best_fit_distribution(self, data, pathway_id, bins=200, ax=None):
        results = {}
        """Model data by finding best fit distribution to data"""
        # Get histogram of original data
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        # Best holders
        best_distributions = []
        distributions = ['alpha',
                        'anglit',
                        'burr',
                        'expon',
                        'exponnorm',
                        'f',
                        'fisk',
                        'gamma',
                        'genpareto',
                        'gibrat',
                        't',
                        'norm']
        # Estimate distribution parameters from data
        for ii, distribution in enumerate([d for d in distributions if not d in ['levy_stable', 'studentized_range']]):
            distribution = getattr(st, distribution)
            # fit dist to data
            try:
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                best_distributions.append((distribution, params, sse))
            except Exception:
                pass
        results[pathway_id] = sorted(best_distributions, key=lambda x:x[2])
        return results

    def fit_distribution(self, random_t_stats):
        results = Parallel(n_jobs=40, require='sharedmem')(delayed(self.best_fit_distribution)(pd.Series(random_t_stats[key]), key, bins=10) for key in random_t_stats.keys())
        fitting = {}
        for result in results:
            fitting.update(result)
        return fitting

    def make_pdf(self, dist, params, size=10000):
        """Generate distributions's Probability Distribution Function """

        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Get sane start and end points of distribution
        start = dist.ppf(0.001, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.001, loc=loc, scale=scale)
        end = dist.ppf(0.999, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.999, loc=loc, scale=scale)

        # Build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        y = dist.pdf(x, loc=loc, scale=scale, *arg)
        pdf = pd.Series(y, x)

        return pdf

    def get_pathway_dist(self, random_t_stats, fitting):
        pathway_dist = {}
        pdfs = {}
        for key in random_t_stats:
            best_dist = fitting[key][0]
            pathway_dist[key] = best_dist
            # Make PDF with best params
            try:
                pdf = self.make_pdf(best_dist[0], best_dist[1])
                pdfs[key] = pdf
            except:
                print(f'pdf failed for pathway {key}')
        return pathway_dist, pdfs

    def make_t(self, dist, params, t):
        """Generate distributions's Probability Distribution Function """

        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        y = 1-dist.cdf(t, loc=loc, scale=scale, *arg)
        t_95 = dist.ppf(0.95, loc=loc, scale=scale, *arg)

        return y, t_95

    def get_p_value(self, pathway_dist, pathway_test_statistics):
        p_dict = {}
        p_tresh = {}
        for key in pathway_dist.keys():
            p_value, t_95 = self.make_t(pathway_dist[key][0], pathway_dist[key][1], pathway_test_statistics[key])
            if pathway_test_statistics[key] != 0:
                p_dict[key] = p_value
                p_tresh[key] = t_95
        statistics_df = pd.DataFrame(p_dict, index=['p_value']).T
        statistics_df.dropna(axis=0,inplace=True)
        data = fdrcorrection(statistics_df['p_value'])[1]
        statistics_df['q_value'] = data
        return statistics_df, p_tresh

    def make_results_table(self, statistics_df, pathways_distances_uniprot):
        significant_pathways_continuum = pd.DataFrame(columns=['pathway_id',
                                                      'pathway_name', 'metabolite', 'genes_uniprot',
                                                      'genes_symbol', 'q_value'])
        i=0
        for pathway in statistics_df[statistics_df['q_value'] <= 0.2].index:
            for keys in pathways_distances_uniprot.keys():
                if (pathway in keys) and (pathways_distances_uniprot[keys]):
                    significant_pathways_continuum.loc[i, 'pathway_id'] = pathway
                    significant_pathways_continuum.loc[i, 'pathway_name'] = self.reactome2pathway_names[pathway]
                    significant_pathways_continuum.at[i, 'metabolite'] = keys[1]
                    significant_pathways_continuum.at[i, 'genes_uniprot'] = pathways_distances_uniprot[keys]
                    symbols = []
                    # WHY THIS STOPPED WORKING????
                    # for gene in pathways_distances_uniprot[keys]:
                    #     symbols.append(self.uniprot2symbol[gene])
                    # significant_pathways_continuum.at[i, 'genes_symbol'] = symbols
                    significant_pathways_continuum.at[i, 'q_value'] = statistics_df.loc[pathway, 'q_value']
                    i=i+1
        return significant_pathways_continuum


    def run_analysis(self):

        print('mapping genes to pathway...')
        self.gene2pathways, self.reactome2pathway_names = self.map_genes_to_pathway(self.uniprot2ensmbl.keys())
        # self.gene2pathways = self.map_genes_to_pathway_alternative(self.symbol2uniprot)
        print('done!')
        print('loading pathways...')
        pathways_to_load = [x for x in set(sum(self.gene2pathways.values(), [])) if x not in ['R-HSA-9752946', 'R-HSA-9759194']]  # NEEDS to be removed in future versions
        self.pathways_graphs, self.g_directed = self.load_pathways(pathways_to_load)
        print('done!')
        print('mapping metabolites to pathways...')
        self.reactome2chebi, self.reactome2nodeid_chebi = self.map_chebi_to_pathway(self.pathways_graphs)
        print('done!')
        print('select metabolites to analyze')
        self.target_chebis2pathways = self.findTargetChebiInPathways(self.reactome2chebi, self.metabolites_dict)
        print('done!')
        print('get metabolites networks ids...')
        self.ids_to_analyze = self.mapIdToAnalyze(self.target_chebis2pathways, self.reactome2nodeid_chebi)
        print('done!')
        print('create random test statistics distributions...')
        self.random_t_stats = self.calculate_random_t_stats(self.ids_to_analyze, self.pathways_graphs)
        print('done!')
        print('calculate test statistics of observables...')
        self.pathways_distances_uniprot, self.pathway_test_statistics = self.calculate_t_stats(self.ids_to_analyze, self.pathways_graphs)
        print('done!')
        print('fitting random test statistics distributions...')
        self.fitting = self.fit_distribution(self.random_t_stats)
        print('done!')
        print('get valid genes...')
        self.pathway_dist, self.pdfs = self.get_pathway_dist(self.random_t_stats, self.fitting)
        print('done!')
        print('calculate p-values ...')
        self.statistics_df, self.p_tresh = self.get_p_value(self.pathway_dist, self.pathway_test_statistics)
        print('done!')
        print('generate results...')
        self.final_table = self.make_results_table(self.statistics_df, self.pathways_distances_uniprot)
        print('finished!')
