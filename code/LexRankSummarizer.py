import pandas as pd
import numpy as np
import MeCab
import neologdn
import re
import matplotlib.pyplot as plt
import itertools
from sklearn.feature_extraction import text
from sklearn.preprocessing import normalize
import networkx as nx

# テキスト前処理 & 名詞抽出
# https://qiita.com/menon/items/f041b7c46543f38f78f7
def preprocessing(text):
    text = neologdn.normalize(text.lower())
    text = re.sub("《.+?》","",text)
    text = re.sub("\[.+?\]","",text)
    text = re.sub("※","",text)
    return text

def extract_nouns_from_text(text, word_length_thresh=2):
    m = MeCab.Tagger()
    parse = m.parse(text).split("\n")
    nouns = list()
    for p in parse:
        parse_split = re.split("\t|,",p)
        if "EOS" in parse_split:
            break
        word = parse_split[0]
        tag = parse_split[1]
        info = parse_split[2]
        if ((len(word) >=word_length_thresh) & (tag=="名詞") & (info=='一般' or info=='固有名詞')):
            nouns.append(word)
    return nouns
def create_text_table(TEXT_FILE):
    with open(TEXT_FILE, 'r') as f:
         txt = f.read()

    text_split = re.split('\n',txt)
    text_split_select = [sent for sent in text_split if (len(sent)>1) and (sent[0] == "\u3000") and (sent[1] != "\u3000")]
    
    # text data frame
    text_split_select_clean = [preprocessing(text) for text in text_split_select]
    nouns_list = [extract_nouns_from_text(text) for text in text_split_select_clean]
    
    id_labels = ["{0:02d}".format(idx) for (idx, _) in enumerate(text_split_select)]
    df_text = pd.DataFrame({'id':id_labels,
                            'text':text_split_select,
                            'text_clean':text_split_select_clean,
                            'nouns':nouns_list}).copy()
    return df_text

# TF-IDF
def compute_bow_tfidf(texts:pd.Series):
    texts = texts.reset_index(drop=True).copy()

    # bow
    bow_tf = text.CountVectorizer()
    bow = bow_tf.fit_transform(texts)
    bow_arr = bow.toarray()

    # bow normalzie (l2)
    l2 = normalize(bow,norm='l2',axis=0)
    l2_arr = l2.toarray()

    # tf-idfの計算
    tfidf_tf = text.TfidfTransformer(norm=None)
    tfidf = tfidf_tf.fit_transform(bow)
    tfidf_arr = tfidf.toarray()

    # 格納
    df_words = pd.DataFrame()
    f_names = bow_tf.get_feature_names()
    arrs = [bow_arr, l2_arr, tfidf_arr]
    arr_names = ['bow','bow_l2','tfidf']
    for (arr, name) in zip(arrs, arr_names):
        df_tmp = pd.DataFrame(arr,columns=f_names)
        df_tmp['label'] = [name for _ in df_tmp.index]
        df_words = pd.concat([df_words, df_tmp])
    return df_words

# 固有ベクトルの中心性
# https://analytics-note.xyz/graph-theory/eigenvector-centrality/
def compute_eigenvector_centrality(adj_matrix):
    points_list = [np.ones(shape=adj_matrix.shape[0])]
    diff_thresh = 1e-3
    i = 0
    while True:
        # ポイントに隣接行列をかける
        points = points_list[i]
        points = points@adj_matrix
        # l2正規化
        points /= np.linalg.norm(points)
        points_list.append(points)

        # beforeとの差分
        if i != 0:
            diff = np.max(np.sqrt((points - points_bef)**2))
            if diff <= diff_thresh:
                print('convergence', 'loop_n=%s'%str(i))
                break

        # loop 設定
        if i == 1000:
            break
        points_bef = points
        i+=1
    return points

# 類似関数
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 関係グラフの構築
def compute_graph_from_similarity(df_vec, id_columns='id', edge_threshold=0.1):
    # 類似度計算
    id_labels = df_vec[id_columns].tolist()
    combinations = itertools.combinations(id_labels, 2)
    sim_results = list()
    for (sent_id1, sent_id2) in combinations:
        df_v1 = df_vec[df_vec[id_columns]==sent_id1].drop(id_columns, axis=1).copy()
        v1 = df_v1.iloc[0].tolist()
        df_v2 = df_vec[df_vec[id_columns]==sent_id2].drop(id_columns, axis=1).copy()
        v2 = df_v2.iloc[0].tolist()

        if not (df_v1.shape[0] == 1) and (df_v2.shape[0] == 1):
            print('warning ',sent_id1, sent_id2)
        sim = cos_sim(v1, v2)
        sim_tuple = (sent_id1, sent_id2, sim)
        sim_results.append(sim_tuple)
    
    # 類似度よりネットワークを構築
    graph = nx.Graph()
    graph.add_nodes_from(id_labels, size=len(id_labels))
    for i, j, w in sim_results:
        if w > edge_threshold:
            graph.add_edge(i, j, weight=w)

    # 隣接行列の作成
    adj_matrix = nx.to_numpy_array(graph)
    adj_matrix_binary = adj_matrix.copy()
    adj_matrix_binary[adj_matrix_binary>0] = 1
    return graph, id_labels, adj_matrix, adj_matrix_binary

# 重要度の高い文を抽出
def extract_hub_nodes(id_labels, adj_matrix, sim_thresh_top=0.1):
    # 類似度の平均
    sim_mean = adj_matrix.mean(axis=1)

    # 平均値の高い順にソート
    sim_sort_index = np.argsort(-sim_mean)
    id_labels_sort = np.array(id_labels)[sim_sort_index]
    sim_mean_sort = sim_mean[sim_sort_index]

    # 上位ノードの抽出
    sim_thresh_top = 0.1
    ids = (sim_mean_sort>=sim_thresh_top).astype(int).sum()
    id_labels_sort_top = id_labels_sort[:ids]
    
    # 上位ノードのインデックス
    sort_top_index = [id_labels.index(node) for node in id_labels_sort_top]
    return id_labels_sort_top, sort_top_index

# 重要かつ互いに類似しない文を抽出
def select_hub_nodes(graph, eigenvector_centrality, id_labels_sort_top, sort_top_index, sim_thresh_div=0.4):
    # 固有ベクトル中心値の取得
    eigenvector_centrality_sort_top = np.array([eigenvector_centrality[index] for index in sort_top_index])
    
    # 固有ベクトル中心値のソート
    cent_sort_index = np.argsort(-eigenvector_centrality_sort_top)
    id_labels_sort_top_sort = id_labels_sort_top[cent_sort_index]
    eigenvector_centrality_sort_top_sort = eigenvector_centrality_sort_top[cent_sort_index]
    
    # 重要度の高い & 類似度の低いノードを抽出
    nodes = list(id_labels_sort_top_sort)
    graph_edges = list(graph.edges(data=True))
    i = 0
    nodes_select = list()
    while True:
        # 選択済みノード
        node_select = nodes[0]
        nodes_select.append(node_select)
        nodes_compare = nodes[1:]
        nodes_next = list()
        for node_compare in nodes_compare:
            sim_list = [sim[2]['weight'] for sim in graph_edges if ((node_select in sim)&(node_compare in sim))]
            if len(sim_list) == 0:
                sim = 0
            else:
                sim = sim_list[0]
            if sim < sim_thresh_div:
                nodes_next.append(node_compare)
        # ループ停止判定
        i+=1
        if (i ==1000) or (1>=len(nodes_next)):
            print('nodes_next:',len(nodes_next))
            break
        nodes = nodes_next
        
    # 対象ノードの固有ベクトル中心値を抽出
    select_index = [list(id_labels_sort_top_sort).index(node) for node in nodes_select]
    cents_select = eigenvector_centrality_sort_top_sort[select_index]
    return nodes_select, cents_select

# 文章要約関数
def summarize_text(df_text, edge_threshold, sim_thresh_top, sim_thresh_div, id_columns='id'):
    # text 基本情報
    id_labels = np.array(df_text[id_columns].tolist())
    
    # tfidf ベクトル
    df_text['nouns_join'] = df_text['nouns'].map(lambda x : ' '.join(x)) 
    nouns_join = df_text.nouns_join
    df_words = compute_bow_tfidf(nouns_join)
    df_vec = df_words[df_words['label']=='tfidf'].copy()
    df_vec = df_vec.drop('label',axis=1).reset_index(drop=True).copy()
    df_vec[id_columns] = id_labels
    
    # グラフ　ノードラベル　隣接行列
    graph, id_labels, adj_matrix, adj_matrix_binary = compute_graph_from_similarity(df_vec, id_columns, edge_threshold)
    # 固有ベクトルの中心性
    eigenvector_centrality = compute_eigenvector_centrality(adj_matrix_binary)
    # ハブノードの抽出（類似度計算）
    id_labels_sort_top, sort_top_index = extract_hub_nodes(id_labels, adj_matrix, sim_thresh_top)
    # ハブノードの選出（多様性考慮）
    nodes_select, cents_select = select_hub_nodes(graph, eigenvector_centrality, id_labels_sort_top, sort_top_index, sim_thresh_div)
    # 要約結果の追加
    df_text['select'] = (df_text[id_columns].map(lambda x : x in nodes_select)).astype(int)
    return df_text

if __name__ == '__main__':
    # file path
    TEXT_FILE = '../data/rashomon.txt'
    # read text & extract nouns
    df_text = create_text_table(TEXT_FILE)
    # summarize_text
    edge_threshold=0.1
    sim_thresh_top=0.1
    sim_thresh_div=0.4
    df_text = summarize_text(df_text, edge_threshold, sim_thresh_top, sim_thresh_div)