import matplotlib.pyplot as plt
import math
import random
import bisect

param_Q = 100
ant_prob_random = 0.1

# サイクル数
total_cycle = 500
total_ants = 1000

# 蒸発率
rou = 0.9
alpha = 5
beta = 3


class Ant:
    def __init__(self):
        self.m_visited_vertex = [False for i in range(47)]
        self.m_visited_path = []

    def calc_next_pheromone(self, dist):
        next_pheromone = [[0 for j in range(47)] for i in range(47)]
        length = self.calc_path_length(dist)
        for i in range(46):
            start = self.m_visited_path[i]
            end = self.m_visited_path[i + 1]
            next_pheromone[start][end] += param_Q / length
            next_pheromone[end][start] += param_Q / length
        return next_pheromone

    def calc_path_length(self, dist):
        length = 0.0
        for i in range(46):
            length += dist[self.m_visited_path[i]][self.m_visited_path[i + 1]]
        length += dist[self.m_visited_path[0]][self.m_visited_path[-1]]
        return length

    def calc_prob_from_v(self, v, dist, latest_pheromone):
        # 確率の合計
        sumV = 0

        # 行き先候補情報
        to_vertexes = []
        to_pheromones = []

        for to in range(47):

            # すでに訪ねていたら
            if (to == v) or self.m_visited_vertex[to]:
                continue

            # フェロモン分子の計算
            pheromone = latest_pheromone[v][to] ** alpha * \
                (param_Q / dist[v][to]) ** beta
            sumV += pheromone

            # 候補に追加
            to_vertexes.append(to)
            to_pheromones.append(pheromone)

        to_prob = [x / sumV for x in to_pheromones]

        # 確率の累積和を取る
        for i in range(len(to_prob) - 1):
            to_prob[i + 1] += to_prob[i]

        return to_vertexes, to_prob


def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


town = []  # ０列目の数字を格納する予定のリスト
long = []  # １列目の数字を格納する予定のリスト
lati = []  # ２列目の数字を格納する予定のリスト
for i, line in enumerate(open('location.csv')):  # ファイルを開いて一行一行読み込む
    if i == 0:  # ０番目の行の場合
        continue  # 次の行に行く
    c = line.split(",")  # 行をコンマで分割したものをcというリストに入れる
    town.append(c[0])  # ０列目の単語townに入れる
    long.append(float(c[1]))  # １列目の単語を実数に変換してlongに入れる
    lati.append(float(c[2]))  # ２列目の単語を実数に変換してlatiに入れる

# すべての都市間で都市間の距離を計算する。
dist = [[0 for j in range(47)] for i in range(47)]
for i, city1 in enumerate(zip(town, lati, long)):
    for j, city2 in enumerate(zip(town, lati, long)):
        dist[i][j] = distance(city1[1], city1[2], city2[1], city2[2])

edges = []
for i, city1 in enumerate(zip(town, lati, long)):
    for j, city2 in enumerate(zip(town, lati, long)):
        edges.append((city1[0], city2[0]))

# 前回のフェロモン，Q/エッジの長さで初期化
latest_pheromone = [[0 for j in range(47)] for i in range(47)]
for i in range(47):
    for j in range(47):
        if i == j:
            continue
        h = param_Q / dist[i][j]
        latest_pheromone[i][j] = h

path_history = []
length_history = []

for i in range(total_cycle):
    print(i)

    min_length = 9999
    min_path = []
    now_pheromone = [[0 for i in range(47)] for i in range(47)]
    ants = [Ant() for i in range(total_ants)]
    lengths = []
    for ant in ants:
        # スタート地点の設定
        ant.m_visited_path.append(0)
        ant.m_visited_vertex[0] = True
        # アリ１匹のパス構築
        for j in range(46):
            # 現在の頂点
            v = ant.m_visited_path[-1]

            # 頂点vから行ける先の(まだ行っていないすべての都道府県)の確率を求める
            to_vertexes, to_prob = ant.calc_prob_from_v(
                v, dist, latest_pheromone)
            to = -1
            # もし一様乱数より小さいなら，完全ランダム
            if random.random() < ant_prob_random:
                to = to_vertexes[random.randint(0, len(to_vertexes)-1)]
            else:
                random_p = random.uniform(0.0, 0.999999999)
                to = to_vertexes[bisect.bisect_left(to_prob, random_p)]

            ant.m_visited_path.append(to)
            ant.m_visited_vertex[to] = True

        # フェロモン配置
        # now_pheromoneに今回のターンのアリのフェロモンを集める．
        ant_pheromone = ant.calc_next_pheromone(dist)
        for i in range(47):
            for j in range(47):
                now_pheromone[i][j] += ant_pheromone[i][j]

        if min_length > ant.calc_path_length(dist):
            min_length = ant.calc_path_length(dist)
            min_path = ant.m_visited_path

    # latestを蒸発させて，latest+nowする
    for i in range(47):
        for j in range(47):
            if i < j:
                latest_pheromone[i][j] = latest_pheromone[i][j] * \
                    rou + now_pheromone[i][j]
                latest_pheromone[j][i] = latest_pheromone[i][j]

    print(min_length)
    print(min_path)
    path_history.append(min_path)
    length_history.append(min_length)


for i in range(47):
    print(town[min_path[i]])


def plot_path(min_path, name):
    last = plt.figure(figsize=(12, 8))
    plt.title("Prefectural capitals in Japan")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    # min_pathをもとに線を引く
    for i in range(46):
        plt.plot([lati[min_path[i]], lati[min_path[i+1]]],
                 [long[min_path[i]], long[min_path[i + 1]]], 'k-', alpha=0.2)
    plt.plot([lati[min_path[0]], lati[min_path[46]]],
             [long[min_path[0]], long[min_path[46]]], 'k-', alpha=0.2)
    plt.scatter(lati, long)
    for city, x, y in zip(town, lati, long):
        plt.text(x, y, city, alpha=0.3, size=12)
    last.savefig(name)


min_index = length_history.index(min(length_history))
history = plt.figure()
x = [i for i in range(total_cycle)]
plt.plot(x, length_history)
history.savefig('history.png')
plot_path(path_history[min_index], f'best_{min(length_history)}.png')
plot_path(min_path, f'last_{min_length}.png')
