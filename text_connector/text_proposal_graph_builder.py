import numpy as np
from text_connector.text_connect_cfg import Config as TextLineCfg


class Graph:
    def __init__(self, graph):
        self.graph = graph

    def sub_graphs_connected(self):
        """
        按行遍历 graph, 在 graph 中标记的位置的横坐标都小于纵坐标
        因为横坐标表示的最大连接的起点, 而纵坐标表示的是最大连接的终点
        假设 graph[0, 3] 和 graph[3, 7] 都表示最大连接

        当 index = 0, not self.graph[:, index].any() and self.graph[index, :].any() = True
            v = 0 然后 v = 3 再然后 v = 7
        :return:
        """
        sub_graphs = []
        for index in range(self.graph.shape[0]):
            # not self.graph[:, index].any() 其实是为了过滤掉已经加入 sub_graphs 的点, 比如这里的 3
            # 当遍历到 index=3 时, self.graph[:, index].any() 返回 True, 不再重复考虑
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs


class TextProposalGraphBuilder:
    """
        Build Text proposals into a graph.
    """

    def __init__(self, MAX_HORIZONTAL_GAP=TextLineCfg.MAX_HORIZONTAL_GAP, MIN_V_OVERLAPS=TextLineCfg.MIN_V_OVERLAPS,
                 MIN_SIZE_SIM=TextLineCfg.MIN_SIZE_SIM):
        """
        @@param:MAX_HORIZONTAL_GAP:文本行间隔最大值
        @@param:MIN_V_OVERLAPS
        @@param:MIN_SIZE_SIM
        MIN_V_OVERLAPS=0.6
        MIN_SIZE_SIM=0.6
        """
        self.MAX_HORIZONTAL_GAP = MAX_HORIZONTAL_GAP
        self.MIN_V_OVERLAPS = MIN_V_OVERLAPS
        self.MIN_SIZE_SIM = MIN_SIZE_SIM

    def get_successions(self, index):
        """
        遍历当前 index 对应 box 右边 MAX_HORIZONTAL_GAP 内的所有 boxes
        如果某个 box 满足 meet_v_iou 的条件, 认为该 box 是当前 box 的 succession
        :param index:
        :return:
        """
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) + 1, min(int(box[0]) + self.MAX_HORIZONTAL_GAP + 1, self.im_size[1])):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def get_precursors(self, index):
        """
        遍历当前 index 对应 box 左边 MAX_HORIZONTAL_GAP 内的所有 boxes
        如果某个 box 满足 meet_v_iou 的条件, 认为该 box 是当前 box 的 precursor
        :param index:
        :return:
        """
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) - 1, max(int(box[0] - self.MAX_HORIZONTAL_GAP), 0) - 1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        # 查找 succession_index 对应 box 的 precursors
        precursors = self.get_precursors(succession_index)
        # 如果当前 box 的 score 大于 precursors 的最大 score
        # 那么认为 index 对应的 box 和 succession_index 对应的 box 是最大连接
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True
        # 否则不是最大连接
        return False

    def meet_v_iou(self, index1, index2):
        """
        判断 index1 和 index2 是否满足两个条件
            条件1: 两个 height 中共有部分 / 两个 height 的较小值 > self.MIN_V_OVERLAPS
            条件2: 两个 height 的较小值 / 两个 height 的较大值 > self.MIN_SIZE_SIM
        :param index1:
        :param index2:
        :return:
        """

        def overlaps_v(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            y0 = max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1 = min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1 - y0 + 1) / min(h1, h2)

        def size_similarity(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            return min(h1, h2) / max(h1, h2)

        return overlaps_v(index1, index2) >= self.MIN_V_OVERLAPS and \
               size_similarity(index1, index2) >= self.MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        # im_size (h, w) 原图像的大小
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        # 每一个 text_proposals 的 height
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1
        # 创建 w 个 list, 用于存放各种 xmin 的 text_proposals
        boxes_table = [[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table = boxes_table

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            # 找到当前 box 的 successions
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            # 从 successions 中找到 score 最大的那一个
            succession_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)
                # have equal scores.
                # 在 graph 的相应位置标记最大连接的 boxes
                graph[index, succession_index] = True
        return Graph(graph)
