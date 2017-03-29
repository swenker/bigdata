__author__ = 'wenjusun'


def load_dataset():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataset):
    C1 = []
    for transaction in dataset:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    return map(frozenset, C1)


def scanD(D, Ck, min_support):
    ss_cnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ss_cnt.has_key(can):
                    ss_cnt[can] = 1
                else:
                    ss_cnt[can] += 1
    num_items = float(len(D))
    ret_list = []
    support_data = {}
    for key in ss_cnt:
        support = ss_cnt[key] / num_items
        if support >= min_support:
            ret_list.insert(0, key)
        support_data[key] = support

    return ret_list, support_data


def apriori_gen(Lk, k):
    ret_list = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                ret_list.append(Lk[i] | Lk[j])

    return ret_list


def apriori(dataset, min_support=0.5):
    C1 = createC1(dataset)
    D = map(set, dataset)
    L1, supported_data = scanD(D, C1, min_support)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):
        Ck = apriori_gen(L[k - 2], k)
        Lk, supK = scanD(D, Ck, min_support)
        supported_data.update(supK)
        L.append(Lk)
        k += 1
    return L, supported_data


def test_it():
    dataset = load_dataset()
    # C1 = createC1(dataset)
    # D = map(set, dataset)
    # L1, supported_data = scanD(D, C1, 0.5)
    L1, supported_data = apriori(dataset)
    print dataset
    # print C1
    # print D
    # print L1
    print supported_data


if __name__ == "__main__":
    test_it()

