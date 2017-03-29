__author__ = 'wenjusun'

class TreeNode:
    def __init__(self,name_value,num_occur,parent_node):
        self.name = name_value
        self.count = num_occur
        self.node_link = None
        self.parent = parent_node
        self.children = {}

    def inc(self,num_occur):
        self.count +=num_occur

    def disp(self,ind=1):
        print ' '*ind,self.name,' ',self.count
        for child in self.children.values():
            child.disp(ind+1)

def create_tree(dataset,min_sup=1):
    header_table={}
    for trans in dataset:
        for item in trans:
            header_table [item] = header_table.get(item,0)+dataset[trans]
        for k in header_table.keys():
            if header_table[k] < min_sup:
                del(header_table[k])
    freq_item_set = set(header_table.keys())
    if len(freq_item_set)==0:
        return None,None

    for k in header_table:
        header_table[k]=[header_table[k],None]

    ret_tree = TreeNode('Null Set',1,None)
    for tran_set,count in dataset.items():
        localD={}
        for item in tran_set:
            if item in freq_item_set:
                localD[item] = header_table[item][0]
        if len(localD)>0:
            ordered_items = [v[0] for v in sorted(localD.items(),key = lambda p:p[1],reverse=True)]
            update_tree(ordered_items,ret_tree,header_table,count)
    return ret_tree,header_table

def update_tree(items,in_tree,header_table,count):
    if items[0] in in_tree.children:
        in_tree.children[items[0]].inc(count)
    else:
        in_tree.children[items[0]] = TreeNode(items[0],count,in_tree)
        if header_table[items[0]][1] == None:
            header_table[items[0]][1] = in_tree.children[items[0]]
        else:
            update_header(header_table[items[0]][1],in_tree.children[items[0]])
    if len(items) >1:
        update_tree(items[1::],in_tree.children[items[0]],header_table,count)

def update_header(node_to_test,target_node):
    while node_to_test.nodeLink :
        node_to_test = node_to_test.nodeLink
    node_to_test.nodeLink = target_node

def load_dataset():
    simple_data=[
        ['r','z','h','j','p'],
        ['z','y','x','w','v','u','t','s'],
        ['z'],
        ['r','x','n','o','s'],
        ['y','r','x','z','q','t','p'],
        ['y','z','x','e','q','s','t','m']
    ]
    return simple_data

def create_init_set(dataset):
    ret_dict = {}
    for trans in dataset:
        ret_dict[frozenset(trans)] = 1
    return ret_dict

def ascend_tree(leaf_node,prefix_path):
    if leaf_node.parent:
        prefix_path.append(leaf_node.name)
        ascend_tree(leaf_node.parent,prefix_path)
def find_prefix_path(base_pat,tree_node):
    cond_pats={}
    while tree_node:
        prefix_path = {}
        ascend_tree(tree_node,prefix_path)
        if len(prefix_path) >1:
            cond_pats[frozenset(prefix_path[1:])]=tree_node.count
        tree_node = tree_node.node_link
    return cond_pats

def mine_tree(in_tree,header_table,minsup,prefix,freq_itemlist):
    bigL=[v[0] for v in sorted(header_table.items(),key=lambda p:p[1])]

    for base_pat in bigL:
        new_freq_set = prefix.copy()
        new_freq_set.add(base_pat)
        freq_itemlist.append(new_freq_set)
        cond_patt_bases = find_prefix_path(base_pat,header_table[base_pat][1])
        my_cond_tree,my_head = create_tree(cond_patt_bases,minsup)
        if my_head:
            mine_tree(my_cond_tree,my_head,minsup,new_freq_set,freq_itemlist)

