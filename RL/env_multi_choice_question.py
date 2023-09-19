
import json
import numpy as np
import os
import random
from utils import *
from torch import nn
import ipdb
from tkinter import _flatten
from collections import Counter
import scipy.sparse as sp
import pandas as pd

def HT_from_H_sp(H, cand_act, variable_weight=False, out_node=None):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence sparse matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    DV = torch.sum(H, dim=1, keepdim=True) + 1e-5
    DE = torch.sum(H, dim=0, keepdim=True)+ 1e-5
    invDE = torch.diag(DE.pow(-1).reshape(-1))[:out_node]
    #invDV = torch.diag(DV[cand_act].pow(-1).reshape(-1))
    HT = H.transpose(0,1)

    HT = invDE.matmul(HT)
    #G = invDV.matmul(H[cand_act,:])
    return HT#, G

class MultiChoiceRecommendEnv(object):
    def __init__(self, kg, dataset, data_name, embed, seed=1, max_turn=15, cand_num=10, cand_item_num=10, attr_num=20, mode='train', ask_num=1, entropy_way='weight entropy', fm_epoch=0,choice_num=4,fea_score="weight entropy"):
        self.data_name = data_name
        self.mode = mode
        self.seed = seed
        self.max_turn = max_turn    #MAX_TURN
        self.attr_state_num = attr_num
        self.kg = kg
        self.device=torch.device('cuda')
        self.dataset = dataset
        self.feature_length = getattr(self.dataset, 'feature').value_len
        self.user_length = getattr(self.dataset, 'user').value_len
        self.item_length = getattr(self.dataset, 'item').value_len
        self.large_feature_length=42
        self.other_feature=self.large_feature_length+1
        self.small_feature_to_large={} # key: small ; value: large
        self.get_feature_dict()
        self.choice_num=choice_num
        self.fea_score=fea_score #[entopy, embedding]
        # action parameters
        self.ask_num = ask_num
        self.rec_num = 10
        self.random_sample_feature = False
        self.random_sample_item = False
        if cand_num == 0:
            self.cand_num = 10
            self.random_sample_feature = True
        else:
            self.cand_num = cand_num
        if cand_item_num == 0:
            self.cand_item_num = 10
            self.random_sample_item = True
        else:
            self.cand_item_num = cand_item_num
        #  entropy  or weight entropy
        self.ent_way = entropy_way

        # user's profile
        self.reachable_feature = []   # user reachable feature
        self.user_acc_feature = []  # user accepted feature which asked by agent
        self.user_rej_feature = []  # user rejected feature which asked by agent
        self.cand_items = []   # candidate items
        self.rej_item=[]
        self.item_feature_pair = {}
        self.cand_item_score = []
        self.friend_to_item=None
        #user_id  item_id   cur_step   cur_node_set
        self.user_id = None
        self.target_item = None
        self.cur_conver_step = 0        #  the number of conversation in current step
        self.cur_node_set = []     # maybe a node or a node set  /   normally save feature node
        # state veactor
        self.user_embed = None
        self.conver_his = []    #conversation_history
        self.attr_ent = []  # attribute entropy

        self.ui_dict,self.u_multi = self.__load_rl_data__(data_name, mode=mode)  # np.array [ u i weight]
        self.user_weight_dict = dict()
        self.user_items_dict = dict()

        #init seed & init user_dict
        set_random_seed(self.seed) # set random seed
        if mode == 'train':
            self.__user_dict_init__() # init self.user_weight_dict  and  self.user_items_dict
        elif mode == 'test':
            self.ui_array = None    # u-i array [ [userID1, itemID1], ...,[userID2, itemID2]]
            self.__test_tuple_generate__()
            self.test_num = 0
        # embeds = {
        #     'ui_emb': ui_emb,
        #     'feature_emb': feature_emb
        # }
        # load fm epoch
        # self.get_sameatt_items()
        embeds = load_embed(data_name, embed, epoch=fm_epoch)
        # ipdb.set_trace()
        if len(embeds)!=0:
            self.ui_embeds=embeds['ui_emb']
            self.feature_emb = embeds['feature_emb']
        else:
            self.ui_embeds = nn.Embedding(self.user_length+self.item_length, 64).weight.data.numpy()
            self.feature_emb = nn.Embedding(self.feature_length, 64).weight.data.numpy()
        # self.feature_length = self.feature_emb.shape[0]-1

        self.action_space = 2

        self.reward_dict = {
            'ask_suc': 0.01,
            'ask_fail': -0.1,
            'rec_suc': 1,
            'rec_fail': -0.1,
            'until_T': -0.3,      # MAX_Turn
            'cand_none': -0.1
        }
        self.history_dict = {
            'ask_suc': 1,
            'ask_fail': -1,
            'rec_scu': 2,
            'rec_fail': -2,
            'until_T': 0
        }
        self.attr_count_dict = dict()   # This dict is used to calculate entropy

    def __load_rl_data__(self, data_name, mode):
        with open(os.path.join(DATA_DIR[data_name], 'Graph_generate_data/second-layer_oringinal_tag_map.json'),
                  encoding='utf-8') as f:
            self.feature_map = json.load(f)
        with open(os.path.join(DATA_DIR[data_name], 'Graph_generate_data/first-layer_merged_tag_map.json'),
                  encoding='utf-8') as f:
            self.large_feature_map = json.load(f)
        self.feature_name=list(self.feature_map.keys())
        self.large_feature_name = list(self.large_feature_map.keys())
        if mode == 'train':
            with open(os.path.join(DATA_DIR[data_name], 'UI_Interaction_data/review_dict_valid.json'),
                      encoding='utf-8') as f:
                print('train_data: load RL valid data')
                mydict = json.load(f)
            with open(os.path.join(DATA_DIR[data_name], 'UI_data/train.pkl'), 'rb') as f:
                u_multi = pickle.load(f)
        elif mode == 'test':
            with open(os.path.join(DATA_DIR[data_name], 'UI_Interaction_data/review_dict_test.json'),
                      encoding='utf-8') as f:
                print('test_data: load RL test data')
                mydict = json.load(f)
            with open(os.path.join(DATA_DIR[data_name], 'UI_data/test.pkl'), 'rb') as f:
                u_multi = pickle.load(f)
        return mydict, u_multi


    def __user_dict_init__(self):   #Calculate the weight of the number of interactions per user
        ui_nums = 0
        for items in self.ui_dict.values():
            ui_nums += len(items)
        for user_str in self.ui_dict.keys():
            user_id = int(user_str)
            self.user_weight_dict[user_id] = len(self.ui_dict[user_str])/ui_nums
        print('user_dict init successfully!')

    def get_feature_dict(self):
        num=0
        # for n in self.kg.G['item']:
        #     enablePrint()
        #     ipdb.set_trace()
        for m in self.kg.G['feature']:
            if len(self.kg.G['feature'][m]['link_to_feature']):
                large=self.kg.G['feature'][m]['link_to_feature'][0]
                self.small_feature_to_large[m]=large
            else:
                self.small_feature_to_large[m]=self.other_feature
                num+=1
            

    def __test_tuple_generate__(self):
        ui_list = []
        for user_str in self.u_multi:
            user_id = int(user_str)
            for pair in self.u_multi[user_str]:
                ui_list.append([user_id, pair])
        self.ui_array = np.array(ui_list)
        np.random.shuffle(self.ui_array)
        """
        for user_str, items in self.u_multi.items():
            user_id = int(user_str)
            for item_id in items:
                ui_list.append([user_id, item_id])
        """
    
    def get_sameatt_items(self):
        users = list(self.ui_dict.keys())
        self.ui_satt_items={}       #each user, each item, reachable items

        for user in users:
            user=int(user)
            all_items = self.ui_dict[str(user)]
            same_att_items={}       # each item => reachable items
            a2i,i2a = {},{}         # map attribute to item , item to attribute
            for item in all_items:
                att=set(self.kg.G['item'][item]['belong_to'])
                i2a[item]=att
                for a in att:
                    if a in a2i:
                        a2i[a].append(item)
                    else:
                        a2i[a]=[item]

            for item in all_items:
                can_att=i2a[item]
                can_items=[]
                for a in can_att:
                    tmp_items=a2i[a]
                    can_items+=tmp_items
                same_att_items[item]=can_items

            self.ui_satt_items[user]=same_att_items
        
    def reset(self, embed=None):
        if embed is not None:
            self.ui_embeds = embed[:self.user_length+self.item_length]
            self.feature_emb = embed[self.user_length+self.item_length:]
        #init  user_id  item_id  cur_step   cur_node_set
        self.cur_conver_step = 0   #reset cur_conversation step
        self.cur_node_set = []
        self.rej_item=[]
        self.neg_item=[]
        if self.mode == 'train':
            # users = list(self.user_weight_dict.keys())
            # # self.user_id = np.random.choice(users, p=list(self.user_weight_dict.values())) # select user  according to user weights
            # self.user_id = np.random.choice(users)
            # self.target_item = np.random.choice(self.ui_dict[str(self.user_id)])
            users = list(self.user_weight_dict.keys())
            self.user_id = np.random.choice(users)
            self.target_item = random.choice(self.u_multi[str(self.user_id)])



        elif self.mode == 'test':
            # self.user_id = self.ui_array[self.test_num, 0]
            # self.target_item = self.ui_array[self.test_num, 1]
            # self.test_num += 1
            self.user_id = self.ui_array[self.test_num, 0]
            self.target_item = self.ui_array[self.test_num, 1]
            self.test_num += 1
        self.friends_ids=list(self.kg.G['user'][self.user_id]['friends'])
        self.friend_items=[]
        for friend in self.friends_ids:
            self.friend_items += self.kg.G['user'][friend]['interact']
        self.friend_items=list(set(self.friend_items))
        # init user's profile
        # enablePrint()
        print('-----------reset state vector------------')
        print('user_id:{}, target_item:{}'.format(self.user_id, self.target_item))
        feature_groundtrue=[]
        feature_groundtruehard=set()
        # ipdb.set_trace()
        for i in self.target_item:
            if len(feature_groundtruehard)==0:
                feature_groundtruehard=set(self.kg.G['item'][i]['belong_to'])
            else:
                feature_groundtruehard=feature_groundtruehard&set(self.kg.G['item'][i]['belong_to'])
            feature_groundtrue += self.kg.G['item'][i]['belong_to']
        self.feature_groundtrue=list(set(feature_groundtrue))
        self.feature_groundtruehard=list(feature_groundtruehard)
        self.reachable_feature = []  # user reachable feature in cur_step
        self.user_acc_feature = []  # user accepted feature which asked by agent
        self.user_rej_feature = []  # user rejected feature which asked by agent
        self.cand_items = list(range(self.item_length))

        # init state vector
        self.user_embed = self.ui_embeds[self.user_id].tolist()  # init user_embed   np.array---list
        self.conver_his = [0] * self.max_turn  # conversation_history
        self.attr_ent = [0] * self.attr_state_num  # attribute entropy

        # initialize dialog by randomly asked a question from ui interaction
        attrs=set(self.kg.G['item'][self.target_item[0]]['belong_to'])
        for i in range(1,len(self.target_item)):
            attrs2=set(self.kg.G['item'][self.target_item[i]]['belong_to'])
            attrs=attrs&attrs2
        attrs=list(attrs)
        user_like_random_fea = random.choice(attrs)

        self.user_acc_feature.append(user_like_random_fea) #update user acc_fea
        self.cur_node_set.append(user_like_random_fea)
        self._update_cand_items([user_like_random_fea], acc_rej=True)
        self.init_items=self.cand_items
        self._updata_reachable_feature()  # self.reachable_feature = []
        self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']
        self.cur_conver_step += 1

        print('=== init user prefer feature: {}'.format(self.feature_name[user_like_random_fea]))
        self._update_feature_entropy()  #update entropy
        print('reset_reachable_feature num: {}'.format(len(self.reachable_feature)))

        # Sort reachable features according to the entropy of features
        reach_fea_score = self._feature_score()
        max_ind_list = []
        for k in range(self.cand_num):
            max_score = max(reach_fea_score)
            max_ind = reach_fea_score.index(max_score)
            reach_fea_score[max_ind] = 0
            if max_ind in max_ind_list:
                break
            max_ind_list.append(max_ind)
        
        max_fea_id = [self.reachable_feature[i] for i in max_ind_list]
        [self.reachable_feature.remove(v) for v in max_fea_id]
        [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]
        cand_act=self._get_cand()
        return self._get_state(cand_act), cand_act, self._get_action_space()

    def _get_cand(self):
        if self.random_sample_feature:
            cand_feature = self._map_to_all_id(random.sample(self.reachable_feature, min(len(self.reachable_feature),self.cand_num)),'feature')
        else:
            cand_feature = self._map_to_all_id(self.reachable_feature[:self.cand_num],'feature')
        if self.random_sample_item:
            cand_item =  self._map_to_all_id(random.sample(self.cand_items, min(len(self.cand_items),self.cand_item_num)),'item')
        else:
            cand_item = self._map_to_all_id(self.cand_items[:self.cand_item_num],'item')
        cand = cand_feature + cand_item
        return cand
    
    def _get_action_space(self):
        action_space = [self._map_to_all_id(self.reachable_feature,'feature'), self._map_to_all_id(self.cand_items,'item')]
        return action_space

    def _get_state(self, cand_act):
        if self.data_name in ['YELP_STAR']:
            self_cand_items = self.cand_items[:5000]
            #set_cand_items = set(self_cand_items)
            self_neg_items = set(self.init_items) - set(self.cand_items)#[item_id - self.user_length for item_id in self.rej_item]
            self_neg_items = list(self_neg_items)[:1000]
        else:
            self_cand_items = self.cand_items
            self_neg_items = set(self.init_items) - set(self.cand_items)
        friend_item = {}
        set_cand_items=set(self.cand_items)
        for friend in self.friends_ids:
            item_list = [item for item in self.kg.G['user'][friend]['interact'] if item in set_cand_items]
            if len(item_list)>0:
                friend_item[friend] = item_list
        self.friends_ids=list(friend_item.keys())
        self.friend_to_item=friend_item
        friend_attribute = {}
        set_reachable_feature=set(self.reachable_feature)
        for friend in self.friends_ids:
            attribute_list = []
            for item in friend_item[friend]:
                attribute_list += [attribute for attribute in self.kg.G['item'][item]['belong_to'] if
                                   attribute in set_reachable_feature]
            friend_attribute[friend] = dict(pd.value_counts(attribute_list))
        friend_ids = self.friends_ids
        user = [self.user_id]
        cand_items = [x + self.user_length for x in self_cand_items]
        neg_items = [x + self.user_length for x in self_neg_items]
        reachable_feature = [x + self.user_length + self.item_length for x in self.reachable_feature]
        pos_feature = [x + self.user_length + self.item_length for x in self.user_acc_feature]
        neg_feature = [x + self.user_length + self.item_length for x in self.user_rej_feature]

        feature_item_pair = {}
        for item in self_cand_items:
            for feature in self.item_feature_pair[item]:
                if feature not in feature_item_pair:
                    feature_item_pair[feature]=set([item])
                else:
                    feature_item_pair[feature].add(item)

        feature_neg_item_pair ={}
        #feature_neg=self.user_rej_feature+self.reachable_feature
        for item in self_neg_items:
            for feature in list(self.kg.G['item'][item]['belong_to']):
                if feature in self.user_rej_feature:
                    if feature not in feature_neg_item_pair:
                        feature_neg_item_pair[feature]= set([item])
                    else:
                        feature_neg_item_pair[feature].add(item)
                elif feature in self.reachable_feature+self.user_acc_feature:
                    if feature not in feature_item_pair:
                        feature_item_pair[feature]= set([item])
                    else:
                        feature_item_pair[feature].add(item)
        hyperneighbors = user + pos_feature + neg_feature+friend_ids + reachable_feature + cand_items + neg_items
        hyperedges = pos_feature + neg_feature +friend_ids + reachable_feature
        idx2 = dict(enumerate(hyperneighbors))
        idx2 = {v: k for k, v in idx2.items()}
        idx3_dic = dict(enumerate(hyperedges))
        idx3 = {v: k for k, v in idx3_dic.items()}
        #hyperindex=[idx3[x] for x in hyperedges]
        i2 = []
        v2 = []


        ssl_set_pos = []
        ssl_set_neg = []

        for friend in self.friends_ids:
            i2.append([idx2[self.user_id], idx3[friend]])
            v2.append(1)
            i2.append([idx2[friend], idx3[friend]])
            v2.append(1)

            for item in friend_item[friend]:
                item_id=item + self.user_length
                if item_id in idx2:
                    i2.append([idx2[item_id], idx3[friend]])
                    v2.append(1/len(friend_item[friend]))

            for attribute in friend_attribute[friend]:
                attribute_id=attribute + self.user_length+self.item_length
                if attribute_id in idx3:
                    i2.append([idx2[attribute_id], idx3[friend]])
                    #v2.append(1)
                    v2.append(1/len(friend_attribute[friend]))#friend_attribute[friend][attribute])

        for fea in self.user_acc_feature:
            fea_id = fea + self.user_length + self.item_length
            #ssl_set_pos.append([idx2[self.user_id], idx2[fea_id]])
            i2.append([idx2[self.user_id], idx3[fea_id]])
            i2.append([idx2[fea_id], idx3[fea_id]])
            v2.append(1)
            v2.append(1)
        for fea in self.user_rej_feature:
            fea_id = fea + self.user_length + self.item_length
            #ssl_set_neg.append([idx2[self.user_id], idx2[fea_id]])
            i2.append([idx2[self.user_id], idx3[fea_id]])
            i2.append([idx2[fea_id], idx3[fea_id]])
            v2.append(-1)
            v2.append(1)
        for fea in feature_neg_item_pair:
            fea_id = fea + self.user_length + self.item_length
            i2.append([idx2[fea_id], idx3[fea_id]])
            v2.append(1)
            for item in feature_neg_item_pair[fea]:
                item_id = item + self.user_length
                i2.append([idx2[item_id], idx3[fea_id]])
                v2.append(1 / len(feature_neg_item_pair[fea]))
        for fea in feature_item_pair:
            fea_id = fea + self.user_length + self.item_length
            i2.append([idx2[fea_id], idx3[fea_id]])
            v2.append(1)
            for item in feature_item_pair[fea]:
                item_id = item + self.user_length
                i2.append([idx2[item_id], idx3[fea_id]])
                v2.append(1 / len(feature_item_pair[fea]))
        #cand_item_score = self.sigmoid(self.cand_item_score)
        ssl_set = {'pos': ssl_set_pos, 'neg': ssl_set_neg}
        hyperneighbors = torch.LongTensor(hyperneighbors)
        #hyperindex=torch.LongTensor(hyperindex)
        i2 = torch.LongTensor(i2)
        v2 = torch.FloatTensor(v2)
        H = torch.sparse.FloatTensor(i2.t(), v2, torch.Size([len(hyperneighbors), len(idx3_dic)]))

        HT = HT_from_H_sp(H.to_dense(), None, out_node=len(self.user_acc_feature)+len(self.user_rej_feature)+len(friend_ids))
        state = {'acc_feature': self.user_acc_feature,
                 'rej_feature': self.user_rej_feature,
                 'rej_item': self.rej_item,
                 'user': self.user_id,
                 'friend':friend_ids,
                 'hyperneigh': hyperneighbors,
                 #'hyperindex': hyperindex,
                 'hyperHT': HT}
        return state
    def step(self, action, sorted_actions, embed=None):  
        if embed is not None:
            self.ui_embeds = embed[:self.user_length+self.item_length]
            self.feature_emb = embed[self.user_length+self.item_length:]

        done = 0
        print('---------------step:{}-------------'.format(self.cur_conver_step))

        if self.cur_conver_step == self.max_turn:
            reward = self.reward_dict['until_T']
            self.conver_his[self.cur_conver_step-1] = self.history_dict['until_T']
            print('--> Maximum number of turns reached !')
            done = 1
        elif action >= self.user_length + self.item_length and len(self.cand_items)>20:   #ask feature
            score={}
            max_score_id=None
            max_score=-999
            large_small={}
            for i in range(len(sorted_actions)):
                act=sorted_actions[i]
                if act < self.user_length + self.item_length:
                    continue
                small_fea=self._map_to_old_id(act)
                large=self.small_feature_to_large[small_fea]
                if large in score:
                    score[large]+=1/(i+1)
                    large_small[large].append(small_fea)
                else:
                    score[large]=0.0
                    large_small[large]=[]
                    score[large]+=1/(i+1)
                    large_small[large].append(small_fea)
                if score[large]>max_score:
                    max_score=score[large]
                    max_score_id=large
            """
            if self.mode== 'test':
                feature_sorted=self.reachable_feature[self.cand_num:]
                for fea in feature_sorted:
                    if len(large_small[max_score_id])>= self.choice_num:
                        break
                    if self.small_feature_to_large[fea] ==max_score_id:
                        large_small[max_score_id].append(fea)
            """
            asked_feature=large_small[max_score_id][:self.choice_num]
            large_name=self.large_feature_name[max_score_id] if max_score_id!=43 else 'Other'
            print('-->action: feature type {}, ask features {}, max entropy feature {}'.format(large_name,[self.feature_name[feature_id] for feature_id in asked_feature], self.reachable_feature[:self.cand_num]))
            """
            if self.friend_to_item:
                for friend in self.friend_to_item:
                    for item in self.target_item:
                        if item in self.friend_to_item[friend]:
                            print('friend name {}, same target item {}, item num {}, friend items {}'.format(friend, item, len(self.friend_to_item[friend]), self.friend_to_item[friend]))
                            break
            """
            reward, done, acc_rej = self._ask_update(asked_feature)  #update user's profile:  user_acc_feature & user_rej_feature
            self._update_cand_items(asked_feature, acc_rej)   #update cand_items
        else:  #recommend items
            
            #===================== rec update=========
            recom_items = []
            recom_items_gnn_id=[]
            for act in sorted_actions:
                if act < self.user_length + self.item_length:
                    recom_items_gnn_id.append(act)
                    recom_items.append(self._map_to_old_id(act))
                    if len(recom_items) == self.rec_num:
                        break
            if self.friend_to_item:
                for friend in self.friend_to_item:
                    for item in self.target_item:
                        if item in self.friend_to_item[friend]:
                            print('friend name {}, same target item {}, item num {}, friend items {}'.format(friend, item, len(self.friend_to_item[friend]), self.friend_to_item[friend]))
                            break
            reward, done = self._recommend_update(recom_items)
            #========================================
            if reward > 0:
                print('-->Recommend successfully!')
            else:
                self.rej_item+=recom_items_gnn_id
                print('-->Recommend fail !')
        
        self._updata_reachable_feature()  # update user's profile: reachable_feature

        print('reachable_feature num: {}'.format(len(self.reachable_feature)))
        print('cand_item num: {}'.format(len(self.cand_items)))

        self._update_feature_entropy()
        if len(self.reachable_feature) != 0:  # if reachable_feature == 0 :cand_item= 1
            reach_fea_score = self._feature_score()  # compute feature score

            max_ind_list = []
            for k in range(self.cand_num):
                max_score = max(reach_fea_score)
                max_ind = reach_fea_score.index(max_score)
                reach_fea_score[max_ind] = 0
                if max_ind in max_ind_list:
                    break
                max_ind_list.append(max_ind)
            max_fea_id = [self.reachable_feature[i] for i in max_ind_list]
            [self.reachable_feature.remove(v) for v in max_fea_id]
            [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]
        self.cur_conver_step += 1
        cand_act=self._get_cand()
        return self._get_state(cand_act), cand_act, self._get_action_space(), reward, done


    def _updata_reachable_feature(self):
        next_reachable_feature = []
        reachable_item_feature_pair = {}
        for cand in self.cand_items:
            fea_belong_items = list(self.kg.G['item'][cand]['belong_to']) # A-I
            next_reachable_feature.extend(fea_belong_items)
            reachable_item_feature_pair[cand] = list(set(fea_belong_items) - set(self.user_rej_feature))
            next_reachable_feature = list(set(next_reachable_feature))
        self.reachable_feature = list(set(next_reachable_feature) - set(self.user_acc_feature) - set(self.user_rej_feature))
        self.item_feature_pair = reachable_item_feature_pair

    def _feature_score(self):
        
        reach_fea_score = []
        for feature_id in self.reachable_feature:
            score = self.attr_ent[feature_id]
            reach_fea_score.append(score)
        return reach_fea_score


    def _item_score(self):
        cand_item_score = []
        if len(self.friend_items)>0:
            fread_embed= self.ui_embeds[[self.user_length + item_id for item_id in self.friend_items]].mean(axis=0)
        for item_id in self.cand_items:
            item_embed = self.ui_embeds[self.user_length + item_id]
            score = 0
            score += np.inner(np.array(self.user_embed), item_embed)
            prefer_embed = self.feature_emb[self.user_acc_feature, :]  #np.array (x*64)
            unprefer_feature = list(set(self.user_rej_feature) & set(self.kg.G['item'][item_id]['belong_to']))
            unprefer_embed = self.feature_emb[unprefer_feature, :]  #np.array (x*64)
            for i in range(len(self.user_acc_feature)):
                score += np.inner(prefer_embed[i], item_embed)
            for i in range(len(unprefer_feature)):
                #score -= self.sigmoid([np.inner(unprefer_embed[i], item_embed)])[0]
                score -= np.inner(unprefer_embed[i], item_embed)
            if len(self.friend_items)>0:
                score += np.inner(fread_embed, item_embed)
            cand_item_score.append(self.sigmoid(score))
        return cand_item_score


    def _ask_update(self, asked_features):
        '''
        :return: reward, acc_feature, rej_feature
        '''
        done = 0
        # TODO datafram!     groundTruth == target_item features
        
        reward=0
        acc_rej = False
        self.conver_his[self.cur_conver_step] = self.history_dict['ask_fail']
        for asked_feature in asked_features:
            if asked_feature in self.feature_groundtrue:
                acc_rej = True
                self.user_acc_feature.append(asked_feature)
                self.cur_node_set.append(asked_feature)
                reward += self.reward_dict['ask_suc']
                self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']   #update conver_his
            else:
                self.user_rej_feature.append(asked_feature)
                reward += self.reward_dict['ask_fail']
                #update conver_his

        if self.cand_items == []:  #candidate items is empty
            done = 1
            reward = self.reward_dict['cand_none']

        return reward, done, acc_rej

    def _update_cand_items(self, asked_feature, acc_rej):
        acc_item=[]
        rej_item=[]
        frd_acc_item=[]
        frd_rej_item=[]
        self.target_item = list(set(self.target_item) & set(self.cand_items))
        feature_groundtrue = []
        # ipdb.set_trace()
        for i in self.target_item:
            feature_groundtrue += self.kg.G['item'][i]['belong_to']
        self.feature_groundtrue = list(set(feature_groundtrue))

        for fea in asked_feature:
            if fea in self.feature_groundtrue:    # accept feature
                print('=== ask acc {}: update cand_items'.format(self.feature_name[fea]))
                feature_items = self.kg.G['feature'][fea]['belong_to']
                cand_items = set(self.cand_items) & set(feature_items)   #  itersection
                friend_items=set(self.friend_items) & set(feature_items)
                acc_item+= list(cand_items)
                frd_acc_item += list(friend_items)
                #self.feature_groundtruehard.remove(fea)
            
            elif fea not in self.feature_groundtrue:    # reject feature
                feature_items = self.kg.G['feature'][fea]['belong_to']
                cand_items = set(self.cand_items) & set(feature_items)
                friend_items=set(self.friend_items) & set(feature_items)
                rej_item+= list(cand_items)
                frd_rej_item += list(friend_items)
                print('=== ask rej {}: update cand_items'.format(self.feature_name[fea]))
                
        
        if len(acc_item)==0:
            cand_items=list(set(self.cand_items)-(set(self.cand_items) & set(rej_item)))
        else:
            cand_items=list(set(acc_item)-(set(acc_item) & set(rej_item)))
        if len(frd_acc_item) == 0:
            friend_items = list(set(self.friend_items) - (set(self.friend_items) & set(frd_rej_item)))
        else:
            friend_items = list(set(frd_acc_item) - (set(frd_acc_item) & set(frd_rej_item)))
        #self.neg_item+=set(self.cand_items)-set(cand_items)
        if len(cand_items)!=0:
            self.cand_items=cand_items
        self.friend_items=friend_items
        #select topk candidate items to recommend
        cand_item_score = self._item_score()
        item_score_tuple = list(zip(self.cand_items, cand_item_score))
        sort_tuple = sorted(item_score_tuple, key=lambda x: x[1], reverse=True)
        self.cand_items, self.cand_item_score = zip(*sort_tuple)
    
    def _recommend_update(self, recom_items):
        print('-->action: recommend items')
        print(set(recom_items) - set(self.cand_items[: self.rec_num]))
        self.cand_items = list(self.cand_items)
        self.cand_item_score = list(self.cand_item_score)
        #recom_items = self.cand_items[: self.rec_num]    # TOP k item to recommend
        # if self.target_item in recom_items:
        hit=False
        for i in self.target_item:
            if i in recom_items:
                hit=True
                break
        if hit:
            reward = self.reward_dict['rec_suc'] 
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_scu'] #update state vector: conver_his
            tmp_score = []
            for item in recom_items:
                idx = self.cand_items.index(item)
                tmp_score.append(self.cand_item_score[idx])
            self.cand_items = recom_items
            self.cand_item_score = tmp_score
            done = recom_items.index(i) + 1
        else:
            reward = self.reward_dict['rec_fail']
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_fail']  #update state vector: conver_his
            if len(self.cand_items) > self.rec_num:
                for item in recom_items:
                    del self.item_feature_pair[item]
                    idx = self.cand_items.index(item)
                    self.cand_items.pop(idx)
                    self.cand_item_score.pop(idx)
                #self.cand_items = self.cand_items[self.rec_num:]  #update candidate items
            done = 0
        return reward, done

    def _update_feature_entropy(self):
        if self.ent_way == 'entropy':
            cand_items_fea_list = []
            for item_id in self.cand_items:
                cand_items_fea_list.append(list(self.kg.G['item'][item_id]['belong_to']))
            cand_items_fea_list = list(_flatten(cand_items_fea_list))
            self.attr_count_dict = dict(Counter(cand_items_fea_list))
            self.attr_ent = [0] * self.attr_state_num  # reset attr_ent
            real_ask_able = list(set(self.reachable_feature) & set(self.attr_count_dict.keys()))
            for fea_id in real_ask_able:
                p1 = float(self.attr_count_dict[fea_id]) / len(self.cand_items)
                p2 = 1.0 - p1
                if p1 == 1:
                    self.attr_ent[fea_id] = 0
                else:
                    ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                    self.attr_ent[fea_id] = ent
        elif self.ent_way == 'weight_entropy':
            cand_items_fea_list = []
            self.attr_count_dict = {}
            friend_features=[]
            if len(self.friend_items) > 0:
                for item in self.friend_items:
                    friend_features+=self.kg.G['item'][item]['belong_to']
                friend_embed = self.feature_emb[[feature_id for feature_id in friend_features]].mean(axis=0)
            #cand_item_score = self._item_score()
            cand_item_score_sig = self.cand_item_score  # sigmoid(score)
            for score_ind, item_id in enumerate(self.cand_items):
                cand_items_fea_list = list(self.kg.G['item'][item_id]['belong_to'])
                for fea_id in cand_items_fea_list:
                    if self.attr_count_dict.get(fea_id) == None:
                        self.attr_count_dict[fea_id] = []
                    self.attr_count_dict[fea_id] += [cand_item_score_sig[score_ind]]

            self.attr_ent = [0] * self.attr_state_num  # reset attr_ent
            real_ask_able = list(set(self.reachable_feature) & set(self.attr_count_dict.keys()))
            sum_score_sig = sum(cand_item_score_sig)

            for fea_id in real_ask_able:
                feature_embe= self.feature_emb[fea_id]
                self.attr_ent[fea_id] = float(np.sum(self.attr_count_dict[fea_id])) / sum_score_sig
                if len(self.friend_items) > 0:
                    self.attr_ent[fea_id]+=self.sigmoid(np.inner(friend_embed, feature_embe))*len(self.attr_count_dict[fea_id])/sum_score_sig

    def sigmoid(self, x_list):
        x_np = np.array(x_list)
        s = 1 / (1 + np.exp(-x_np))
        return s.tolist()

    def _map_to_all_id(self, x_list, old_type):
        if old_type == 'item':
            return [x + self.user_length for x in x_list]
        elif old_type == 'feature':
            return [x + self.user_length + self.item_length for x in x_list]
        else:
            return x_list

    def _map_to_old_id(self, x):
        if x >= self.user_length + self.item_length:
            x -= (self.user_length + self.item_length)
        elif x >= self.user_length:
            x -= self.user_length
        return x

