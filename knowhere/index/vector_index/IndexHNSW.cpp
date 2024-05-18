// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <algorithm>
#include <chrono>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <set>
#include <omp.h>

#include "common/Exception.h"
#include "common/Log.h"
#include "hnswlib/hnswlib/hnswalg.h"
#include "index/vector_index/IndexHNSW.h"
#include "index/vector_index/adapter/VectorAdapter.h"
#include "index/vector_index/helpers/FaissIO.h"

namespace knowhere {

BinarySet
IndexHNSW::Serialize(const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    try {
        MemoryIOWriter writer;
        index_->saveIndex(writer);
        std::shared_ptr<uint8_t[]> data(writer.data_);

        BinarySet res_set;
        res_set.Append("HNSW", data, writer.rp);
        Disassemble(res_set, config);
        return res_set;
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IndexHNSW::Load(const BinarySet& index_binary) {
    try {
        Assemble(const_cast<BinarySet&>(index_binary));
        auto binary = index_binary.GetByName("HNSW");

        MemoryIOReader reader;
        reader.total = binary->size;
        reader.data_ = binary->data.get();

        hnswlib::SpaceInterface<float>* space = nullptr;
        index_ = std::make_shared<hnswlib::HierarchicalNSW<float>>(space);
        index_->stats_enable_ = (STATISTICS_LEVEL >= 3);
        index_->loadIndex(reader);
#if 0
        auto hnsw_stats = std::static_pointer_cast<LibHNSWStatistics>(stats);
        if (STATISTICS_LEVEL >= 3) {
            auto lock = hnsw_stats->Lock();
            hnsw_stats->update_level_distribution(index_->maxlevel_, index_->level_stats_);
        }
#endif
        // LOG_KNOWHERE_DEBUG_ << "IndexHNSW::Load finished, show statistics:";
        // LOG_KNOWHERE_DEBUG_ << hnsw_stats->ToString();
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IndexHNSW::Train(const DatasetPtr& dataset_ptr, const Config& config) {
    try {
        GET_TENSOR_DATA_DIM(dataset_ptr)

        hnswlib::SpaceInterface<float>* space;
        std::string metric_type = GetMetaMetricType(config);
        if (metric_type == metric::L2) {
            space = new hnswlib::L2Space(dim);
        } else if (metric_type == metric::IP) {
            space = new hnswlib::InnerProductSpace(dim);
        } else {
            KNOWHERE_THROW_MSG("Metric type not supported: " + metric_type);
        }
        index_ = std::make_shared<hnswlib::HierarchicalNSW<float>>(space, rows, GetIndexParamHNSWM(config),
                                                                   GetIndexParamEfConstruction(config));
        index_->stats_enable_ = (STATISTICS_LEVEL >= 3);
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
IndexHNSW::AddWithoutIds(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }

    GET_TENSOR_DATA(dataset_ptr)

    index_->addPoint(p_data, 0);
if (CheckKeyInConfig(config, meta::BUILD_THREAD_NUM))
    omp_set_num_threads(GetMetaBuildThreadNum(config));
#pragma omp parallel for
    for (int i = 1; i < rows; ++i) {
        index_->addPoint((reinterpret_cast<const float*>(p_data) + Dim() * i), i);
    }
#if 0
    if (STATISTICS_LEVEL >= 3) {
        auto hnsw_stats = std::static_pointer_cast<LibHNSWStatistics>(stats);
        auto lock = hnsw_stats->Lock();
        hnsw_stats->update_level_distribution(index_->maxlevel_, index_->level_stats_);
    }
#endif
    // LOG_KNOWHERE_DEBUG_ << "IndexHNSW::Train finished, show statistics:";
    // LOG_KNOWHERE_DEBUG_ << GetStatistics()->ToString();
}

struct cmp {
    bool operator()(const std::pair<int,hnswlib::tableint>& x,std::pair<int,hnswlib::tableint>& y) const {
        if (x.first == y.first)
            return x.second < y.second;
        return x.first > y.first;
    }
};

void 
IndexHNSW::Merge_build(const DatasetPtr& dataset_ptr, const Config& config, const VecIndexPtr index1, const VecIndexPtr index2) {
    GET_TENSOR_DATA_DIM(dataset_ptr)
    hnswlib::SpaceInterface<float>* space;
    int smpN = 8;
    std::string metric_type = GetMetaMetricType(config);
    if (metric_type == metric::L2) {
        space = new hnswlib::L2Space(dim);
    } else if (metric_type == metric::IP) {
        space = new hnswlib::InnerProductSpace(dim);
    } else {
        KNOWHERE_THROW_MSG("Metric type not supported: " + metric_type);
    }
    index_ = std::make_shared<hnswlib::HierarchicalNSW<float>>(space, rows, GetIndexParamHNSWM(config),
                                                                GetIndexParamEfConstruction(config));
    index_->stats_enable_ = (STATISTICS_LEVEL >= 3);
    //copy the data and update the links
    //level0
    std::shared_ptr<IndexHNSW> idx1 = std::dynamic_pointer_cast<IndexHNSW> (index1);
    std::shared_ptr<IndexHNSW> idx2 = std::dynamic_pointer_cast<IndexHNSW> (index2);

    std::shared_ptr<hnswlib::HierarchicalNSW<float>> graph1 = idx1->index_;
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> graph2 = idx2->index_;

    //copy the data and neighbors
    auto level0_offset = graph1->max_elements_ * graph1->size_data_per_element_;
    memcpy(index_->data_level0_memory_, graph1->data_level0_memory_, level0_offset);
    memcpy(index_->data_level0_memory_ + level0_offset, graph2->data_level0_memory_, graph2->max_elements_ * graph2->size_data_per_element_);
    index_->cur_element_count = graph1->cur_element_count + graph2->cur_element_count;
    index_->enterpoint_node_ = graph1->maxlevel_ >= graph2->maxlevel_ ? graph1->enterpoint_node_ : graph2->enterpoint_node_;
    
    auto half = graph1->cur_element_count;
    if (graph1->cur_element_count != graph1->max_elements_ || graph2->cur_element_count != graph2->max_elements_)
        std::cout << "not insert all" << std::endl;
    for(int i = 0;i < index_->cur_element_count;i ++) {
        if (i < half) {
            index_->element_levels_[i] = graph1->element_levels_[i];
            if (graph1->element_levels_[i]) {
                index_->linkLists_[i] = (char*)malloc(index_->size_links_per_element_ * graph1->element_levels_[i] + 1);
                if (index_->linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memcpy(index_->linkLists_[i], graph1->linkLists_[i], index_->size_links_per_element_ * graph1->element_levels_[i] + 1);
            }
        } else {
            index_->element_levels_[i] = graph2->element_levels_[i - half];
            if (graph2->element_levels_[i - half]) {
                index_->linkLists_[i] = (char*)malloc(index_->size_links_per_element_ * graph2->element_levels_[i - half] + 1);
                if (index_->linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memcpy(index_->linkLists_[i], graph2->linkLists_[i - half], index_->size_links_per_element_ * graph2->element_levels_[i - half] + 1);

                for(int level = 1;level <= graph2->element_levels_[i - half];level ++) {
                    unsigned int* data = index_->get_linklist(i, level);
                    int size = index_->getListCount(data);
                    hnswlib::tableint* datal = (hnswlib::tableint*)(data + 1);
                    for (int i = 0; i < size; i++) {
                            hnswlib::tableint& cand = datal[i];
                            cand += half;
                    }           
                }
            }
            unsigned int* data = index_->get_linklist0(i);
            int size = index_->getListCount(data);
            hnswlib::tableint* datal = (hnswlib::tableint*)(data + 1);
            for (int i = 0; i < size; i++) {
                    hnswlib::tableint& cand = datal[i];
                    cand += half;
                } 
            
        }
    }
    //std::cout << "finish to copy neighbor!" << std::endl;

    for(int i = 0;i < index_->cur_element_count;i ++) {
        for(int j = 0;j <= index_->element_levels_[i];j ++) {
            auto data = index_->get_linklist_at_level(i, j);
            int size = index_->getListCount(data);
            hnswlib::tableint* datal = (hnswlib::tableint*)(data + 1);
            for(int k = 0;k < size;k ++) {
                auto cand = datal[k];
                if (index_->element_levels_[cand] < j) {
                    std::cout << "copy neighbor failed!" << std::endl;
                }
            }
        }
    }

    std::vector<std::pair<int,hnswlib::tableint>> node1,node2;
    for(int i = 0;i < half;i ++)
        node1.push_back({index_->element_levels_[i], i});
    for(int i = half;i < index_->cur_element_count;i ++)
        node2.push_back({index_->element_levels_[i], i});
    sort(node1.begin(), node1.end(), cmp());
    sort(node2.begin(), node2.end(), cmp());
    //connect the two graph by random edges
    int pointer1 = 0, pointer2 = 0;
    for(int level = std::min(graph1->maxlevel_, graph2->maxlevel_);level >= 0;level --) {
        if (level == 0)
            pointer1 = node1.size(), pointer2 = node2.size();
        for(;pointer1 < node1.size();pointer1 ++)  
            if (node1[pointer1].first < level)
                break;

        for(;pointer2 < node2.size();pointer2 ++)  
            if (node2[pointer2].first < level)
                break; 
        //printf("now is level %d, g1 has %d points, g2 has %d points.\n", level, pointer1, pointer2);
        using  pqc = std::priority_queue<std::pair<float, hnswlib::tableint>, std::vector<std::pair<float, hnswlib::tableint>>, hnswlib::HierarchicalNSW<float>::CompareByFirst>;
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib1(0, pointer1 - 1),distrib2(0, pointer2 - 1);
        //nn list of all node's neighbor in this layer
        std::vector<std::vector<std::pair<float, hnswlib::tableint>>> anng(pointer1 + pointer2);
        std::vector<pqc> pq_node(index_->cur_element_count);
        std::vector<std::unordered_map<hnswlib::tableint, int>> visited_node(index_->cur_element_count);
#pragma omp parallel for
        for(int i = 0;i < pointer1;i ++) {
            std::set<int> NewNode;
            while (NewNode.size() < std::min(smpN, pointer2)) {
                int randomNum = distrib2(gen);
                NewNode.insert(randomNum);
            }
            for(auto v : NewNode) 
                anng[i].push_back({index_->Calc_dist(node2[v].second, node1[i].second), node2[v].second});

            sort(anng[i].begin(), anng[i].end());
            for(int j = 0;j < NewNode.size();j ++) {
                pq_node[node1[i].second].push(anng[i][j]);
                visited_node[node1[i].second][anng[i][j].second] = 1;
            }
                
        }
#pragma omp parallel for
        for(int i = 0;i < pointer2;i ++) {
            std::set<int> NewNode;
            while (NewNode.size() < std::min(smpN, pointer1)) {
                int randomNum = distrib1(gen);
                NewNode.insert(randomNum);
            }
            for(auto v : NewNode) 
                anng[i + pointer1].push_back({index_->Calc_dist(node1[v].second, node2[i].second), node1[v].second});
            sort(anng[i + pointer1].begin(), anng[i + pointer1].end());
            for(int j = 0;j < NewNode.size();j ++) {
                pq_node[node2[i].second].push(anng[i + pointer1][j]);
                visited_node[node2[i].second][anng[i + pointer1][j].second] = 1;
            }
                
        }
        //std::cout << "finish to init neighbor!" << std::endl;
        std::vector<std::pair<hnswlib::tableint, hnswlib::tableint>> level_point;
        for(int i = 0;i < pointer1 + pointer2;i ++) {
            if (i < pointer1)
                level_point.push_back({node1[i].second, i});
            else level_point.push_back({node2[i - pointer1].second, i});
        }
        
        // for(int i = 0;i < pointer1 + pointer2;i ++) {
        //     if (i < pointer1)
        //         std::cout << node1[i].second << ' ' << index_->element_levels_[node1[i].second] << std::endl;
        //     else std::cout << node2[i - pointer1].second << ' ' << index_->element_levels_[node2[i - pointer1].second] << std::endl;
        // }
        // for(int i = 0;i < pointer1 + pointer2; i ++) {
        //     for(int j = 0;j < anng[i].size();j ++)
        //         printf("%d ", anng[i][j].second);
        //     std::cout << std::endl;
        // }

        // for(int i = 0;i < pointer1 + pointer2; i ++) {
        //     for(int j = 0;j < anng[i].size();j ++)
        //         if (index_->element_levels_[anng[i][j].second] < level) {
        //             std::cout << i << ' ' << j << std::endl;
        //             return;
        //         }
        // }


        std::vector<std::pair<hnswlib::tableint, hnswlib::tableint>> level_point_copy(level_point);
        if (level == 0) {
            int sample_size = int(0.1 * (pointer1 + pointer2));
            std::shuffle(level_point.begin(), level_point.end(), gen);
            level_point.resize(sample_size);
        }

        int all_num = pointer1 + pointer2;
        //printf("%d\n", all_num);
        std::atomic<int> static_num = 0;
        //std::vector<std::unordered_map<hnswlib::tableint, int>> visited_neighbor(index_->cur_element_count);
        //std::cout << "finish to sample neighbor!" << std::endl;
 
        while (1.0 * static_num / all_num <= 0.75) {
            static_num = all_num;
            std::vector<hnswlib::tableint> changed_point(index_->cur_element_count);
#pragma omp parallel for
            for(auto v : level_point) {       
                unsigned int* data = index_->get_linklist_at_level(v.first, level);
                int size = index_->getListCount(data);
                hnswlib::tableint* datal = (hnswlib::tableint*)(data + 1);

                for(int j = 0;j < std::min(int(anng[v.second].size()), smpN);j ++) {
                    hnswlib::tableint q = anng[v.second][j].second;
                    // if (visited_neighbor[v.first].count(q))
                    //     continue;
                    // visited_neighbor[v.first][q] = 1;

                    for(int i = 0; i < std::min(size, smpN);i ++) {
                        hnswlib::tableint p = datal[i];
                        float dist = index_->Calc_dist(p, q);
                        {
                            std::lock_guard<std::mutex> lock(index_->link_list_locks_[p]);
                            if (pq_node[p].size() < index_->ef_construction_ || pq_node[p].top().first > dist) {
                                if (!visited_node[p].count(q)) {
                                    visited_node[p][q] = 1;
                                    changed_point[p] = 1;
                                    pq_node[p].push({dist, q});
                                }
                                while (pq_node[p].size() > index_->ef_construction_) {
                                    visited_node[p].erase(pq_node[p].top().second);
                                    pq_node[p].pop();
                                }

                            }
                        }

                        {
                            std::lock_guard<std::mutex> lock(index_->link_list_locks_[q]);
                            if (pq_node[q].size() < index_->ef_construction_ || pq_node[q].top().first > dist) {
                                if (!visited_node[q].count(p)) {
                                    visited_node[q][p] = 1;
                                    changed_point[q] = 1;
                                    pq_node[q].push({dist, p});
                                }
                                while (pq_node[q].size() > index_->ef_construction_) {
                                    visited_node[q].erase(pq_node[q].top().second);
                                    pq_node[q].pop();
                                }

                            }
                        }
                    }
                }
                
            }
            //std::cout << "finish to nn descent!" << std::endl;
#pragma omp parallel for
            for(int i = 0;i < pointer1 + pointer2;i ++) {
                hnswlib::tableint now = level_point_copy[i].first;
                // if (level == 3)
                //     printf("%d %d\n", now, changed_point[now]);
                if (changed_point[now]) {
                    static_num --;
                    pqc pq(pq_node[now]);
                    anng[i].resize(pq.size());
                    for(int idx = pq.size() - 1;idx >= 0;idx --) {
                        anng[i][idx] = pq.top();
                        pq.pop();
                    }
                }
            }
            //std::cout << "finish to nn descent2!" << std::endl;
            //printf("%d\n", static_num);
        }
        //std::cout << "finish to change point!" << std::endl;
#pragma omp parallel for
        for(auto v : level_point_copy) {
            hnswlib::tableint now = v.first;
            unsigned int* data = index_->get_linklist_at_level(now, level);
            int size = index_->getListCount(data);
            hnswlib::tableint* datal = (hnswlib::tableint*)(data + 1);
            for(int i = 0; i < size;i ++) {
                hnswlib::tableint p = datal[i];
                pq_node[now].push({index_->Calc_dist(now, p), p});
            }
            int Mcurmax = level == 0 ? index_->maxM0_ : index_->maxM_;
            auto neighbors = index_->getNeighborsByHeuristic2(pq_node[now], Mcurmax);
            index_->setListCount(data, neighbors.size());
            for(int i = 0;i < neighbors.size();i ++)
                datal[i] = neighbors[i]; 
        }
        //std::cout << "finish to update neighbor!" << std::endl;
        
    }

}

void
IndexHNSW::Delete_By_Rate(double rate) {
    int num = index_->cur_element_count * rate;
    //int last = index_->cur_element_count - num;
    std::unordered_map<hnswlib::tableint, int> del_set;
    // assume that delete is simple ,but below Alg is designed for all situation.
    for(int i = num;i < index_->cur_element_count;i ++) 
        del_set[i] = 1;
    std::cout << "deleted points are " << index_->cur_element_count - num << std::endl;

    //int bad_num = 0;
    using  pqc = std::priority_queue<std::pair<float, hnswlib::tableint>, std::vector<std::pair<float, hnswlib::tableint>>, hnswlib::HierarchicalNSW<float>::CompareByFirst>;
    std::vector<std::pair<int, hnswlib::tableint>> level_point;
    for(int i = 0;i < index_->cur_element_count;i ++) {
        if (!del_set.count(i))
            level_point.push_back({index_->element_levels_[i], i});
    }
    std::sort(level_point.begin(), level_point.end(), cmp());
    if (del_set.count(index_->enterpoint_node_)) {
        index_->enterpoint_node_ = level_point[0].second;
        index_->maxlevel_ = level_point[0].first;
    }
    printf("%d %d\n", index_->enterpoint_node_, index_->maxlevel_);

    int pointer = 0;
    for(int level = index_->maxlevel_;level >= 0;level --) {
        if (level == 0)
            pointer = level_point.size();
        for(;pointer < level_point.size(); pointer ++) {
            if (level_point[pointer].first < level)
                break;
        }
        std::vector<pqc> candidate(pointer);
        //std::vector<std::atomic<bool>> changed(pointer);
        std::vector<std::atomic<bool>> changed(pointer);
        std::mutex mtx;
#pragma omp parallel for
        for(int i = 0;i < pointer;i ++) {
            unsigned int* data = index_->get_linklist_at_level(level_point[i].second, level);
            int size = index_->getListCount(data);
            hnswlib::tableint* datal = (hnswlib::tableint*)(data + 1);
            for(int j = 0;j < size;j ++) {
                hnswlib::tableint cand = datal[j];
                if (del_set.count(cand)) {
                    changed[i] = true;
                    break;
                }
            }

            if (changed[i]) 
                for(int j = 0;j < size;j ++) {
                    hnswlib::tableint cand = datal[j];
                    if (!del_set.count(cand)) {
                        float dist = index_->Calc_dist(cand, level_point[i].second);
                        candidate[i].push({dist, cand});
                    } else {
                        unsigned int* d_data = index_->get_linklist_at_level(cand, level);
                        int d_size = index_->getListCount(d_data);
                        hnswlib::tableint* d_datal = (hnswlib::tableint*)(d_data + 1);
                        for(int k = 0;k < d_size;k ++) {
                            hnswlib::tableint d_cand = d_datal[k];
                            if (!del_set.count(d_cand)) {
                                float d = index_->Calc_dist(d_cand, level_point[i].second);
                                candidate[i].push({d, d_cand});
                            } 
                        }
                    }
                }
                
            }
#pragma omp parallel for
            //update while all points are finished
            for(int i = 0;i < pointer;i ++) {
                if (changed[i]) {
                    unsigned int* data = index_->get_linklist_at_level(level_point[i].second, level);
                    int Mcurmax = level == 0 ? index_->maxM0_ : index_->maxM_;
                    auto neighbors = index_->getNeighborsByHeuristic2(candidate[i], Mcurmax);
                    index_->setListCount(data, neighbors.size());
                    hnswlib::tableint* datal = (hnswlib::tableint*)(data + 1);
                    for(int j = 0;j < neighbors.size();j ++) {
                        datal[j] = neighbors[j];
                }

            }
        }
    }
    //printf("bad num is %d\n", bad_num);
    // if (del_set.count(index_->enterpoint_node_))
    //     puts("del error 1");
    // for(int i = 0;i < index_->cur_element_count;i ++) {
    //     if (del_set.count(i))
    //         continue;
    //     for(int j = 0;j <= index_->element_levels_[i];j ++) {
    //         auto data = index_->get_linklist_at_level(i, j);
    //         int size = index_->getListCount(data);
    //         for(int k = 1;k <= size;k ++)
    //             if (del_set.count(data[k])) {
    //                 puts("del error 2");
    //                 printf("%d %d %d\n", i, index_->element_levels_[i], data[k]);
    //                 exit(1);
    //             }
    //     }
    // }

    // auto data = index_->get_linklist_at_level(index_->enterpoint_node_, index_->maxlevel_);
    // index_->getListCount(data);
}

DatasetPtr
IndexHNSW::GetVectorById(const DatasetPtr& dataset_ptr, const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }

    GET_DATA_WITH_IDS(dataset_ptr)

    float* p_x = nullptr;
    try {
        p_x = new float[dim * rows];
        for (int64_t i = 0; i < rows; i++) {
            int64_t id = p_ids[i];
            KNOWHERE_THROW_IF_NOT_FMT(id >= 0 && id < index_->cur_element_count, "invalid id %ld", id);
            memcpy(p_x + i * dim, index_->getDataByInternalId(id), dim * sizeof(float));
        }
    } catch (std::exception& e) {
        if (p_x != nullptr) {
            delete[] p_x;
        }
        KNOWHERE_THROW_MSG(e.what());
    }
    return GenResultDataset(p_x);
}

DatasetPtr
IndexHNSW::Query(const DatasetPtr& dataset_ptr, const Config& config, const faiss::BitsetView bitset) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }
    GET_TENSOR_DATA_DIM(dataset_ptr)

    auto k = GetMetaTopk(config);
    auto p_id = new int64_t[k * rows];
    auto p_dist = new float[k * rows];
    std::vector<hnswlib::StatisticsInfo> query_stats;
    auto hnsw_stats = std::dynamic_pointer_cast<LibHNSWStatistics>(stats);
    if (STATISTICS_LEVEL >= 3) {
        query_stats.resize(rows);
        for (auto i = 0; i < rows; ++i) {
            query_stats[i].target_level_ = hnsw_stats->target_level;
        }
    }

    size_t ef = GetIndexParamEf(config);
    hnswlib::SearchParam param{ef};
    bool transform = (index_->metric_type_ == 1);  // InnerProduct: 1

    std::chrono::high_resolution_clock::time_point query_start, query_end;
    query_start = std::chrono::high_resolution_clock::now();

if (CheckKeyInConfig(config, meta::QUERY_THREAD_NUM))
    omp_set_num_threads(GetMetaQueryThreadNum(config));
#pragma omp parallel for
    for (unsigned int i = 0; i < rows; ++i) {
        auto single_query = (float*)p_data + i * dim;
        std::priority_queue<std::pair<float, hnswlib::labeltype>> rst;
        if (STATISTICS_LEVEL >= 3) {
            rst = index_->searchKnn(single_query, k, bitset, query_stats[i], &param);
        } else {
            auto dummy_stat = hnswlib::StatisticsInfo();
            rst = index_->searchKnn(single_query, k, bitset, dummy_stat, &param);
        }
        size_t rst_size = rst.size();

        auto p_single_dis = p_dist + i * k;
        auto p_single_id = p_id + i * k;
        size_t idx = rst_size - 1;
        while (!rst.empty()) {
            auto& it = rst.top();
            p_single_dis[idx] = transform ? (1 - it.first) : it.first;
            p_single_id[idx] = it.second;
            rst.pop();
            idx--;
        }

        for (idx = rst_size; idx < k; idx++) {
            p_single_dis[idx] = float(1.0 / 0.0);
            p_single_id[idx] = -1;
        }
    }
    query_end = std::chrono::high_resolution_clock::now();

#if 0
    if (STATISTICS_LEVEL) {
        auto lock = hnsw_stats->Lock();
        if (STATISTICS_LEVEL >= 1) {
            hnsw_stats->update_nq(rows);
            hnsw_stats->update_ef_sum(index_->ef_ * rows);
            hnsw_stats->update_total_query_time(
                std::chrono::duration_cast<std::chrono::milliseconds>(query_end - query_start).count());
        }
        if (STATISTICS_LEVEL >= 2) {
            hnsw_stats->update_filter_percentage(bitset);
        }
        if (STATISTICS_LEVEL >= 3) {
            for (auto i = 0; i < rows; ++i) {
                for (auto j = 0; j < query_stats[i].accessed_points.size(); ++j) {
                    auto tgt = hnsw_stats->access_cnt_map.find(query_stats[i].accessed_points[j]);
                    if (tgt == hnsw_stats->access_cnt_map.end())
                        hnsw_stats->access_cnt_map[query_stats[i].accessed_points[j]] = 1;
                    else
                        tgt->second += 1;
                }
            }
        }
    }
#endif
    // LOG_KNOWHERE_DEBUG_ << "IndexHNSW::Query finished, show statistics:";
    // LOG_KNOWHERE_DEBUG_ << GetStatistics()->ToString();

    return GenResultDataset(p_id, p_dist);
}

DatasetPtr
IndexHNSW::QueryByRange(const DatasetPtr& dataset,
                        const Config& config,
                        const faiss::BitsetView bitset) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }
    GET_TENSOR_DATA_DIM(dataset)

    auto range_k = GetIndexParamHNSWK(config);
    auto radius = GetMetaRadius(config);
    size_t ef = GetIndexParamEf(config);
    hnswlib::SearchParam param{ef};
    bool is_IP = (index_->metric_type_ == 1);  // InnerProduct: 1

    if (!is_IP) {
        radius *= radius;
    }

    std::vector<std::vector<int64_t>> result_id_array(rows);
    std::vector<std::vector<float>> result_dist_array(rows);
    std::vector<size_t> result_lims(rows + 1, 0);

//#pragma omp parallel for
    for (unsigned int i = 0; i < rows; ++i) {
        auto single_query = (float*)p_data + i * dim;

        auto dummy_stat = hnswlib::StatisticsInfo();
        auto rst =
            index_->searchRange(single_query, range_k, (is_IP ? 1.0f - radius : radius), bitset, dummy_stat, &param);

        for (auto& p : rst) {
            result_dist_array[i].push_back(is_IP ? (1 - p.first) : p.first);
            result_id_array[i].push_back(p.second);
        }
        result_lims[i+1] = result_lims[i] + rst.size();
    }

    LOG_KNOWHERE_DEBUG_ << "Range search radius: " << radius << ", result num: " << result_lims.back();

    auto p_id = new int64_t[result_lims.back()];
    auto p_dist = new float[result_lims.back()];
    auto p_lims = new size_t[rows + 1];

    for (int64_t i = 0; i < rows; i++) {
        size_t start = result_lims[i];
        size_t size = result_lims[i+1] - result_lims[i];
        memcpy(p_id + start, result_id_array[i].data(), size * sizeof(int64_t));
        memcpy(p_dist + start, result_dist_array[i].data(), size * sizeof(float));
    }
    memcpy(p_lims, result_lims.data(), (rows + 1) * sizeof(size_t));

    return GenResultDataset(p_id, p_dist, p_lims);
}

int64_t
IndexHNSW::Count() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->cur_element_count;
}

int64_t
IndexHNSW::Dim() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return (*static_cast<size_t*>(index_->dist_func_param_));
}

int64_t
IndexHNSW::Size() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    return index_->cal_size();
}

int64_t
IndexHNSW::Getcmp() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    auto tmp = index_->get_cmp();
    return tmp;
}

int64_t
IndexHNSW::Get_metric_cmp() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize");
    }
    auto tmp = index_->get_metric_cmp();
    return tmp;
}

#if 0
void
IndexHNSW::ClearStatistics() {
    if (!STATISTICS_LEVEL)
        return;
    auto hnsw_stats = std::static_pointer_cast<LibHNSWStatistics>(stats);
    auto lock = hnsw_stats->Lock();
    hnsw_stats->clear();
}
#endif

}  // namespace knowhere
