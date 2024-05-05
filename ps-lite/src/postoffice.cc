/**
 *  Copyright (c) 2015 by Contributors
 */
#include <unistd.h>
#include <thread>
#include <chrono>
#include "ps/internal/postoffice.h"
#include "ps/internal/message.h"
#include "ps/base.h"
#include "common/logging.h"

namespace ps {

Postoffice *Postoffice::po_scheduler_ = nullptr;
std::mutex Postoffice::init_mu_;
std::vector<Postoffice *> Postoffice::po_worker_group_;
std::vector<Postoffice *> Postoffice::po_server_group_;
bool Postoffice::initialized_ = false;

void Postoffice::Init(ps::Node::Role role) {
    std::lock_guard<std::mutex> lk(init_mu_);
    if (initialized_)
        return;

    int group_size = 1;
    auto val = Environment::Get()->find("DMLC_GROUP_SIZE");
    if (val)
        group_size = atoi(val);
    CHECK(group_size >= 1);

    if (role == ps::Node::SCHEDULER) {
        po_scheduler_ = new Postoffice(0);
    }
    if (role == ps::Node::WORKER) {
        for (int i = 0; i < group_size; ++i) {
            po_worker_group_.push_back(new Postoffice(i));
        }
    }
    if (role == ps::Node::SERVER) {
        for (int i = 0; i < group_size; ++i) {
            po_server_group_.push_back(new Postoffice(i));
        }
    }
    initialized_ = true;
}

Postoffice::Postoffice(int instance_idx) {
    env_ref_ = Environment::_GetSharedRef();
    instance_idx_ = instance_idx;
}

void Postoffice::InitEnvironment() {
    const char *val = NULL;
    std::string van_type = GetEnv("DMLC_PS_VAN_TYPE", "zmq");
    van_ = Van::Create(van_type, this);
    val = Environment::Get()->find("DMLC_GROUP_SIZE");
    group_size_ = val ? atoi(val) : 1;
    val = CHECK_NOTNULL(Environment::Get()->find("DMLC_NUM_WORKER"));
    num_workers_ = atoi(val);
    val = CHECK_NOTNULL(Environment::Get()->find("DMLC_NUM_SERVER"));
    num_servers_ = atoi(val);
    val = CHECK_NOTNULL(Environment::Get()->find("DMLC_ROLE"));
    std::string role(val);
    is_worker_ = role == "worker";
    is_server_ = role == "server";
    is_scheduler_ = role == "scheduler";
    verbose_ = GetEnv("PS_VERBOSE", 0);
}

void Postoffice::Start(int customer_id, const Node::Role role,
                       const bool do_barrier, const char *argv0) {
    start_mu_.lock();
    if (init_stage_ == 0) {
        InitEnvironment();
        // init glog
        if (argv0) {
            dmlc::InitLogging(argv0);
        } else {
            dmlc::InitLogging("ps-lite\0");
        }

        // init node info.
        for (int i = 0; i < num_workers_ * group_size_; ++i) {
            int id = WorkerRankToID(i);
            for (int g : {id, kWorkerGroup, kWorkerGroup + kServerGroup,
                          kWorkerGroup + kScheduler,
                          kWorkerGroup + kServerGroup + kScheduler}) {
                node_ids_[g].push_back(id);
            }
        }

        for (int i = 0; i < num_servers_ * group_size_; ++i) {
            int id = ServerRankToID(i);
            for (int g : {id, kServerGroup, kWorkerGroup + kServerGroup,
                          kServerGroup + kScheduler,
                          kWorkerGroup + kServerGroup + kScheduler}) {
                node_ids_[g].push_back(id);
            }
        }

        for (int g : {kScheduler, kScheduler + kServerGroup + kWorkerGroup,
                      kScheduler + kWorkerGroup, kScheduler + kServerGroup}) {
            node_ids_[g].push_back(kScheduler);
        }
        init_stage_++;
    }
    LOG(INFO) << "Ready to unlock";
    start_mu_.unlock();
    LOG(INFO) << "Unlock done";

    // start van
    van_->Start(customer_id, false);
    start_mu_.lock();
    if (init_stage_ == 1) {
        // record start time
        start_time_ = time(NULL);
        init_stage_++;
    }
    start_mu_.unlock();
    LOG(INFO) << "Do a barrier";
    // do a barrier here
    if (do_barrier)
        Barrier(customer_id, kWorkerGroup + kServerGroup + kScheduler);
    LOG(INFO) << "Exit the barrier";
}

void Postoffice::Finalize(const int customer_id, const bool do_barrier) {
    if (init_stage_ == 0)
        return;
    if (do_barrier)
        Barrier(customer_id, kWorkerGroup + kServerGroup + kScheduler);
    if (customer_id == 0) {
        num_workers_ = 0;
        num_servers_ = 0;
        van_->Stop();
        init_stage_ = 0;
        customers_.clear();
        node_ids_.clear();
        barrier_done_.clear();
        server_key_ranges_.clear();
        heartbeats_.clear();
        if (exit_callback_)
            exit_callback_();
    }
}

void Postoffice::AddCustomer(Customer *customer) {
    std::lock_guard<std::mutex> lk(mu_);
    int app_id = CHECK_NOTNULL(customer)->app_id();
    // check if the customer id has existed
    int customer_id = CHECK_NOTNULL(customer)->customer_id();
    CHECK_EQ(customers_[app_id].count(customer_id), (size_t)0)
        << "customer_id " << customer_id << " already exists\n";
    customers_[app_id].insert(std::make_pair(customer_id, customer));
    std::unique_lock<std::mutex> ulk(barrier_mu_);
    barrier_done_[app_id].insert(std::make_pair(customer_id, false));
}

void Postoffice::RemoveCustomer(Customer *customer) {
    std::lock_guard<std::mutex> lk(mu_);
    int app_id = CHECK_NOTNULL(customer)->app_id();
    int customer_id = CHECK_NOTNULL(customer)->customer_id();
    customers_[app_id].erase(customer_id);
    if (customers_[app_id].empty()) {
        customers_.erase(app_id);
    }
}

Customer *Postoffice::GetCustomer(int app_id, int customer_id,
                                  int timeout) const {
    Customer *obj = nullptr;
    for (int i = 0; i < timeout * 1000 + 1; ++i) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            const auto it = customers_.find(app_id);
            if (it != customers_.end()) {
                std::unordered_map<int, Customer *> customers_in_app =
                    it->second;
                obj = customers_in_app[customer_id];
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return obj;
}

void Postoffice::Barrier(int customer_id, int node_group) {
    int node_group_size = static_cast<int>(GetNodeIDs(node_group).size());
    if (node_group_size <= 1)
        return;
    if (node_group_size <= group_size_)
        return;
    auto role = van_->my_node().role;
    if (role == Node::SCHEDULER) {
        CHECK(node_group & kScheduler);
    } else if (role == Node::WORKER) {
        CHECK(node_group & kWorkerGroup);
    } else if (role == Node::SERVER) {
        CHECK(node_group & kServerGroup);
    }
    std::unique_lock<std::mutex> ulk(barrier_mu_);
    barrier_done_[0][customer_id] = false;
    Message req;
    req.meta.recver = kScheduler;
    req.meta.request = true;
    req.meta.control.cmd = Control::BARRIER;
    // req.meta.control.cmd = Control::INSTANCE_BARRIER;
    req.meta.app_id = 0;
    req.meta.customer_id = customer_id;
    req.meta.control.barrier_group = node_group;
    req.meta.timestamp = van_->GetTimestamp();
    CHECK_GT(van_->Send(req), 0);
    barrier_cond_.wait(
        ulk, [this, customer_id] { return barrier_done_[0][customer_id]; });
}

const std::vector<Range> &Postoffice::GetServerKeyRanges() {
    server_key_ranges_mu_.lock();
    if (server_key_ranges_.empty()) {
        for (int i = 0; i < num_servers_; ++i) {
            server_key_ranges_.push_back(Range(
                kMaxKey / num_servers_ * i, kMaxKey / num_servers_ * (i + 1)));
        }
    }
    server_key_ranges_mu_.unlock();
    return server_key_ranges_;
}

void Postoffice::Manage(const Message &recv) {
    CHECK(!recv.meta.control.empty());
    const auto &ctrl = recv.meta.control;
    if ((ctrl.cmd == Control::BARRIER || ctrl.cmd == Control::INSTANCE_BARRIER)
        && !recv.meta.request) {
        barrier_mu_.lock();
        auto size = barrier_done_[recv.meta.app_id].size();
        for (size_t customer_id = 0; customer_id < size; customer_id++) {
            barrier_done_[recv.meta.app_id][customer_id] = true;
        }
        barrier_mu_.unlock();
        barrier_cond_.notify_all();
    }
}

std::vector<int> Postoffice::GetDeadNodes(int t) {
    std::vector<int> dead_nodes;
    if (!van_->IsReady() || t == 0)
        return dead_nodes;

    time_t curr_time = time(NULL);
    const auto &nodes = is_scheduler_ ?
                            GetNodeIDs(kWorkerGroup + kServerGroup) :
                            GetNodeIDs(kScheduler);
    {
        std::lock_guard<std::mutex> lk(heartbeat_mu_);
        for (int r : nodes) {
            auto it = heartbeats_.find(r);
            if ((it == heartbeats_.end() || it->second + t < curr_time)
                && start_time_ + t < curr_time) {
                dead_nodes.push_back(r);
            }
        }
    }
    return dead_nodes;
}
} // namespace ps
