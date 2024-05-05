/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef PS_INTERNAL_POSTOFFICE_H_
#define PS_INTERNAL_POSTOFFICE_H_
#include <mutex>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <memory>
#include "ps/range.h"
#include "ps/internal/env.h"
#include "ps/internal/customer.h"
#include "ps/internal/van.h"
namespace ps {
/**
 * \brief the center of the system
 */
class Postoffice {
public:
    /**
     * \brief return the first valid Postoffice instance in the following order:
     * scheduler, server, worker.
     */
    static Postoffice *Get() {
        CHECK(initialized_) << "Please call ps::StartPS() first";
        if (po_scheduler_)
            return po_scheduler_;
        if (po_server_group_.size())
            return po_server_group_.at(0);
        return po_worker_group_.at(0);
    }

    /**
     * \brief return the Postoffice instance for scheduler if it exists.
     * Otherwise, return the one for the server.
     * \param index the instance offset inside the worker group.
     * it should be less than DMLC_GROUP_SIZE.
     */
    static Postoffice *GetServer(int index = 0) {
        CHECK(initialized_) << "Please call ps::StartPS() first";
        if (po_scheduler_)
            return po_scheduler_;
        return po_server_group_.at(index);
    }

    /**
     * \brief return the Postoffice instance for scheduler.
     */
    static Postoffice *GetScheduler() {
        CHECK(initialized_) << "Please call ps::StartPS() first";
        return po_scheduler_;
    }

    /**
     * \brief return the Postoffice instance for worker.
     * \param index the instance offset inside the worker group.
     * it should be less than DMLC_GROUP_SIZE.
     */
    static Postoffice *GetWorker(int index = 0) {
        CHECK(initialized_) << "Please call ps::StartPS() first";
        return po_worker_group_.at(index);
    }
    // /**
    //  * \brief return the singleton object
    //  */
    // static Postoffice *Get() {
    //     static Postoffice e;
    //     return &e;
    // }
    /** \brief get the van */
    Van *van() {
        return van_;
    }
    /**
     * \brief start the system
     *
     * This function will block until every nodes are started.
     * \param argv0 the program name, used for logging.
     * \param do_barrier whether to block until every nodes are started.
     */
    void Start(int customer_id, const Node::Role role, const bool do_barrier,
               const char *argv0);
    /**
     * \brief terminate the system
     *
     * All nodes should call this function before existing.
     * \param do_barrier whether to do block until every node is finalized,
     * default true.
     */
    void Finalize(const int customer_id, const bool do_barrier = true);
    /**
     * \brief add an customer to the system. threadsafe
     */
    void AddCustomer(Customer *customer);
    /**
     * \brief remove a customer by given it's id. threasafe
     */
    void RemoveCustomer(Customer *customer);
    /**
     * \brief get the customer by id, threadsafe
     * \param app_id the application id
     * \param customer_id the customer id
     * \param timeout timeout in sec
     * \return return nullptr if doesn't exist and timeout
     */
    Customer *GetCustomer(int app_id, int customer_id, int timeout = 0) const;
    
    int NumofCustomer() const {
        std::lock_guard<std::mutex> lk(mu_);
        return customers_.size();
    }
    /**
     * \brief get the id of a node (group), threadsafe
     *
     * if it is a node group, return the list of node ids in this
     * group. otherwise, return {node_id}
     */
    const std::vector<int> &GetNodeIDs(int node_id) const {
        const auto it = node_ids_.find(node_id);
        CHECK(it != node_ids_.cend()) << "node " << node_id << " doesn't exist";
        return it->second;
    }
    /**
     * \brief return the key ranges of all server nodes
     */
    const std::vector<Range> &GetServerKeyRanges();
    /**
     * \brief the template of a callback
     */
    using Callback = std::function<void()>;
    /**
     * \brief Register a callback to the system which is called after Finalize()
     *
     * The following codes are equal
     * \code {cpp}
     * RegisterExitCallback(cb);
     * Finalize();
     * \endcode
     *
     * \code {cpp}
     * Finalize();
     * cb();
     * \endcode
     * \param cb the callback function
     */
    void RegisterExitCallback(const Callback &cb) {
        exit_callback_ = cb;
    }
    /**
     * \brief convert from a worker rank into a node id
     * \param rank the worker rank
     */
    static inline int WorkerRankToID(int rank) {
        return rank * 2 + 9;
    }
    /**
     * \brief convert from a server rank into a node id
     * \param rank the server rank
     */
    static inline int ServerRankToID(int rank) {
        return rank * 2 + 8;
    }
    /**
     * \brief convert from a node id into a server or worker rank
     * \param id the node id
     */
    static inline int IDtoRank(int id) {
#ifdef _MSC_VER
#undef max
#endif
        return std::max((id - 8) / 2, 0);
    }
    int group_size() const {
        return group_size_;
    }
    /** \brief Returns the number of worker nodes */
    int num_workers() const {
        return num_workers_;
    }
    /** \brief Returns the number of server nodes */
    int num_servers() const {
        return num_servers_;
    }
    /** \brief Returns the number of worker instances */
    int num_worker_instances() const {
        return num_workers_ * group_size_;
    }
    /** \brief Returns the number of server instances */
    int num_server_instances() const {
        return num_servers_ * group_size_;
    }
    /** \brief Returns the rank of this node in its group
     *
     * Each worker will have a unique rank within [0, NumWorkers()). So are
     * servers. This function is available only after \ref Start has been
     * called.
     */
    int my_rank() const {
        return IDtoRank(van_->my_node().id);
    }

    int preferred_rank() const {
        return preferred_rank_;
    }

    /** \brief Returns true if this node is a worker node */
    int is_worker() const {
        return is_worker_;
    }
    /** \brief Returns true if this node is a server node. */
    int is_server() const {
        return is_server_;
    }
    /** \brief Returns true if this node is a scheduler node. */
    int is_scheduler() const {
        return is_scheduler_;
    }

    std::string role_str() {
        std::string role;
        if (is_worker_)
            return "worker";
        if (is_server_)
            return "server";
        if (is_scheduler_)
            return "scheduler";
        return role;
    }

    static bool initialized() {
        return initialized_;
    }

    /** \brief Returns the verbose level. */
    int verbose() const {
        return verbose_;
    }
    /** \brief Return whether this node is a recovery node */
    bool is_recovery() const {
        return van_->my_node().is_recovery;
    }
    /**
     * \brief barrier
     * \param node_id the barrier group id
     */
    void Barrier(int customer_id, int node_group);
    /**
     * \brief process a control message, called by van
     * \param the received message
     */
    void Manage(const Message &recv);
    /**
     * \brief update the heartbeat record map
     * \param node_id the \ref Node id
     * \param t the last received heartbeat time
     */
    void UpdateHeartbeat(int node_id, time_t t) {
        std::lock_guard<std::mutex> lk(heartbeat_mu_);
        heartbeats_[node_id] = t;
    }
    /**
     * \brief get node ids that haven't reported heartbeats for over t seconds
     * \param t timeout in sec
     */
    std::vector<int> GetDeadNodes(int t = 60);

    // initialize all instances in the group for this role
    static void Init(ps::Node::Role role);

private:
    Postoffice(int instance_idx);
    ~Postoffice() {
        delete van_;
    }

    static Postoffice *po_scheduler_;
    static std::mutex init_mu_;
    // the group of postoffices for workers
    static std::vector<Postoffice *> po_worker_group_;
    // the group of postoffices for servers
    static std::vector<Postoffice *> po_server_group_;

    // initialization
    static bool initialized_;

    void InitEnvironment();
    Van *van_;
    mutable std::mutex mu_;
    // app_id -> (customer_id -> customer pointer)
    std::unordered_map<int, std::unordered_map<int, Customer *>> customers_;
    std::unordered_map<int, std::vector<int>> node_ids_;
    std::mutex server_key_ranges_mu_;
    std::vector<Range> server_key_ranges_;
    bool is_worker_, is_server_, is_scheduler_;
    int num_servers_, num_workers_, group_size_;

    // a hint for preferred rank
    int preferred_rank_;
    std::unordered_map<int, std::unordered_map<int, bool>> barrier_done_;
    int verbose_;
    std::mutex barrier_mu_;
    std::condition_variable barrier_cond_;
    std::mutex heartbeat_mu_;
    std::mutex start_mu_;
    int init_stage_ = 0;
    int instance_idx_ = 0;
    std::unordered_map<int, time_t> heartbeats_;
    Callback exit_callback_;
    /** \brief Holding a shared_ptr to prevent it from being destructed too
     * early */
    std::shared_ptr<Environment> env_ref_;
    time_t start_time_;
    DISALLOW_COPY_AND_ASSIGN(Postoffice);
};

/** \brief verbose log */
#define PS_VLOG(x) LOG_IF(INFO, x <= Postoffice::Get()->verbose())
} // namespace ps
#endif // PS_INTERNAL_POSTOFFICE_H_
