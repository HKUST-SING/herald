#pragma once

#include "common/sarray.h"
#include "common/thread_safe_hash_map.h"
#include "ps/internal/mem_pool.h"
#include "ps/internal/postoffice.h"
#include "ps/psf/PSFunc.h"
#include "ps/psf/serializer.h"
#include "callback_store.h"
#include "ps/kvapp.h"
#include "ps/partitioner.h"
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <utility>
#include <vector>
#include <memory>
#include <fstream>
#include <unordered_map>
#include <string>
#include <sstream>

namespace ps {

template <PsfType>
struct KVWorkerRegisterHelper;

class KVWorker : private KVApp {
public:
    Partitioner *par;
    /**
     * \brief constructor
     *
     * \param app_id the app id, should match with \ref KVServer's id
     * \param customer_id the customer id which is unique locally
     */
    explicit KVWorker(int app_id, int customer_id) : KVApp(app_id) {
        KVAppRegisterHelper<PsfType(0), KVWorker>::init(this);
        par = new AveragePartitioner(); // now use naive partitioner
    }

    ~KVWorker() {
        delete par;
        if (logOut.is_open())
            logOut.close();
    }

    void startRecord(std::string dirPath) {
        logOut.open(dirPath + "/loads_" + std::to_string(MyRank()) + ".txt");
        assert(logOut.is_open());
    }

    void recordLoads() {
        for (auto iter = loads.begin(); iter != loads.end(); ++iter) {
            logOut << getPSFunctionName(iter->first) << ": "
                   << (iter->second).first << ' ' << (iter->second).second
                   << std::endl;
        }
        logOut << std::endl;
        loads.clear();
    }

    void PackDataToMemBuf(Message &msg) {
        // move data to dedicated buffer registered for RDMA
        // we tend to allocate a oversized buffer for each message to avoid
        // frequent allocation
        auto len = msg.meta.data_size * MEM_BUF_SCALE;
        auto buffer = (char *)mem_pool.getMemBuf(msg.meta.psftype, len);
        CHECK(buffer != nullptr);
        memset(buffer, 0, len);
        msg.meta.local_data_addr = reinterpret_cast<uint64_t>(buffer);
        PS_VLOG(3) << "Worker allocates send data buffer at "
                   << msg.meta.local_data_addr << " with len " << len;
        for (size_t i = 0; i < msg.data.size(); i++) {
            auto &x = msg.data[i];
            if (x.size() == 0)
                continue;
            memcpy(buffer, x.data(), x.size());
            x.reset(buffer, x.size(), [](void *) {});
            PS_VLOG(3) << "Reallocate item " << i << " at "
                       << reinterpret_cast<uint64_t>(buffer) << " with len "
                       << x.size();
            buffer += x.size();
        }
        msg.meta.local_data_buf_len = len;
    }

    /**
     * \brief Waits until a Request has been finished
     *
     * Sample usage:
     * \code
     *   _kvworker.Wait(ts);
     * \endcode
     *
     * \param timestamp the timestamp returned by kvworker.Request
     */
    void Wait(int timestamp) {
        obj_->WaitRequest(timestamp);
    }
    /**
     * \brief make a new Request
     *
     * Sample usage:
     * \code
     *   int ts = _kvworker.Request<DensePush>(request, callback);
     * \endcode
     *
     * \param request create request by PSFData<PsfType>::Request
     * \param cb the callback returned by getCallback<PSfType>(args...)
     */
    template <PsfType ftype, typename Tuple, typename CallBack>
    int Request(const Tuple &request, const CallBack &cb) {
        int timestamp = obj_->NewRequest(kServerGroup);
        CallbackStore<ftype>::Get()->store(timestamp, cb);
        // Find the server
        Key key = get<0>(request);
        int target_server_id = par->queryServer(key);
        // Create message
        Message msg;
        tupleEncode(request, msg.data);
        if (logOut.is_open()) {
            for (auto x : msg.data) {
                loads[ftype].first += x.size();
            }
        }
        PS_VLOG(3) << "Worker sends requests: " << ftype;
        msg.meta.app_id = obj_->app_id();
        msg.meta.customer_id = obj_->customer_id();
        msg.meta.timestamp = timestamp;
        msg.meta.sender = Postoffice::Get()->van()->my_node().id;
        msg.meta.recver = Postoffice::Get()->ServerRankToID(target_server_id);
        msg.meta.psftype = ftype;
        msg.meta.request = true;
        msg.set_data_num_and_length();
        Postoffice::Get()->van()->Send(msg);
        return timestamp;
    }

    int RequestForParamInit(
        const PSFData<ParamInit>::Request request,
        const std::function<void(const PSFData<ParamInit>::Response &)> &cb) {
        int timestamp = obj_->NewRequest(kServerGroup);
        auto constexpr ftype = ParamInit;
        CallbackStore<ftype>::Get()->store(timestamp, cb);
        // Find the server
        Key key = get<0>(request);
        int target_server_id = par->queryServer(key);
        // Create message
        Message msg;
        // only one element is SArray, others are scalers
        msg.data.clear();
        tupleEncode(request, msg.data);
        if (logOut.is_open()) {
            for (auto x : msg.data) {
                loads[ftype].first += x.size();
            }
        }
        msg.meta.app_id = obj_->app_id();
        msg.meta.customer_id = obj_->customer_id();
        msg.meta.timestamp = timestamp;
        msg.meta.sender = Postoffice::Get()->van()->my_node().id;
        msg.meta.recver = Postoffice::Get()->ServerRankToID(target_server_id);
        msg.meta.psftype = ftype;
        msg.meta.request = true;
        msg.meta.push = true;
        // use cache table key to identify the message
        msg.meta.key = key;
        if (ftype == ParamInit) {
            auto width = get<3>(request);
            if (try_lookup_and_insert(key, width))
                PS_VLOG(3) << "Worker registered cache table width";
            else
                LOG(FATAL) << "Failed to register cache table at worker";
        }
        msg.set_data_num_and_length();
        Postoffice::Get()->van()->Send(msg);
        return timestamp;
    }

    int RequestForSync(
        const PSFData<kSyncEmbedding>::Request &request,
        const std::function<void(const PSFData<kSyncEmbedding>::Response &)>
            &cb) {
        auto constexpr ftype = kSyncEmbedding;
        int timestamp = obj_->NewRequest(kServerGroup);
        CallbackStore<ftype>::Get()->store(timestamp, cb);
        // Find the server
        Key key = get<0>(request);
        int target_server_id = par->queryServer(key);
        // Create message
        Message msg;
        tupleEncode(request, msg.data);
        if (logOut.is_open()) {
            for (auto x : msg.data) {
                loads[ftype].first += x.size();
            }
        }
        msg.meta.app_id = obj_->app_id();
        msg.meta.customer_id = obj_->customer_id();
        msg.meta.timestamp = timestamp;
        msg.meta.sender = Postoffice::Get()->van()->my_node().id;
        msg.meta.recver = Postoffice::Get()->ServerRankToID(target_server_id);
        msg.meta.psftype = ftype;
        msg.meta.request = true;
        msg.meta.push = false;
        msg.meta.key = key;
        msg.meta.version_bound = get<3>(request);
        if (ftype == kSyncEmbedding) {
            if (table_width.find(key) == table_width.end()) {
                LOG(FATAL) << "Table width should be recorded";
            }
            size_t width = table_width[key];
            auto &keys = get<1>(request);
            size_t value_set_size = keys.size() * width;
            size_t mem_len = keys.size() * 2 + value_set_size;
            mem_len = mem_len * sizeof(uint64_t);
            msg.meta.val_len = mem_len;
            // prepare memory for recv pull response from remote
            msg.meta.addr =
                reinterpret_cast<uint64_t>(reg_buffer.find_buffer_or_register(
                    msg.meta.recver, msg.meta.psftype, mem_len));
        }
        // NOTE: all the scalar types (key, version_bound, sender) will be
        // encoded into the first item. We just remove it, as the msg.meta
        // contains all of them
        msg.data.erase(msg.data.begin());
        msg.set_data_num_and_length();
        // there should be only two elements in msg.data
        msg.meta.local_data_addr =
            reinterpret_cast<uint64_t>(msg.data.end()->data());
        msg.meta.local_data_buf_len = -1;
        Postoffice::Get()->van()->Send(msg);
        return timestamp;
    }

    int RequestForPush(
        const PSFData<kPushEmbedding>::Request &request,
        const std::function<void(const PSFData<kPushEmbedding>::Response &)>
            &cb) {
        auto constexpr ftype = kPushEmbedding;
        int timestamp = obj_->NewRequest(kServerGroup);
        CallbackStore<ftype>::Get()->store(timestamp, cb);
        // Find the server
        Key key = get<0>(request);
        int target_server_id = par->queryServer(key);
        // Create message
        Message msg;
        tupleEncode(request, msg.data);
        if (logOut.is_open()) {
            for (auto x : msg.data) {
                loads[ftype].first += x.size();
            }
        }
        msg.meta.app_id = obj_->app_id();
        msg.meta.customer_id = obj_->customer_id();
        msg.meta.timestamp = timestamp;
        msg.meta.sender = Postoffice::Get()->van()->my_node().id;
        msg.meta.recver = Postoffice::Get()->ServerRankToID(target_server_id);
        msg.meta.psftype = ftype;
        msg.meta.request = true;
        msg.meta.push = true;
        msg.meta.key = key;
        // remove the key info from msg.data
        msg.data.erase(msg.data.begin());
        msg.set_data_num_and_length();
        msg.meta.local_data_addr =
            reinterpret_cast<uint64_t>(msg.data.end()->data());
        msg.meta.local_data_buf_len = -1;
        Postoffice::Get()->van()->Send(msg);
        return timestamp;
    }

    int RequestForPushSync(
        const PSFData<kPushSyncEmbedding>::Request &request,
        const std::function<void(const PSFData<kPushSyncEmbedding>::Response &)>
            &cb) {
        auto constexpr ftype = kPushSyncEmbedding;
        int timestamp = obj_->NewRequest(kServerGroup);
        CallbackStore<ftype>::Get()->store(timestamp, cb);
        // Find the server
        Key key = get<0>(request);
        int target_server_id = par->queryServer(key);
        // Create message
        Message msg;
        tupleEncode(request, msg.data);
        if (logOut.is_open()) {
            for (auto x : msg.data) {
                loads[ftype].first += x.size();
            }
        }
        PS_VLOG(3) << "Worker sends requests: " << ftype;
        msg.meta.app_id = obj_->app_id();
        msg.meta.customer_id = obj_->customer_id();
        msg.meta.timestamp = timestamp;
        msg.meta.sender = Postoffice::Get()->van()->my_node().id;
        msg.meta.recver = Postoffice::Get()->ServerRankToID(target_server_id);
        msg.meta.psftype = ftype;
        msg.meta.request = true;
        // actually this message is also a pull request
        msg.meta.push = true;
        msg.meta.key = key;
        // msg.cache_lines.CopyFrom(get<1>(request));
        if (ftype == kPushSyncEmbedding) {
            if (table_width.find(key) == table_width.end()) {
                LOG(FATAL) << "Table width should be recorded";
            }
            size_t width = table_width[key];
            auto &keys = get<1>(request);
            size_t value_set_size = keys.size() * width;
            size_t mem_len = keys.size() * 2 + value_set_size;
            mem_len = mem_len * sizeof(uint64_t) * MEM_BUF_SCALE;
            msg.meta.val_len = mem_len;
            // prepare memory for recv pull response from remote
            msg.meta.addr =
                reinterpret_cast<uint64_t>(mem_pool.getMemBuf(ftype, mem_len));
            memset(reinterpret_cast<void *>(msg.meta.addr), 0, mem_len);
            PS_VLOG(3) << "Worker allocated memory for pull response: "
                       << msg.meta.addr << " len: " << mem_len;
        }
        msg.set_data_num_and_length();
        PackDataToMemBuf(msg);
        Postoffice::Get()->van()->Send(msg);
        return timestamp;
    }

private:
    template <PsfType ftype>
    void onReceive(const Message &msg, int tid) {
        typename PSFData<ftype>::Response response;
        if (logOut.is_open()) {
            for (auto x : msg.data) {
                loads[ftype].second += x.size();
            }
        }
        tupleDecode(response, msg.data);
        int timestamp = msg.meta.timestamp;
        CallbackStore<ftype>::Get()->run(timestamp, response);
    }

    bool try_lookup_and_insert(Key key, size_t value) {
        static std::mutex init_mtx;
        std::lock_guard<std::mutex> lock(init_mtx);
        if (table_width.find(key) != table_width.end())
            return false;
        else {
            table_width[key] = value;
            return true;
        }
    }

    // store width of each cache table
    typedef threadsafe_unordered_map<Key, size_t> width_t;
    width_t table_width;

    template <PsfType, typename>
    friend struct KVAppRegisterHelper;
    std::unordered_map<PsfType, std::pair<long long, long long>> loads;
    std::ofstream logOut;
    SimpleMemPool mem_pool;
    RegisteredBuffer reg_buffer;
};

} // namespace ps
