#pragma once

#include "common/logging.h"
#include "common/sarray.h"
#include "ps/internal/mem_pool.h"
#include "ps/internal/postoffice.h"
#include "ps/internal/threadsafe_pqueue.h"
#include "ps/internal/threadsafe_queue.h"
#include "ps/psf/PSFunc.h"
#include "ps/server/PSFHandle.h"
#include "ps/server/ssp_handler.h"
#include "ps/server/preduce_handler.h"
#include "ps/psf/serializer.h"
#include "ps/kvapp.h"
#include <chrono>
#include <cstdint>
#include <memory>
#include <ratio>
#include <sys/types.h>
#include <unordered_map>
#include <vector>
#include <utility>
#include <queue>
namespace ps {

static const size_t kDefaultMemBufSize = 1 << 24;

template <PsfType>
struct KVServerRegisterHelper;

/**
 * \brief A server node for maintaining key-value pairs
 */
class KVServer : public KVApp {
public:
    /**
     * \brief constructor
     * \param app_id the app id, should match with \ref KVWorker's id
     */
    explicit KVServer(int app_id) : KVApp(app_id) {
        // TODO : change this to index_sequence if c++14 is available
        handler_[static_cast<int>(PsfGroup::kParameterServer)] =
            std::make_shared<PSHandler<PsfGroup::kParameterServer>>();
        handler_[static_cast<int>(PsfGroup::kSSPControl)] =
            std::make_shared<PSHandler<PsfGroup::kSSPControl>>();
        handler_[static_cast<int>(PsfGroup::kPReduceScheduler)] =
            std::make_shared<PSHandler<PsfGroup::kPReduceScheduler>>();
        KVAppRegisterHelper<PsfType(0), KVServer>::init(this);
        // now it is saft to register the consumer for receiving message
        // register_consumer(app_id);
    }

private:
    template <PsfType ftype>
    void onReceive(const Message &msg, int tid) {
        typename PSFData<ftype>::Request request;
        typename PSFData<ftype>::Response response;
        tupleDecode(request, msg.data);
        constexpr PsfGroup group = PSFData<ftype>::group;
        auto handler = std::dynamic_pointer_cast<PSHandler<group>>(
            handler_[static_cast<int>(group)]);
        assert(handler);
        handler->serve(request, response);
        PS_VLOG(3) << "Server finished serving for " << msg.meta.psftype;
        Message rmsg;
        rmsg.data.clear();
        tupleEncode(response, rmsg.data);
        // rmsg.meta = msg.meta;
        rmsg.meta.app_id = msg.meta.app_id;
        rmsg.meta.customer_id = msg.meta.customer_id;
        rmsg.meta.timestamp = msg.meta.timestamp;
        rmsg.meta.sender = Postoffice::Get()->van()->my_node().id;
        rmsg.meta.recver = msg.meta.sender;
        rmsg.meta.request = false;
        rmsg.meta.push = msg.meta.push;
        // echo the psf type of request
        rmsg.meta.psftype = msg.meta.psftype;
        // just reuse the key of request
        rmsg.meta.key = msg.meta.key;
        // for pull request
        rmsg.meta.addr = msg.meta.addr;
        rmsg.meta.val_len = msg.meta.val_len;
        rmsg.meta.option = msg.meta.option;
        // echo local addr of request
        rmsg.meta.local_option = msg.meta.local_data_addr;
        // set data size
        rmsg.set_data_num_and_length();
        if (rmsg.meta.psftype == kSyncEmbedding) {
            rmsg.meta.local_data_addr =
                reinterpret_cast<uint64_t>(rmsg.data[3].data());
            rmsg.meta.local_data_buf_len = -1;
        }
        Postoffice::Get()->van()->Send(rmsg);
    }

    /** \brief request handle */
    std::unordered_map<int, std::shared_ptr<PSHandler<PsfGroup::kBaseGroup>>>
        handler_;

    template <PsfType, typename>
    friend struct KVAppRegisterHelper;

    SimpleMemPool mem_pool;
};

} // namespace ps
