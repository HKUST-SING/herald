// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef PS_RDMA_TRANSPORT_H_
#define PS_RDMA_TRANSPORT_H_

#include "common/logging.h"
#include "common/sarray.h"
#include "ps/psf/PSFunc.h"
#include <asm-generic/errno.h>
#include <cstdint>
#include <infiniband/verbs.h>
#ifdef DMLC_USE_RDMA

#include "rdma_utils.h"

#define MAX_NUM_SAGE 10

namespace ps {

class Postoffice;
class Transport;

struct Endpoint {
    enum ConnectionStatus { IDLE, CONNECTING, CONNECTED, REJECTED };

    ConnectionStatus status;
    int node_id;
    std::condition_variable cv;
    std::mutex connect_mu;
    struct rdma_cm_id *cm_id;
    std::shared_ptr<Transport> trans;

    int kStartDepth = 256;
    int kRxDepth = 256;
    int kReplyDepth = kRxDepth;
    WRContext *rx_ctx;
    WRContext *start_ctx;
    WRContext *reply_ctx;

    ThreadsafeQueue<WRContext *> free_start_ctx;
    ThreadsafeQueue<WRContext *> free_reply_ctx;

    bool inited = false;

    Endpoint() : status(IDLE), node_id(Node::kEmpty), cm_id(nullptr), rx_ctx() {
        auto byteps_rx_depth = Environment::Get()->find("BYTEPS_RDMA_RX_DEPTH");
        auto byteps_start_depth =
            Environment::Get()->find("BYTEPS_RDMA_START_DEPTH");
        const char *role_val =
            CHECK_NOTNULL(Environment::Get()->find("DMLC_ROLE"));
        std::string role_str(role_val);
        // for joint mode with large number of workers, the default value of
        // rx/tx depth is reduced for less memory consumption.
        if (role_str == "scheduler") {
            kStartDepth = 256;
            kRxDepth = 256 * 1;
        }
        kStartDepth =
            byteps_start_depth ? atoi(byteps_start_depth) : kStartDepth;
        kRxDepth = byteps_rx_depth ? atoi(byteps_rx_depth) : kRxDepth;
        kReplyDepth = kRxDepth;

        start_ctx = new WRContext[kStartDepth];
        reply_ctx = new WRContext[kReplyDepth];
        rx_ctx = new WRContext[kRxDepth];
    }

    ~Endpoint() {
        for (int i = 0; i < kRxDepth; ++i) {
            if (!(rx_ctx[i].buffer)) {
                continue;
            }
            free(rx_ctx[i].buffer->addr);
            CHECK_EQ(ibv_dereg_mr(rx_ctx[i].buffer), 0);
        }

        for (int i = 0; i < kStartDepth; ++i) {
            if (start_ctx[i].buffer) {
                free(start_ctx[i].buffer->addr);
                CHECK_EQ(ibv_dereg_mr(start_ctx[i].buffer), 0);
            }
        }

        for (int i = 0; i < kReplyDepth; ++i) {
            if (reply_ctx[i].buffer) {
                free(reply_ctx[i].buffer->addr);
                CHECK_EQ(ibv_dereg_mr(reply_ctx[i].buffer), 0);
            }
        }

        rdma_destroy_qp(cm_id);
        CHECK_EQ(rdma_destroy_id(cm_id), 0) << strerror(errno);
    }

    void SetTransport(std::shared_ptr<Transport> t) {
        trans = t;
    }

    std::shared_ptr<Transport> GetTransport() {
        return trans;
    }

    void Disconnect() {
        std::unique_lock<std::mutex> lk(connect_mu);
        CHECK_EQ(rdma_disconnect(cm_id), 0) << strerror(errno);
        cv.wait(lk, [this] { return status == IDLE; });
        trans.reset();
    }

    void SetNodeID(int id) {
        node_id = id;
    }

    void InitSendContextHelper(struct ibv_pd *pd, WRContext *ctx,
                               ThreadsafeQueue<WRContext *> *queue, size_t num,
                               WRContextType type) {
        for (size_t i = 0; i < num; ++i) {
            void *buf;
            aligned_malloc((void **)&buf, kMempoolChunkSize);
            CHECK(buf);
            struct ibv_mr *mr = ibv_reg_mr(pd, buf, kMempoolChunkSize, 0);
            CHECK(mr)
                << "ibv_reg_mr failed: " << strerror(errno)
                << "\nYou can try to reduce BYTEPS_RDMA_START_DEPTH (current "
                << kStartDepth << ") or BYTEPS_RDMA_RX_DEPTH (current "
                << kRxDepth << ").";

            ctx[i].type = type;
            ctx[i].buffer = mr;
            ctx[i].private_data = this;
            queue->Push(&ctx[i]);
        }
    }

    void Init(struct ibv_cq *cq, struct ibv_pd *pd) {
        struct ibv_qp_init_attr attr;
        memset(&attr, 0, sizeof(ibv_qp_init_attr));
        attr.send_cq = cq;
        attr.recv_cq = cq;
        attr.cap.max_send_wr = kStartDepth + kReplyDepth;
        attr.cap.max_recv_wr = kRxDepth;
        attr.cap.max_send_sge = kSGEntry;
        attr.cap.max_recv_sge = kSGEntry;
        attr.qp_type = IBV_QPT_RC;
        attr.sq_sig_all = 0;

        CHECK_EQ(rdma_create_qp(cm_id, pd, &attr), 0)
            << "Create RDMA queue pair failed: " << strerror(errno);

        if (inited == false) {
            InitSendContextHelper(pd, start_ctx, &free_start_ctx, kStartDepth,
                                  kRendezvousStartContext);
            InitSendContextHelper(pd, reply_ctx, &free_reply_ctx, kReplyDepth,
                                  kRendezvousReplyContext);
        }

        for (int i = 0; i < kRxDepth; ++i) {
            if (inited == false) {
                void *buf;
                aligned_malloc((void **)&buf, kMempoolChunkSize);
                CHECK(buf);
                struct ibv_mr *mr = ibv_reg_mr(pd, buf, kMempoolChunkSize,
                                               IBV_ACCESS_LOCAL_WRITE);
                CHECK(mr)
                    << "ibv_reg_mr failed: " << strerror(errno)
                    << "\nYou can try to reduce BYTEPS_RDMA_START_DEPTH (default 128)"
                    << " or BYTEPS_RDMA_RX_DEPTH (default 2048)";

                rx_ctx[i].type = kReceiveContext;
                rx_ctx[i].buffer = mr;
                rx_ctx[i].private_data = this;
            }

            PostRecv(&rx_ctx[i]);
        }
        inited = true;
    }

    void PostRecv(WRContext *ctx) {
        struct ibv_recv_wr wr, *bad_wr = nullptr;
        memset(&wr, 0, sizeof(wr));

        struct ibv_sge sge;
        sge.addr = reinterpret_cast<uint64_t>(ctx->buffer->addr);
        sge.length = kMempoolChunkSize;
        sge.lkey = ctx->buffer->lkey;

        wr.wr_id = reinterpret_cast<uint64_t>(ctx);
        wr.next = nullptr;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        CHECK_EQ(ibv_post_recv(cm_id->qp, &wr, &bad_wr), 0)
            << "ibv_post_recv failed.";
    }
};

class Transport {
public:
    virtual void RDMAWriteWithImm(MessageBuffer *msg_buf, uint64_t remote_addr,
                                  uint32_t rkey, uint32_t idx) = 0;

    virtual int RecvPushRequest(Message *msg, BufferContext *buffer_ctx,
                                int meta_len) = 0;
    virtual int RecvPullRequest(Message *msg, BufferContext *buffer_ctx,
                                int meta_len) = 0;
    virtual int RecvPushResponse(Message *msg, BufferContext *buffer_ctx,
                                 int meta_len) = 0;
    virtual int RecvPullResponse(Message *msg, BufferContext *buffer_ctx,
                                 int meta_len) = 0;

    virtual void Send(Message &msg, MessageBuffer *msg_buf,
                      RemoteTuple remote_tuple) = 0;
    virtual void SendPullRequest(Message &msg, MessageBuffer *msg_buf,
                                 RemoteTuple remote_tuple) = 0;
    virtual void SendPushRequest(Message &msg, MessageBuffer *msg_buf,
                                 RemoteTuple remote_tuple) = 0;
    virtual void SendPushResponse(Message &msg, MessageBuffer *msg_buf,
                                  RemoteTuple remote_tuple) = 0;
    virtual void SendPullResponse(Message &msg, MessageBuffer *msg_buf,
                                  RemoteTuple remote_tuple, size_t lkey) = 0;
    virtual void SendRendezvousBegin(Message &msg, MessageBuffer *msg_buf) = 0;
    virtual void SendRendezvousReply(RendezvousStart *req,
                                     AddressPool<BufferContext> &pool) = 0;

    virtual SArray<char> CreateFunctionalSarray(void *value, size_t size) = 0;

}; // class Transport

class RDMATransport : public Transport {
public:
    explicit RDMATransport(Endpoint *endpoint, MemoryAllocator *allocator,
                           Postoffice *postoffice) {
        endpoint_ = CHECK_NOTNULL(endpoint);
        allocator_ = CHECK_NOTNULL(allocator);
        pagesize_ = sysconf(_SC_PAGESIZE);

        postoffice_ = postoffice;
        is_server_ = postoffice_->is_server();
    };

    ~RDMATransport(){};

    virtual void RDMAWriteWithImm(MessageBuffer *msg_buf, uint64_t remote_addr,
                                  uint32_t rkey, uint32_t idx) {
        struct ibv_sge sge;
        sge.addr = reinterpret_cast<uint64_t>(msg_buf->inline_buf);
        sge.length = msg_buf->inline_len;
        sge.lkey = allocator_->LocalKey(msg_buf->inline_buf);

        struct ibv_send_wr wr, *bad_wr = nullptr;
        memset(&wr, 0, sizeof(wr));
        wr.wr_id = reinterpret_cast<uint64_t>(msg_buf);
        wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        wr.next = nullptr;
        wr.imm_data = idx;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.wr.rdma.remote_addr = remote_addr;
        wr.wr.rdma.rkey = rkey;

        CHECK_EQ(ibv_post_send(endpoint_->cm_id->qp, &wr, &bad_wr), 0)
            << "ibv_post_send failed.";
    }

    void SendRendezvousBegin(Message &msg, MessageBuffer *msg_buf) {
        WRContext *context = nullptr;
        endpoint_->free_start_ctx.WaitAndPop(&context);

        RendezvousStart *req =
            reinterpret_cast<RendezvousStart *>(context->buffer->addr);
        req->meta_len = msg_buf->inline_len;
        req->origin_addr = reinterpret_cast<uint64_t>(msg_buf);
        req->data_num = msg_buf->data.size();
        int total_data_len = 0;
        for (size_t i = 0; i < req->data_num; ++i) {
            req->data_len[i] = msg.data[i].size();
            total_data_len += req->data_len[i];
        }
        PS_VLOG(3) << "SendRendezvousBegin for message " << msg.meta.psftype
                   << ", meta_len=" << req->meta_len
                   << ", data_num=" << req->data_num
                   << ", total_data_len=" << total_data_len;

        struct ibv_sge sge;
        sge.addr = reinterpret_cast<uint64_t>(req);
        sge.lkey = context->buffer->lkey;
        sge.length = sizeof(RendezvousStart);

        struct ibv_send_wr wr, *bad_wr = nullptr;
        memset(&wr, 0, sizeof(wr));
        wr.wr_id = reinterpret_cast<uint64_t>(context);
        wr.opcode = IBV_WR_SEND_WITH_IMM;
        wr.next = nullptr;
        wr.imm_data = kRendezvousStart;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        CHECK_EQ(ibv_post_send(endpoint_->cm_id->qp, &wr, &bad_wr), 0)
            << strerror(errno);
    }

    void SendRendezvousReply(RendezvousStart *req,
                             AddressPool<BufferContext> &addrpool) {
        BufferContext *buf_ctx = new BufferContext();
        buf_ctx->meta_len = req->meta_len;
        buf_ctx->data_num = req->data_num;

        size_t data_len = 0;
        for (size_t i = 0; i < req->data_num; ++i) {
            buf_ctx->data_len[i] = req->data_len[i];
            buf_ctx->total_data_len += req->data_len[i];
            data_len += req->data_len[i];
        }

        // worker only needs a buffer for receving meta
        // other data message would be sent via dedicated memory space
        // server needs a buffer for both meta and data and we wants to reuse it
        // as much as possible, so we allocate a large buffer
        auto buf_size =
            is_server_ ? req->meta_len + data_len * 300 : req->meta_len;
        // is_server_ ? align_ceil(req->meta_len, pagesize_) + data_len * 100 :
        //              req->meta_len;
        // auto buf_size = req->meta_len;
        char *buffer = allocator_->Alloc(buf_size);

        CHECK(buffer);
        buf_ctx->buffer = buffer;

        WRContext *reply_ctx = nullptr;
        endpoint_->free_reply_ctx.WaitAndPop(&reply_ctx);

        RendezvousReply *resp =
            reinterpret_cast<RendezvousReply *>(reply_ctx->buffer->addr);

        resp->addr = reinterpret_cast<uint64_t>(buffer);
        resp->rkey = allocator_->RemoteKey(buffer);
        resp->origin_addr = req->origin_addr;
        resp->idx = addrpool.StoreAddress(buf_ctx);
        resp->addr_len = buf_size;
        PS_VLOG(3) << "RendezvousReply: Allocated buffer at "
                   << reinterpret_cast<uint64_t>(buffer) << " of size "
                   << buf_size << ", rkey is " << resp->rkey;
        struct ibv_sge sge;
        sge.addr = reinterpret_cast<uint64_t>(resp);
        sge.length = sizeof(RendezvousReply);
        sge.lkey = reply_ctx->buffer->lkey;

        struct ibv_send_wr wr, *bad_wr = nullptr;
        memset(&wr, 0, sizeof(wr));

        wr.wr_id = reinterpret_cast<uint64_t>(reply_ctx);
        wr.opcode = IBV_WR_SEND_WITH_IMM;
        wr.next = nullptr;

        wr.imm_data = kRendezvousReply;

        wr.send_flags = IBV_SEND_SIGNALED;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        PS_VLOG(3) << "SendRendezvousReply";

        CHECK_EQ(ibv_post_send(endpoint_->cm_id->qp, &wr, &bad_wr), 0)
            << "ibv_post_send failed.";
    }

    void Send(Message &msg, MessageBuffer *msg_buf, RemoteTuple remote_tuple) {
        auto raddr = std::get<0>(remote_tuple);
        auto rkey = std::get<1>(remote_tuple);
        auto idx = std::get<2>(remote_tuple);

        RDMAWriteWithImm(msg_buf, raddr, rkey, idx);
    }

    void SendMsgData(Message &msg, MessageBuffer *msg_buf, uint64_t raddr,
                     uint32_t rkey, bool is_pull_resp = false) {
        size_t num_send_sge = msg.data.size();

        struct ibv_sge my_sge[MAX_NUM_SAGE];

        int sge_idx = 0;
        if (msg.meta.local_data_addr) {
            my_sge[sge_idx].addr = msg.meta.local_data_addr;
            my_sge[sge_idx].length = msg.meta.data_size;
            my_sge[sge_idx].lkey = msg_buf->mrs[0].first->lkey;
            PS_VLOG(3) << "Prepare one large sge " << sge_idx << " for data "
                       << my_sge[sge_idx].addr << " " << my_sge[sge_idx].length
                       << " " << my_sge[sge_idx].lkey;
            sge_idx++;
        } else {
            CHECK_EQ(num_send_sge, msg_buf->mrs.size());
            for (size_t i = 0; i < num_send_sge; ++i) {
                CHECK(msg_buf->data[i].size() == msg_buf->mrs[i].second);
                if (msg_buf->data[i].size() == 0) {
                    PS_VLOG(3) << "Skip empty data " << i;
                    continue;
                }
                CHECK_NOTNULL(msg_buf->mrs[i].first);
                my_sge[sge_idx].addr =
                    reinterpret_cast<uint64_t>(msg_buf->mrs[i].first->addr);
                my_sge[sge_idx].length = msg.data[i].size();
                PS_VLOG(3) << "Prepare sge " << sge_idx << " for data " << i
                           << " "
                           << reinterpret_cast<uint64_t>(my_sge[sge_idx].addr)
                           << " " << my_sge[sge_idx].length;
                my_sge[sge_idx].lkey = msg_buf->mrs[i].first->lkey;
                sge_idx++;
            }
        }
        // this rdma-write will not trigger any signal both remotely and locally
        struct ibv_send_wr wr, *bad_wr = nullptr;
        memset(&wr, 0, sizeof(wr));
        wr.wr_id = 0;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.next = nullptr;
        wr.sg_list = my_sge;
        wr.num_sge = sge_idx;
        wr.wr.rdma.rkey = rkey;

        if (!msg.meta.request) {
            // server sends inline message to client
            // wr.send_flags = IBV_SEND_INLINE;
        }

        // write to the next page-aligned address (remote_addr should already be
        // aligned)
        if (!is_pull_resp) {
            wr.wr.rdma.remote_addr = raddr + msg_buf->inline_len;
            // raddr + align_ceil(msg_buf->inline_len, pagesize_);
        } else {
            wr.wr.rdma.remote_addr = raddr;
        }

        PS_VLOG(3) << "Write message data at " << wr.wr.rdma.remote_addr << " "
                   << "with rkey " << wr.wr.rdma.rkey << " "
                   << "with sge num " << wr.num_sge << " with data num "
                   << msg.data.size();

        int ret = ibv_post_send(endpoint_->cm_id->qp, &wr, &bad_wr);
        CHECK_EQ(ret, 0) << "ibv_post_send failed: " << ret;
    }

    int RecvMsgData(Message *msg, BufferContext *buffer_ctx, char *cur,
                    bool release_mem = false) {
        int recv_len = 0;

        // we use the info in meta to unpack the data
        auto data_num = msg->meta.request ? msg->meta.req_data_num :
                                            msg->meta.resp_data_num;
        auto each_data_len = msg->meta.request ? msg->meta.req_data_len :
                                                 msg->meta.resp_data_len;

        // restore the data from end to begin
        for (size_t i = 0; i < data_num; i++) {
            SArray<char> item;
            auto len = each_data_len[i];
            if (msg->meta.psftype == kPushEmbedding)
                len = each_data_len[data_num - i - 1];
            // for pull response, the first data is the scalar value
            if (i > 0 && !msg->meta.request
                && msg->meta.psftype == kSyncEmbedding)
                len = each_data_len[data_num - i];
            // len should be equal to the size of the data
            if (len == 0) {
                PS_VLOG(3) << "Recv skip empty data " << i;
                item.clear();
                msg->data.push_back(item);
                continue;
            }
            item.reset(cur, len, [](char *) {});
            PS_VLOG(3) << "Unpack message: " << i << " " << len << " "
                       << item.DebugString();
            // SArray<char> copyed;
            // copyed.CopyFrom(item);
            if (msg->meta.psftype == ParamInit) {
                msg->data.push_back(item);
            } else {
                auto pos = msg->data.begin();
                // only restore the data for tensor data (start from the second)
                if (!msg->meta.request && msg->meta.psftype == kSyncEmbedding
                    && i > 0)
                    pos = msg->data.begin() + 1;
                msg->data.insert(pos, item);
            }
            cur += len;
            recv_len += len;
        }

        // the buffer_ctx is re-used, so we doesn't need to check the recv_len

        // CHECK_EQ(recv_len, buffer_ctx->total_data_len)
        //     << "Incorrect total data size: " << recv_len << " "
        //     << buffer_ctx->total_data_len;
        return recv_len;
    }

    void SendPushRequest(Message &msg, MessageBuffer *msg_buf,
                         RemoteTuple remote_tuple) {
        // PS_VLOG(3) << "Begin to send Push request metadata";
        auto raddr = std::get<0>(remote_tuple);
        auto rkey = std::get<1>(remote_tuple);
        auto idx = std::get<2>(remote_tuple);

        // send the data
        SendMsgData(msg, msg_buf, raddr, rkey);

        // send the meta data
        RDMAWriteWithImm(msg_buf, raddr, rkey, idx);
    }

    void SendPullRequest(Message &msg, MessageBuffer *msg_buf,
                         RemoteTuple remote_tuple) {
        auto raddr = std::get<0>(remote_tuple);
        auto rkey = std::get<1>(remote_tuple);
        auto idx = std::get<2>(remote_tuple);
        SendMsgData(msg, msg_buf, raddr, rkey);
        RDMAWriteWithImm(msg_buf, raddr, rkey, idx);
    }

    virtual void SendPushResponse(Message &msg, MessageBuffer *msg_buf,
                                  RemoteTuple remote_tuple) {
        CHECK_EQ(msg_buf->mrs.size(), 0);
        Send(msg, msg_buf, remote_tuple);
    }

    virtual void SendPullResponse(Message &msg, MessageBuffer *msg_buf,
                                  RemoteTuple remote_tuple, size_t lkey) {
        auto raddr = msg.meta.addr;
        auto rkey = msg.meta.option;
        CHECK_GE((size_t)msg.meta.val_len, msg.meta.data_size);

        PS_VLOG(3) << "Send Pull response: " << msg.meta.val_len << " "
                   << msg.meta.data_size << " " << msg.meta.addr << " "
                   << msg.meta.option;

        // send data
        SendMsgData(msg, msg_buf, raddr, rkey, true);

        // send meta data
        Send(msg, msg_buf, remote_tuple);
    }

    virtual int RecvPushResponse(Message *msg, BufferContext *buffer_ctx,
                                 int meta_len) {
        // Xudong: we do not need to check here
        // CHECK_EQ(buffer_ctx->data_num, 0);
        PS_VLOG(3) << "Recv Push responce, do nothing";
        return 0;
    }

    virtual int RecvPullRequest(Message *msg, BufferContext *buffer_ctx,
                                int meta_len) {
        char *cur = buffer_ctx->buffer + meta_len;
        // buffer_ctx->buffer + align_ceil((size_t)meta_len, pagesize_);

        int recv_len = RecvMsgData(msg, buffer_ctx, cur);

        if (msg->meta.psftype == kSyncEmbedding) {
            // restores the key, version bound, sender id into msg->data
            // following the recursive order
            SArray<char> vals = CreateFunctionalSarray(
                &msg->meta.sender, sizeof(msg->meta.sender));
            msg->data.insert(msg->data.begin(), vals);
            // restore the version bound
            size_t cur_size = msg->data[0].size();
            msg->data[0].resize(cur_size + sizeof(msg->meta.version_bound));
            uint64_t *ptr =
                reinterpret_cast<uint64_t *>(msg->data[0].data() + cur_size);
            *ptr = msg->meta.version_bound;
            // restore the key
            cur_size = msg->data[0].size();
            msg->data[0].resize(cur_size + sizeof(Key));
            auto id_ptr =
                reinterpret_cast<Key *>(msg->data[0].data() + cur_size);
            *id_ptr = msg->meta.key;
        }

        return recv_len;
    }

    virtual int RecvPushRequest(Message *msg, BufferContext *buffer_ctx,
                                int meta_len) {
        char *cur = buffer_ctx->buffer + meta_len;
        // buffer_ctx->buffer + align_ceil((size_t)meta_len, pagesize_);

        int recv_len = RecvMsgData(msg, buffer_ctx, cur);
        if (msg->meta.psftype == kPushEmbedding) {
            SArray<char> keys =
                CreateFunctionalSarray(&msg->meta.key, sizeof(Key));
            msg->data.insert(msg->data.begin(), keys);
        }
        return recv_len;
    }

    virtual int RecvPullResponse(Message *msg, BufferContext *buffer_ctx,
                                 int meta_len) {
        char *cur = reinterpret_cast<char *>(msg->meta.addr);

        int recv_len = RecvMsgData(msg, buffer_ctx, cur);

        PS_VLOG(3) << "Recv Pull respnse: data num: " << msg->data.size() << " "
                   << "data len: " << recv_len;
        return recv_len;
    }

    SArray<char> CreateFunctionalSarray(void *value, size_t size) {
        SArray<char> sarr;
        void *p = malloc(size);
        memcpy(p, value, size);
        sarr.reset((char *)p, size, [p](void *) { free(p); });
        return sarr;
    }

protected:
    size_t pagesize_ = 4096;
    Endpoint *endpoint_;
    MemoryAllocator *allocator_;
    bool is_server_;

    Postoffice *postoffice_;

}; // class Transport

class IPCTransport : public RDMATransport {
public:
    explicit IPCTransport(Endpoint *endpoint, MemoryAllocator *allocator,
                          Postoffice *postoffice) :
        RDMATransport(endpoint, allocator, postoffice) {
        auto val = Environment::Get()->find("BYTEPS_IPC_COPY_NUM_THREADS");
        ipc_copy_nthreads_ = val ? atoi(val) : 4;
        for (int i = 0; i < ipc_copy_nthreads_; ++i) {
            auto q = new ThreadsafeQueue<AsyncCopy>;
            async_copy_queue_.push_back(q);
        }
        for (int i = 0; i < ipc_copy_nthreads_; ++i) {
            auto t = new std::thread(&IPCTransport::AsyncCopyThread, this, i);
            ipc_copy_thread_list_.push_back(t);
        }
        val = Environment::Get()->find("BYTEPS_PARTITION_BYTES");
        byteps_partition_bytes_ = val ? atoi(val) : 4096000;

        val = Environment::Get()->find("BYTEPS_ENCODING_SCHEME_VERSION");
        encoding_scheme_version_ = val ? atoi(val) : 0;

        val = Environment::Get()->find("BYTEPS_LOCAL_SIZE");
        auto byteps_local_size = val ? atoi(val) : 8;
        byteps_partition_bytes_ = RoundUp(
            byteps_partition_bytes_, byteps_local_size * sysconf(_SC_PAGESIZE));

        val = Environment::Get()->find("BYTEPS_IPC_ENABLE_ASYNC_COPY");
        enable_async_copy_ = val ? atoi(val) : 1; // default enabled
        if (!enable_async_copy_)
            LOG(INFO)
                << "Async copy has been disabled, this could affect the performance";

        val = Environment::Get()->find("BYTEPS_PCIE_SWITCH_SIZE");
        auto byteps_nccl_pcie_size = val ? atoi(val) : 8;
        if (byteps_local_size % byteps_nccl_pcie_size != 0) {
            // local_size < pcie_size or unbalance PCIe switches
            byteps_nccl_pcie_size = byteps_local_size;
        }
        // ensure this name corresponds with that in
        // BytePSSharedMemory::openPcieSharedMemory()
        if (byteps_local_size > byteps_nccl_pcie_size) {
            // cross pcie switch, use the last pcie cpu buffer
            auto byteps_pcie_num = byteps_local_size / byteps_nccl_pcie_size;
            shm_prefix_ =
                kShmPciePrefix + std::to_string(byteps_pcie_num - 1) + "_Shm_";
        } else {
            shm_prefix_ = kShmPrefix;
        }
        val = Environment::Get()->find("BYTEPS_JOB_ID");
        std::string _job_id = val ? std::string(val) : "0";
        shm_prefix_ = shm_prefix_ + _job_id + "_";
    };

    ~IPCTransport() {
        for (size_t i = 0; i < ipc_copy_thread_list_.size(); ++i) {
            AsyncCopy m;
            m.shutdown = true;
            async_copy_queue_[i]->Push(m);
            ipc_copy_thread_list_[i]->join();
        }
    }

    void SendPushRequest(Message &msg, MessageBuffer *msg_buf,
                         RemoteTuple remote_tuple) {
        Send(msg, msg_buf, remote_tuple);
    }

    void SendPullResponse(Message &msg, MessageBuffer *msg_buf,
                          RemoteTuple remote_tuple, size_t lkey) {
        auto addr = (void *)CHECK_NOTNULL(msg.data[1].data());
        void *shm_addr =
            CHECK_NOTNULL(GetSharedMemory(shm_prefix_, msg.meta.key));

        if (enable_async_copy_) {
            // async copy with a simple load-balancing strategy
            AsyncCopy m = {msg_buf, remote_tuple,     shm_addr,
                           addr,    msg.meta.val_len, false};
            auto cnt = cpy_counter_.fetch_add(1);
            async_copy_queue_[cnt % ipc_copy_nthreads_]->Push(m);
        } else {
            // synchronous copy
            memcpy(shm_addr, addr, msg.meta.val_len);
            Send(msg, msg_buf, remote_tuple);
        }
    }

    int RecvPushRequest(Message *msg, BufferContext *buffer_ctx, int meta_len) {
        // get data message from local shared memory
        auto key = msg->meta.key;
        auto len = msg->meta.val_len;

        SArray<char> keys = CreateFunctionalSarray(&msg->meta.key, sizeof(Key));

        SArray<char> vals;
        void *addr = GetSharedMemory(shm_prefix_, key);
        vals.reset(reinterpret_cast<char *>(addr), len, [](void *) {});

        SArray<char> lens =
            CreateFunctionalSarray(&msg->meta.val_len, sizeof(int));

        msg->data.push_back(keys);
        msg->data.push_back(vals);
        msg->data.push_back(lens);

        return keys.size() + vals.size() + lens.size();
    }

private:
    struct AsyncCopy {
        MessageBuffer *msg_buf;
        RemoteTuple remote_tuple;
        void *dst;
        void *src;
        int len;
        bool shutdown;
    };

    void AsyncCopyThread(int i) {
        auto &q = async_copy_queue_[i];
        while (true) {
            AsyncCopy m;
            q->WaitAndPop(&m);
            if (m.shutdown)
                break;
            if (m.len == 0)
                continue;

            // TODO: use parallel copy
            CHECK(m.dst);
            CHECK(m.src);
            memcpy(m.dst, m.src, m.len);

            auto raddr = std::get<0>(m.remote_tuple);
            auto rkey = std::get<1>(m.remote_tuple);
            auto idx = std::get<2>(m.remote_tuple);

            RDMAWriteWithImm(m.msg_buf, raddr, rkey, idx);
        }
    }

    void *GetSharedMemory(const std::string &prefix, uint64_t key) {
        std::lock_guard<std::mutex> lock(shm_mu_);
        auto worker_key = DecodeWorkerKey(key);
        auto seq_num = worker_key % (1 << 16);
        // Total key space is [0, 2^64 - 1]
        // It will be divided to N PS servers, for now we assume N <= 2^16
        // Then we have 2^48 key space left.
        // Encoding scheme version 0:
        //   Then we have 2^48 key space left (top 16 bits for different
        //   servers) MXNet server has a bug dealing with keys larger than 2^32
        //   Below we support up to 2^16 tensors, and up to 2^16 partitions per
        //   tensor
        // Encoding scheme version 1:
        //   Top 16 bits out of the 48 bits encodes the sender rank
        //   Mid 16 bits out of the 48 bits encodes the tensor id
        //   The next 6 bits encodes request types (pushpull, send, etc)
        //   The last 10 bits encodes the partition id
        //   Therefore, we support up to 2^16 tensors, and up to 2^10 partitions
        //   per tensor
        if (encoding_scheme_version_ == 1) {
            seq_num = worker_key % (1 << 10);
        }
        auto base_key = worker_key - seq_num;
        uint64_t offset = byteps_partition_bytes_ * seq_num;
        if (key_shm_addr_.find(base_key) != key_shm_addr_.end()) {
            return (void *)((char *)key_shm_addr_[base_key] + offset);
        }
        std::string shm_name(prefix);
        std::stringstream stream;
        stream << std::hex << base_key;

        shm_name += stream.str();
        int shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
        CHECK_GE(shm_fd, 0)
            << "shm_open failed for " << shm_name << ", " << strerror(errno);

        struct stat sb;
        CHECK_EQ(0, fstat(shm_fd, &sb)) << strerror(errno);
        auto total_shm_size = sb.st_size;

        void *base_ptr = mmap(0, total_shm_size, PROT_READ | PROT_WRITE,
                              MAP_SHARED, shm_fd, 0);
        CHECK_NE(base_ptr, (void *)-1) << strerror(errno);
        key_shm_addr_[base_key] = base_ptr;

        PS_VLOG(1) << "open Shared Memory: " << shm_name << " offset=" << offset
                   << " (in bytes) size=" << total_shm_size;
        return (void *)((char *)key_shm_addr_[base_key] + offset);
    }

    int ipc_copy_nthreads_;
    std::vector<std::thread *> ipc_copy_thread_list_;
    std::vector<ThreadsafeQueue<AsyncCopy> *> async_copy_queue_;
    std::atomic<unsigned long long> cpy_counter_{0};

    int byteps_partition_bytes_ = 4096000;

    std::string shm_prefix_;

    std::mutex shm_mu_;
    std::unordered_map<uint64_t, void *> key_shm_addr_;

    bool enable_async_copy_;
    int encoding_scheme_version_ = 0;
}; // class IPCTransport

}; // namespace ps

#endif // DMLC_USE_RDMA
#endif // PS_RDMA_VAN_H_
