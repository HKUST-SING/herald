#pragma once

#include "ps/internal/postoffice.h"
#include "ps/internal/customer.h"
#include "ps/internal/message.h"
#include <chrono>

namespace ps {

// Recursively register receive message handler (from 0 to kNumPSfunction)
template <PsfType ftype, typename app>
struct KVAppRegisterHelper {
    static void init(app *ptr) {
        ptr->message_handlers[ftype] = std::bind(
            &app::template onReceive<ftype>, ptr, std::placeholders::_1, std::placeholders::_2);
        KVAppRegisterHelper<PsfType(ftype + 1), app>::init(ptr);
    }
};

template <typename app>
struct KVAppRegisterHelper<kNumPSfunction, app> {
    static void init(app *ptr) {
    }
};

class KVApp {
public:
    explicit KVApp(int app_id) {
        obj_.reset(new Customer(
            app_id, app_id,
            std::bind(&KVApp::Process, this, std::placeholders::_1, std::placeholders::_2)));
    }
    std::unique_ptr<Customer> obj_;

private:
    void Process(const Message &msg, int tid) {
        CHECK_LT(msg.meta.psftype, kNumPSfunction)
            << "Unknown PS Function Received";
        // RDMA van may receive the message before the handler is registered
        while (message_handlers[msg.meta.psftype] == nullptr) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        message_handlers[msg.meta.psftype](msg, tid);
    }

    template <PsfType, typename>
    friend struct KVAppRegisterHelper;

protected:
    typedef std::function<void(const Message &, int tid)> MessageHandle;
    MessageHandle message_handlers[kNumPSfunction];
};

} // namespace ps
