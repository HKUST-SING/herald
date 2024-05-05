#ifndef REG_BUFFER_H
#define REG_BUFFER_H

#include <unordered_map>
#include <tuple>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "common/logging.h"
#include "ps/internal/postoffice.h"

class RegisteredBuffer {
public:
    RegisteredBuffer() {
    }

    char *find_buffer_or_register(int node_id, int type, size_t size,
                                  size_t scale_factor = kDefaultScale) {
        if (registered_buffer.find(node_id) == registered_buffer.end()) {
            registered_buffer[node_id] = TypeBuffer();
        }
        auto &node_buffer = registered_buffer[node_id];
        if (node_buffer.find(type) == node_buffer.end()) {
            node_buffer[type] = RegisteredAddr(nullptr, 0);
        }
        auto &type_buffer = node_buffer[type];
        auto &addr = std::get<0>(type_buffer);
        auto &addr_size = std::get<1>(type_buffer);
        if (addr == nullptr || addr_size < size) {
            if (addr != nullptr) {
                // delete[] addr;
                LOG(INFO) << "Free memory for " << type << " at " << addr
                          << " size " << addr_size;
                // free(addr);
                // TODO unpin memory in RDMA
            }
            // addr = new char[size * scale_factor];
            addr = (char*)malloc(size * scale_factor);
            addr_size = size * scale_factor;
            // memset(addr, 0, size * scale_factor);
            // pin memory in RDMA
            ps::Postoffice::Get()->van()->PinMemory(addr, size * scale_factor,
                                                false);
            LOG(INFO) << "Allocate memory for " << type << " at " << addr
                      << " size " << size * scale_factor;
        }
        return addr;
    }

private:
    static const size_t kDefaultScale = 100;
    typedef std::tuple<char *, size_t> RegisteredAddr;
    // request type -> {addr, size}
    typedef std::unordered_map<int, RegisteredAddr> TypeBuffer;
    // node id -> {request type -> {addr, size}}
    std::unordered_map<int, TypeBuffer> registered_buffer;
};

#endif // REG_BUFFER_H