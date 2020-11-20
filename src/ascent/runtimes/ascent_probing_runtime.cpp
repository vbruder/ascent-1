//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-716457
//
// All rights reserved.
//
// This file is part of Ascent.
//
// For details, see: http://ascent.readthedocs.io/.
//
// Please also read ascent/LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: ascent_probing_runtime.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_probing_runtime.hpp"

// hola
#include <ascent_hola.hpp>
#ifdef ASCENT_MPI_ENABLED
#include <ascent_hola_mpi.hpp>
#endif

// standard lib includes
#include <string.h>
#include <cassert>
#include <numeric>
#include <cmath>
#include <valarray>
#include <algorithm>
#include <ostream>
#include <iterator>

#include <thread>
#include <deque>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit_blueprint.hpp>

// mpi related includes
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
// -- conduit relay mpi
#include <conduit_relay_mpi.hpp>
#endif

#if defined(ASCENT_VTKM_ENABLED)
#include <vtkm/cont/Error.h>
#include <vtkm/filter/VectorMagnitude.h>

#include <vtkh/vtkh.hpp>
#include <vtkh/Error.hpp>
#include <vtkh/Logger.hpp>
#include <vtkh/compositing/Image.hpp>
#include <vtkh/compositing/ImageCompositor.hpp>
#include <vtkh/compositing/Compositor.hpp>

#endif // ASCENT_VTKM_ENABLED


using namespace conduit;
using namespace std;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Creation and Destruction
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
ProbingRuntime::ProbingRuntime()
    : Runtime()
{
}

//-----------------------------------------------------------------------------
ProbingRuntime::~ProbingRuntime()
{
    Cleanup();
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//
// Main runtime interface methods called by the ascent interface.
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void ProbingRuntime::Initialize(const conduit::Node &options)
{
#if ASCENT_MPI_ENABLED
    if (!options.has_child("mpi_comm") ||
        !options["mpi_comm"].dtype().is_integer())
    {
        ASCENT_ERROR("Missing Ascent::open options missing MPI communicator (mpi_comm)");
    }
#endif
    // check for probing options (?)

    m_runtime_options = options;
}

//-----------------------------------------------------------------------------
void ProbingRuntime::Info(conduit::Node &out)
{
    out.reset();
    out["runtime/type"] = "probing";
}

//-----------------------------------------------------------------------------
void ProbingRuntime::Cleanup()
{
}

//-----------------------------------------------------------------------------
void ProbingRuntime::Publish(const conduit::Node &data)
{
    Node verify_info;
    bool verify_ok = conduit::blueprint::mesh::verify(data, verify_info);

#if ASCENT_MPI_ENABLED

    MPI_Comm mpi_comm = MPI_Comm_f2c(m_runtime_options["mpi_comm"].to_int());

    // parallel reduce to find if there were any verify errors across mpi tasks
    // use an mpi sum to check if all is ok
    // Node n_src, n_reduce;

    // if (verify_ok)
    //     n_src = (int)0;
    // else
    //     n_src = (int)1;

    // conduit::relay::mpi::sum_all_reduce(n_src,
    //                                     n_reduce,
    //                                     mpi_comm);

    // int num_failures = n_reduce.value();
    // if (num_failures != 0)
    // {
    //     ASCENT_ERROR("Mesh Blueprint Verify failed on "
    //                  << num_failures
    //                  << " MPI Tasks");

    //     // you could use mpi to find out where things went wrong ...
    // }

#else
    if (!verify_ok)
    {
        ASCENT_ERROR("Mesh Blueprint Verify failed!"
                     << std::endl
                     << verify_info.to_json());
    }
#endif

    // create our own tree, with all data zero copied.
    m_data.set_external(data);
}


void debug_break()
{
    volatile int vi = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == vi)
        sleep(5);
}


// structs
//
//
#if ASCENT_MPI_ENABLED
struct MPI_Properties
{
    int size = 0;
    int rank = 0;
    int sim_node_count = 0;
    int vis_node_count = 0;
    MPI_Comm comm_world;
    MPI_Comm comm_vis;
    MPI_Group vis_group;

    MPI_Properties(int size,
                   int rank,
                   int sim_node_count,
                   int vis_node_count,
                   MPI_Comm comm_world,
                   MPI_Comm comm_vis,
                   MPI_Group vis_group)
     : size(size)
     , rank(rank)
     , sim_node_count(sim_node_count)
     , vis_node_count(vis_node_count)
     , comm_world(comm_world)
     , comm_vis(comm_vis)
     , vis_group(vis_group)
    {
    }
};
#else
struct MPI_Properties
{
    int size = 0;
    int rank = 0;
    int sim_node_count = 0;
    int vis_node_count = 0;

    MPI_Properties(int size,
                   int rank,
                   int sim_node_count,
                   int vis_node_count)
     : size(size)
     , rank(rank)
     , sim_node_count(sim_node_count)
     , vis_node_count(vis_node_count)
    {
    }
};
#endif

struct RenderConfig
{
    int max_count = 0;
    double probing_factor = 0.0;
    std::string insitu_type = "hybrid";
    std::string sampling_method = "random";
    int probing_stride = 0;
    int probing_count = 0;
    int non_probing_count = 0;
    int batch_count = 1;
    std::vector<int> probing_ids;

    const static int WIDTH = 800;
    const static int HEIGHT = 800;
    const static int CHANNELS = 4 + 4; // RGBA + depth (float)

    /**
     * Constructor.
     */
    RenderConfig(const int max_render_count, const double probing_factor = 0.0,
                 const std::string &insitu_type = "hybrid", const int batch_count = 1,
                 const std::string &sampling_method = "random")
     : max_count(max_render_count)
     , probing_factor(probing_factor)
     , insitu_type(insitu_type)
     , batch_count(batch_count)
     , sampling_method(sampling_method)
    {
        if (sampling_method == "random")
        {
            probing_count = int(probing_factor * max_count);

            std::srand(42);
            const int range_from  = 0;
            const int range_to    = max_count;
            // std::random_device rand_dev;
            // std::mt19937 generator(rand_dev());
            // std::uniform_int_distribution<int> distr(range_from, range_to);

            probing_ids.resize(probing_count);
            for (int i = 0; i < probing_count; ++i)
            {
                // probing_ids[i] = distr(generator);
                // worse-is-better solution: just a simple random sequence, no fanciness required
                int r = (double(std::rand()) / double(RAND_MAX - 1)) * (range_to - range_from + 1) + range_from;
                probing_ids[i] = r;
            }
            std::sort(probing_ids.begin(), probing_ids.end());

            // std::cout << "random sequence: ";
            // for (auto &a : probing_ids)
            //     std::cout << a << " ";
            // std::cout << std::endl;
        }
        else    // systematic sampling
        {
            // infer probing stride
            if (probing_factor <= 0.0)
                probing_stride = 0;
            else
                probing_stride = std::round(max_count / (probing_factor*max_count));

            for (int i = 0; i*probing_stride < max_count; ++i)
            {
                probing_ids[i] = i*probing_stride;
            }

            probing_count = int(probing_ids.size());
        }

        // infer render count without probing renders
        non_probing_count = max_count - probing_count;
    }

    /**
     * Get the number of probings for a specific part of the rendering sequence.
     */
    int get_probing_count_part(const int render_count, const int render_offset = 0) const
    {
        int probing_count_part = 0;
        for (const int &id : probing_ids)
        {
            if (id >= render_offset && id < render_count + render_offset)
            {
                // std::cout << "id " << id << " | render_count " << render_count << " | render_offset " 
                //           << render_offset << std::endl;
                ++probing_count_part;
            }
        }
        return probing_count_part;
    }

    /**
     * Returns the number of total renders given the number of non-probing renders.
     */
    int get_render_count_from_non_probing(const int non_probing_renders, const int render_offset = 0) const
    {
        int total = non_probing_renders;
        for (const int &id : probing_ids)
        {
            if (id >= render_offset && id < total)
                ++total;
        }
        return total;
    }

    /**
     * Get the next biggest probing id.
     * @returns true if there is a valid bigger probing id, false otherwise.
     */
    bool get_next_probing_id(const int render_id, int &next_probing_id) const
    {
        for (int i = 0; i < probing_ids.size(); ++i)
        {
            if (probing_ids[i] > render_id)
            {   
                next_probing_id = probing_ids[i];
                return true;
            }
        }

        // there is no bigger probing id
        return false;
    }
};


struct RenderBatch
{
    int runs = 0;
    int size = 0;   // size of regular batch
    int rest = 0;   // size of last batch
};


struct NodeConfig
{
    // TODO: move all my_ vars here
    // my_vis_rank
    // my_render_recv_cnt
    // my_data_recv_cnt

    // my_sim_estimate
    // my_probing_times
    // my_avg_probing_time
    // my_data_size
};


/**
 * Assign part of the vis load to the vis nodes.
 */
std::vector<int> load_assignment(const std::vector<float> &sim_estimate,
                                 const std::vector<float> &vis_estimates,
                                 const std::vector<float> &vis_overheads,
                                 const std::vector<int> &node_map,
                                 const RenderConfig render_cfg,
                                 const MPI_Properties mpi_props,
                                 const float skipped_renders)
{
    // optional render factors for sim and/or vis nodes (empirically determined)
    const float sim_factor = 1.0f;
    const float vis_factor = 1.01f;

    assert(sim_estimate.size() == vis_estimates.size());

    std::valarray<float> t_inline(0.f, mpi_props.sim_node_count);
    for (size_t i = 0; i < mpi_props.sim_node_count; i++)
        t_inline[i] = vis_estimates[i] * sim_factor * render_cfg.non_probing_count;

    // compositing time per image determined on stampede2 with 2/10 and 6/33 nodes
    const float t_compose = 0.11f + 0.025f * mpi_props.vis_node_count;
    const float t_compose_skipped = 0.01f * mpi_props.vis_node_count;
    // estimate with average compositing cost
    const float t_compositing = (skipped_renders*t_compose_skipped + (1.f-skipped_renders)*t_compose)
                                 * render_cfg.max_count;
     if (mpi_props.rank == 0)
        std::cout << "=== compositing estimate: " << t_compositing << std::endl;
    // data send overhead
    const float t_send = 1.0f * std::ceil((1.f-skipped_renders) * mpi_props.sim_node_count / mpi_props.vis_node_count);

    std::valarray<float> t_intransit(t_compositing + t_send, mpi_props.vis_node_count);
    std::valarray<float> t_sim(sim_estimate.data(), mpi_props.sim_node_count);

    std::vector<int> render_counts_sim(mpi_props.sim_node_count, 0);
    std::vector<int> render_counts_vis(mpi_props.vis_node_count, 0);

    // initially: push all vis load to vis nodes (=> all intransit case)
    for (size_t i = 0; i < mpi_props.sim_node_count; i++)
    {
        const int target_vis_node = node_map[i];

        t_intransit[target_vis_node] += t_inline[i] * (vis_factor/sim_factor);
        if (t_inline[i] <= 0.f + std::numeric_limits<float>::min())
            t_inline[i] = 0;
        else
            t_inline[i] = vis_overheads[i];
        render_counts_vis[target_vis_node] += render_cfg.non_probing_count;

    }

    if (render_cfg.insitu_type != "intransit")
    {
        // push back load to sim nodes until
        // intransit time is smaller than max(inline + sim)
        // NOTE: this loop is ineffective w/ higher node counts
        int i = 0;
        std::valarray<float> t_inline_sim = t_inline + t_sim;
        float t_inline_sim_max = t_inline_sim.max();

        while (t_inline_sim.max() < t_intransit.max())
        {
            // always push back to the fastest sim node
            int min_id = -1;
            float min_val = std::numeric_limits<float>::max();
            // std::min_element(begin(t_inline_sim), end(t_inline_sim)) - begin(t_inline_sim);
            for (int j = 0; j < t_inline_sim.size(); j++)
            {
                // don't process skipped nodes on sim nodes
                if (t_inline[j] > 0 && t_inline_sim[j] < min_val)
                {
                    min_id = j;
                    min_val = t_inline_sim[j];
                }
            }
            if (min_id == -1)   // no rendering at all
                break;

            // find the corresponding vis node
            const int target_vis_node = node_map[min_id];

            if (render_counts_vis[target_vis_node] > 0)
            {
                t_intransit[target_vis_node] -= vis_estimates[min_id] * vis_factor;
                // Add render receive cost to vis node.
                // t_intransit[target_vis_node] += 0.09f;
                render_counts_vis[target_vis_node]--;

                t_inline[min_id] += vis_estimates[min_id] * sim_factor;
                render_counts_sim[min_id]++;
            }
            else    // We ran out of renderings on this vis node. This should not happen.
            {
                std::cout << "=== Ran out of renderings on node "
                        << target_vis_node << std::endl;
                break;
            }

            // if sim node got all its images back for inline rendering
            // -> take it out of consideration
            if (render_counts_sim[min_id] == render_cfg.non_probing_count)
                t_inline[min_id] = -1.f; //std::numeric_limits<float>::max() - t_sim[min_id];

            // recalculate inline + sim time
            t_inline_sim = t_inline + t_sim;
            ++i;
            if (i > render_cfg.non_probing_count * mpi_props.sim_node_count)
                ASCENT_ERROR("Error during load distribution.")
        }
    }

    std::vector<int> render_counts_combined(render_counts_sim);
    render_counts_combined.insert(render_counts_combined.end(),
                                  render_counts_vis.begin(),
                                  render_counts_vis.end());

    if (mpi_props.rank == 0)
    {
        std::cout << "=== render_counts ";
        for (auto &a : render_counts_combined)
            std::cout << a << " ";
        std::cout << std::endl;
    }

    return render_counts_combined;
}


/**
 * Sort ranks in descending order according to sim + vis times estimations.
 */
std::vector<int> sort_ranks(const std::vector<float> &sim_estimates,
                            const std::vector<float> &vis_estimates,
                            const int render_count)
{
    assert(sim_estimates.size() == vis_estimates.size());
    std::vector<int> rank_order(sim_estimates.size());
    std::iota(rank_order.begin(), rank_order.end(), 0);

    std::stable_sort(rank_order.begin(),
                     rank_order.end(),
                     [&](int i, int j)
                     {
                         return sim_estimates[i] + vis_estimates[i]*render_count
                              > sim_estimates[j] + vis_estimates[j]*render_count;
                     }
                     );
    return rank_order;
}

/**
 * Assign sim nodes to vis nodes based on their overall sim+vis times.
 */
std::vector<int> node_assignment(const std::vector<float> &g_sim_estimates,
                                 const std::vector<float> &g_vis_estimates,
                                 const int vis_node_count, const int render_count)
{
    const std::vector<int> rank_order = sort_ranks(g_sim_estimates, g_vis_estimates, render_count);
    const int sim_node_count = rank_order.size() - vis_node_count;
    std::vector<float> vis_node_cost(vis_node_count, 0.f);
    std::vector<int> map(sim_node_count, -1);

    for (int i = 0; i < sim_node_count; ++i)
    {
        // pick node with lowest cost
        const int target_vis_node = std::min_element(vis_node_cost.begin(), vis_node_cost.end())
                                    - vis_node_cost.begin();
        // asssign the sim to to the vis node
        map[rank_order[i]] = target_vis_node;
        // adapt the cost on the vis node
        vis_node_cost[target_vis_node] += g_vis_estimates[rank_order[i]];
    }
    return map;
}

/**
 *
 */
std::vector<int> job_assignment(const std::vector<float> &sim_estimate,
                                const std::vector<float> &vis_estimates,
                                const std::vector<int> &rank_order,
                                const int vis_node_count,
                                const std::string &insitu_type)
{
    assert(sim_estimate.size() == vis_estimates.size() == rank_order.size());
    std::vector<int> map(rank_order.size(), -1);

    if (insitu_type == "inline")
        return map;
    std::vector<float> sum(vis_node_count, 0.f);

    // loop over sorted ranks excluding vis nodes
    for (int i,j = 0; i < rank_order.size() - vis_node_count; ++i, ++j)
    {
        int vis_node = j % vis_node_count;
        if (vis_estimates[rank_order[i]] + sim_estimate[rank_order[i]] > sum[vis_node]
            || insitu_type == "intransit")
        {
            // assign to vis node
            map[rank_order[i]] = vis_node;
            sum[vis_node] += vis_estimates[rank_order[i]];
        }
    }
    return map;
}


/**
 *
 */
template <typename T>
std::vector<int> sort_indices(const std::vector<T> &v)
{
    std::vector<int> indices(v.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&v](int i, int j) { return v[i] < v[j]; }
             );
    return indices;
}


//-----------------------------------------------------------------------------
std::string get_timing_file_name(const int value, const int precision, const std::string &prefix,
                                 const std::string &path = "timings")
{
    std::ostringstream oss;
    oss << path;
    oss << "/";
    oss << prefix;
    oss << "_";
    oss << std::setw(precision) << std::setfill('0') << value;
    oss << ".txt";
    return oss.str();
}

//-----------------------------------------------------------------------------
void log_time(std::chrono::time_point<std::chrono::system_clock> start,
              const std::string &description,
              const int rank)
{
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Elapsed time: " << elapsed.count()
    //           << "s rank " << rank << std::endl;
    std::ofstream out(get_timing_file_name(rank, 5, "vis"), std::ios_base::app);
    out << description << elapsed.count() << std::endl;
    out.close();
}

void log_duration(const std::chrono::duration<double> elapsed,
                  const std::string &description,
                  const int rank)
{
    std::ofstream out(get_timing_file_name(rank, 5, "vis"), std::ios_base::app);
    out << description << elapsed.count() << std::endl;
    out.close();
}

void log_global_time(const std::string &description,
                     const int rank)
{
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    std::ofstream out(get_timing_file_name(rank, 5, "global"), std::ios_base::app);
    out << description << " : " << millis << std::endl;
    out.close();
}

void print_time(std::chrono::time_point<std::chrono::system_clock> start,
                const std::string &description,
                const int rank = -1,
                const double factor = 1.0)
{
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << rank << description << elapsed.count() * factor << std::endl;
}



#ifdef ASCENT_MPI_ENABLED
int recv_any_using_schema(Node &node, const int src, const int tag, const MPI_Comm comm)
{
    MPI_Status status;

    int mpi_error = MPI_Probe(src, tag, comm, &status);

    int buffer_size = 0;
    MPI_Get_count(&status, MPI_BYTE, &buffer_size);

    Node n_buffer(DataType::uint8(buffer_size));

    mpi_error = MPI_Recv(n_buffer.data_ptr(),
                         buffer_size,
                         MPI_BYTE,
                         src,
                         tag,
                         comm,
                         &status);

    uint8 *n_buff_ptr = (uint8*)n_buffer.data_ptr();

    Node n_msg;
    // length of the schema is sent as a 64-bit signed int
    // NOTE: we aren't using this value  ...
    n_msg["schema_len"].set_external((int64*)n_buff_ptr);
    n_buff_ptr +=8;
    // wrap the schema string
    n_msg["schema"].set_external_char8_str((char*)(n_buff_ptr));
    // create the schema
    Schema rcv_schema;
    Generator gen(n_msg["schema"].as_char8_str());
    gen.walk(rcv_schema);

    // advance by the schema length
    n_buff_ptr += n_msg["schema"].total_bytes_compact();

    // apply the schema to the data
    n_msg["data"].set_external(rcv_schema,n_buff_ptr);

    // copy out to our result node
    node.update(n_msg["data"]);

    if (mpi_error)
        std::cout << "ERROR receiving dataset from " << status.MPI_SOURCE << std::endl;

    return status.MPI_SOURCE;
}
#endif

void pack_node(Node &node, Node &packed)
{
    conduit::Schema s_data_compact;

    // schema will only be valid if compact and contig
    if( node.is_compact() && node.is_contiguous())
    {
        s_data_compact = node.schema();
    }
    else
    {
        node.schema().compact_to(s_data_compact);
    }

    std::string snd_schema_json = s_data_compact.to_json();

    conduit::Schema s_msg;
    s_msg["schema_len"].set(DataType::int64());
    s_msg["schema"].set(DataType::char8_str(snd_schema_json.size()+1));
    s_msg["data"].set(s_data_compact);

    // create a compact schema to use
    conduit::Schema s_msg_compact;
    s_msg.compact_to(s_msg_compact);

    packed.set_schema(s_msg_compact);
    // these sets won't realloc since schemas are compatible
    packed["schema_len"].set((int64)snd_schema_json.length());
    packed["schema"].set(snd_schema_json);

    // packed["data"].set_external(node);
    packed["data"].update(node);

    // Node ninfo;
    // packed.info(ninfo);
    // ninfo.print();
}


void unpack_node(const Node &node, Node &unpacked)
{
    // debug_break();
    uint8 *n_buff_ptr = (uint8*)node.data_ptr();

    Node n_msg;
    // length of the schema is sent as a 64-bit signed int
    // NOTE: we aren't using this value  ...
    n_msg["schema_len"].set_external((int64*)n_buff_ptr);
    n_buff_ptr +=8;
    // wrap the schema string
    n_msg["schema"].set_external_char8_str((char*)(n_buff_ptr));
    // create the schema
    Schema rcv_schema;
    Generator gen(n_msg["schema"].as_char8_str());
    gen.walk(rcv_schema);

    // n_msg["schema"].print();

    // advance by the schema length
    n_buff_ptr += n_msg["schema"].total_bytes_compact();

    // apply the schema to the data
    n_msg["data"].set_external(rcv_schema, n_buff_ptr);

    // set data to our result node (external, no copy)
    unpacked.update_external(n_msg["data"]);
    // unpacked.set_external(n_msg["data"]);
    // unpacked.update(n_msg["data"]);

    // std::cout << "unpack " << std::endl;
    // Node ninfo;
    // unpacked.info(ninfo);
    // ninfo.print();
}


/**
 * Detach and free MPI buffer.
 * Blocks until all buffered send messanges got received.
 */
#ifdef ASCENT_MPI_ENABLED
void detach_mpi_buffer()
{
    int size;
    char *bsend_buf;
    // block until all messages currently in the buffer have been transmitted
    MPI_Buffer_detach(&bsend_buf, &size);
    // clean up old buffer
    free(bsend_buf);
}
#endif

/**
 * Calculate the message size for sending the render chunks.
 */
int calc_render_msg_size(const int render_count, const int width = 800, const int height = 800,
                         const int channels = 4+4)
{
    const int overhead_render = 396 + 512;    // TODO: add correct bytes for name
    const int overhead_global = 288;
    return render_count * channels * width * height +
           render_count * overhead_render + overhead_global;
}

/**
 * Calculate all batch sizes from a number of non-probing renders. 
 * @param render_count The number of non-probing renders to be divided into batches.
 * @param render_cfg The render configuration.
 * @param include_probing Indicates if the result should include probing renders.
 * @param min_batch_size The minimum size of a single batch.
 * @return Vector containing all batch sizes.
 */
std::vector<int> get_batch_sizes(const int render_count, const RenderConfig render_cfg,
                                 const bool include_probing, const int min_batch_size = 32)
{
    // assert(render_cfg.batch_count > 0 && render_cfg.probing_stride > 2);
    if (render_count <= 0)
        return std::vector<int>();

    // make sure batch size is at least the given minimum size
    int batch_count = render_cfg.batch_count;
    while (render_count/batch_count < min_batch_size && batch_count > 1)
        --batch_count;

    std::vector<int> batch_sizes(batch_count);

    int total_count = render_cfg.get_render_count_from_non_probing(render_count);

    int size = total_count / batch_count;
    int offset = 0;
    int end = 0;
    for (int i = 0; i < batch_count; ++i)
    {
        end = offset + size;
        if (end >= total_count)  // last batch
        {
            batch_sizes[i] = total_count - offset;
            // corner case: last image(s) is a probing -> TODO: do we need this??
            // int j = 1;
            // while ( render_cfg.probing_count - j >= 0 
            //         && render_cfg.probing_ids.at(render_cfg.probing_count - j) == total_count - j
            //         && batch_sizes[i] > j)
            // {
            //     batch_sizes[i] -= 1;
            //     ++j;
            // }
        }
        else
        {
            bool is_next_probing = render_cfg.get_next_probing_id(offset + size, end);
            if (is_next_probing)
                batch_sizes[i] = end - offset;
            else
                batch_sizes[i] = size;
        }
        offset += batch_sizes[i];
    }
    if (!include_probing)    // do not include probing images in count
    {
        std::vector<int> batch_sizes_probing(batch_sizes.size());
        std::copy(batch_sizes.begin(), batch_sizes.end(), batch_sizes_probing.begin());
        offset = 0;
        for (int i = 0; i < batch_count; ++i)
        {
            batch_sizes[i] -= render_cfg.get_probing_count_part(batch_sizes[i], offset);
            offset += batch_sizes_probing[i];
        }
    }

    // --- legacy code
    // round to next image before a probing, so that we always start a batch with a probing image
    // if (size > 0 && include_probing)
    //     size += (render_cfg.probing_stride) - (size % (render_cfg.probing_stride));
    // else if (size > 0)
    //     size += (render_cfg.probing_stride - 1) - (size % (render_cfg.probing_stride - 1));

    // int sum = 0;
    // for (size_t i = 0; i < batch_sizes.size(); ++i)
    // {
    //     batch_sizes[i] = size;
    //     sum += size;
    // }

    // // last batch renders the rest
    // if (total_count - sum > 0)
    //     batch_sizes.push_back(total_count - sum);
    return batch_sizes;
}


int get_current_batch_size(const RenderBatch batch, const int iteration)
{
    int current_batch_size = batch.size;
    if ((iteration == batch.runs - 1) && (batch.rest != 0))
        current_batch_size = batch.rest;
    return current_batch_size;
}

/**
 * Make a unique pointer (for backward compatability, native since c++14).
 */
template<typename T, typename... Args>
unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

typedef std::vector<std::shared_ptr<conduit::Node> > vec_node_sptr;
typedef std::vector<vec_node_sptr> vec_vec_node_sptr;

typedef std::vector<std::unique_ptr<conduit::Node> > vec_node_uptr;
typedef std::vector<vec_node_uptr> vec_vec_node_uptr;

static const int SLEEP = 0; // milliseconds

void save_image(vtkh::Image *image, const std::string &name)
{
    image->Save(name, true);
}

void image_consumer(std::mutex &mu, std::condition_variable &cond,
                    std::deque<std::pair<vtkh::Image *, std::string> > &buffer)
{
    // std::cout << "Created consumer " << std::this_thread::get_id() << std::endl;
    while (true)
    {
        std::unique_lock<std::mutex> mlock(mu);
        cond.wait(mlock, [&buffer](){ return buffer.size() > 0; });
        std::pair<vtkh::Image *, std::string> image = buffer.front();
        // std::cout << "consumed " << image.second << std::endl;
        buffer.pop_front();
        mlock.unlock();
        cond.notify_all();

        if (image.first == nullptr && image.second == "KILL") // poison indicator
        {
            // std::cout << "Killed consumer " << std::this_thread::get_id() << std::endl;
            return;
        }
        else
        {
            image.first->Save(image.second, true);
        }
    }
}


void ActivePixelDecoding(unsigned char *colors, float *depths, const int size,
                        std::vector<unsigned char> &dec_colors, std::vector<float> &dec_depths)
{
    // TODO: assert depth.size() >= 2
    dec_depths.resize(size);
    dec_colors.resize(size * 4);
    // set first value
    int pos = 0;
    int dec_pos = 0;
    for (int i = 0; i < 4; i++)
        dec_colors[dec_pos + i] = colors[dec_pos + i];
    dec_depths[dec_pos++] = depths[pos++];

    while (dec_pos < size - 1)
    {
        for (int i = 0; i < 4; i++)
            dec_colors[dec_pos*4 + i] = colors[pos*4 + i];
        dec_depths[dec_pos++] = depths[pos++];

        if (dec_depths.at(dec_pos - 1) == dec_depths.at(dec_pos))
        {
            int count = int(depths[pos++]);
            for (int i = 0; i < count - 1; i++)
            {
                for (int i = 0; i < 4; i++)
                    dec_colors[dec_pos*4 + i] = colors[(pos - 2)*4 + i];
                dec_depths[dec_pos++] = depths[pos - 2];
            }
        }
    }
    // TODO: validate if this works as intended
}


#ifdef ASCENT_MPI_ENABLED
/**
 *  Composite render chunks from probing, simulation nodes, and visualization nodes.
 */
void hybrid_compositing(const vec_node_uptr &render_chunks_probe,
                        vec_vec_node_uptr &render_chunks_sim,
                        const vec_node_sptr &render_chunks_vis,
                        const std::vector<int> &g_render_counts,
                        const std::vector<int> &src_ranks,
                        const std::vector<int> &depth_id_order,
                        const std::map<int, int> &recv_counts,
                        const int my_vis_rank,
                        const int my_render_recv_cnt, const int my_data_recv_cnt,
                        const RenderConfig &render_cfg, const MPI_Properties &mpi_props,
                        const MPI_Comm active_vis_comm)
{
    auto t_start0 = std::chrono::system_clock::now();

    // unpack sent renders
    vec_node_sptr parts_probing;
    for (auto const& p : render_chunks_probe)
    {
        parts_probing.emplace_back(make_shared<Node>());
        if (p)
            unpack_node(*p, *parts_probing.back());
    }
    // sender / batches
    vec_vec_node_sptr parts_sim(my_render_recv_cnt);
    for (int i = 0; i < my_render_recv_cnt; ++i)
    {
        for (auto const& batch : render_chunks_sim[i])
        {
            parts_sim[i].emplace_back(make_shared<Node>());
            unpack_node(*batch, *parts_sim[i].back());
        }
    }

    // std::vector<RenderBatch> batches(my_render_recv_cnt);
    std::vector<std::vector<int> > sim_batch_sizes(my_render_recv_cnt);
    for (int i = 0; i < sim_batch_sizes.size(); ++i)
    {
        sim_batch_sizes[i] = get_batch_sizes(g_render_counts[src_ranks[i]], render_cfg, false);
        // std::cout << mpi_props.rank << " batch_sizes comp ";
        // for (auto &b : sim_batch_sizes[i])
        //     std::cout << b << " ";
        // std::cout << std::endl;
    }

    //     batches[i] = get_batch(g_render_counts[src_ranks[i]], render_cfg.batch_count);

    std::cout << mpi_props.rank << " * VIS: sort renders for compositing " << std::endl;

    // arrange render order
    vector<int> probing_enum_sim(my_data_recv_cnt, 0);
    vector<int> probing_enum_vis(my_data_recv_cnt, 0);
    // images / sender / values
    vec_vec_node_sptr render_ptrs(render_cfg.max_count);
    std::vector<std::vector<int> > render_arrangement(render_cfg.max_count);
    int probing_it = 0;

    bool print_compositing_order = false;   // debug out for compositing sort
    if (false && mpi_props.rank == 9)
        print_compositing_order = true;

    for (int j = 0; j < render_cfg.max_count; ++j)
    {
        render_ptrs[j].reserve(my_data_recv_cnt);
        render_arrangement[j].reserve(my_data_recv_cnt);

        if (print_compositing_order)
            std::cout << "\nimage " << j << std::endl;
        for (int i = 0; i < my_data_recv_cnt; ++i)
        {
            if (print_compositing_order)
                std::cout << "  " << i << " " << probing_enum_sim[i];
            // if (render_cfg.probing_stride && (j % render_cfg.probing_stride == 0)) // probing image
            if (probing_it < render_cfg.probing_ids.size() 
                && render_cfg.probing_ids[probing_it] == j)
            {
                // const index_t id = j / render_cfg.probing_stride;
                if (parts_probing[i]->has_child("render_file_names"))
                {
                    render_ptrs[j].emplace_back(parts_probing[i]);
                    render_arrangement[j].emplace_back(probing_it);
                    if (print_compositing_order)
                        std::cout << " " << mpi_props.rank << " probe  " << probing_it << std::endl;
                }
                else
                {
                    if (print_compositing_order)
                        std::cout << " " << mpi_props.rank << " skip probe " << probing_it << std::endl;
                }

                {   // keep track of probing images
                    // reset probing counter if first render in vis chunks
                    if (j == g_render_counts[src_ranks[i]] + probing_enum_sim[i])
                        probing_enum_vis[i] = 0;
                    if (j < g_render_counts[src_ranks[i]] + probing_enum_sim[i])
                        ++probing_enum_sim[i];
                    else
                        ++probing_enum_vis[i];
                }
                // increase probing iterator only after processing renders from the last data set
                if (i == my_data_recv_cnt - 1)   
                    ++probing_it;
            }
            else if (j < g_render_counts[src_ranks[i]] + probing_enum_sim[i]) // part comes from sim node (inline)
            {
                int batch_id = 0;
                int sum = 0;
                for (size_t k = 0; k < sim_batch_sizes[i].size(); k++)
                {
                    sum += sim_batch_sizes[i][k];
                    // if (j >= render_cfg.get_render_count_from_non_probing(sum))
                    if (j >= sum + probing_enum_sim[i])
                        ++batch_id;
                }

                index_t id = j;
                for (int k = 0; k < batch_id; k++)
                    id -= sim_batch_sizes[i][k];

                id -= probing_enum_sim[i];

                if (parts_sim[i][batch_id]->has_child("render_file_names"))
                {
                    render_ptrs[j].emplace_back(parts_sim[i][batch_id]);
                    render_arrangement[j].emplace_back(id);
                    if (print_compositing_order)
                        std::cout << " " << mpi_props.rank << " sim  " << id << " | batch " << batch_id << std::endl;
                    // std::cout << "     " << mpi_props.rank << " batch size  " << sim_batch_sizes[i][0] << " | probe " << probing_enum_sim[i] << std::endl;
                }
                else
                {
                    if (print_compositing_order)
                        std::cout << " " << mpi_props.rank << " skip sim  " << id << std::endl;
                }
            }
            else    // part rendered on this vis node
            {
                // Reset the probing counter if this is the first render in vis node chunks
                // and this is not a probing render.
                if (j == g_render_counts[src_ranks[i]] + probing_enum_sim[i])
                    probing_enum_vis[i] = 0;

                const index_t id = j - (g_render_counts[src_ranks[i]] + probing_enum_sim[i])
                                     - probing_enum_vis[i];

                if (render_chunks_vis[i] && render_chunks_vis[i]->has_child("render_file_names"))
                {
                    render_ptrs[j].emplace_back(render_chunks_vis[i]);
                    render_arrangement[j].emplace_back(id);
                    if (print_compositing_order)
                        std::cout << " " << mpi_props.rank << " vis  " << id << std::endl;
                }
                else
                {
                    if (print_compositing_order)
                        std::cout << " " << mpi_props.rank << " skip vis " << id << std::endl;
                }
            }
        }
    }

    std::cout << mpi_props.rank << " * VIS: start compositing " << std::endl;

    // Set the vis_comm to be the vtkh comm.
    vtkh::SetMPICommHandle(int(MPI_Comm_c2f(active_vis_comm)));
    MPI_Barrier(active_vis_comm);

    // Set the number of receiving depth values per node and the according displacements.
    std::vector<int> counts_recv;
    std::vector<int> displacements(1, 0);
    for (const auto &e : recv_counts)
    {
        counts_recv.push_back(e.second);
        displacements.push_back(displacements.back() + e.second);
    }
    displacements.pop_back();

    unsigned int thread_count = std::thread::hardware_concurrency();
    thread_count = std::min(thread_count, 16u);     // limit to 16 consumers to avoid overhead
    std::mutex mu;
    const int max_buffer_size = thread_count * 4;   // buffer a max of 4 images per thread
    std::condition_variable cond;
    std::deque<std::pair<vtkh::Image *, std::string> > buffer;
    std::vector<std::thread> consumers(thread_count);
    if (my_vis_rank == 0)
    {
        for (int i = 0; i < consumers.size(); ++i)
            consumers[i] = std::thread(&image_consumer, std::ref(mu), std::ref(cond), std::ref(buffer));
    }

    std::vector<vtkh::Compositor> compositors(render_cfg.max_count);
    std::vector<vtkh::Image *> results(render_cfg.max_count);
    std::vector<std::thread> threads;

    // loop over images (camera positions)
    for (int j = 0; j < render_cfg.max_count; ++j)
    {
        compositors[j].SetCompositeMode(vtkh::Compositor::VIS_ORDER_BLEND);
        auto t_start = std::chrono::system_clock::now();
        // gather all dephts values from vis nodes
        std::vector<float> v_depths(mpi_props.sim_node_count, std::numeric_limits<float>::lowest());
        std::vector<float> depths(render_ptrs[j].size());

        for (int i = 0; i < render_ptrs[j].size(); i++)
        {
            // std::cout << ": " << j << " " << i << " " << std::endl;
            depths[i] = (*render_ptrs[j][i])["depths"].child(render_arrangement[j][i]).to_float();
        }

        MPI_Allgatherv(depths.data(), depths.size(),
                        MPI_FLOAT, v_depths.data(), counts_recv.data(), displacements.data(),
                        MPI_FLOAT, active_vis_comm);

        std::vector<std::pair<float, int> > depth_id(v_depths.size());

        for (int k = 0; k < v_depths.size(); k++)
            depth_id[k] = std::make_pair(v_depths[k], depth_id_order[k]);
        // sort based on depth values
        std::sort(depth_id.begin(), depth_id.end());

        // convert the depth order to an integer ranking
        std::vector<int> depths_order;
        for(auto it = depth_id.begin(); it != depth_id.end(); it++)
            depths_order.push_back(it->second);

        // get a mapping from MPI src rank to depth rank
        std::vector<int> depths_order_id = sort_indices(depths_order);

        log_global_time("end getDepthOrder", mpi_props.rank);

        int image_cnt = 0;
        // loop over render parts (= 1 per sim node) and add as images
        for (int i = 0; i < render_ptrs.at(j).size(); ++i)
        {
            const int id = depths_order_id.at(src_ranks.at(i));

            unsigned char *cb = (*render_ptrs[j][i])["color_buffers"].child(render_arrangement[j][i]).as_unsigned_char_ptr();
            float *db = (*render_ptrs[j][i])["depth_buffers"].child(render_arrangement[j][i]).as_float_ptr();
            // std::cout << mpi_props.rank << " | ..add image " << j << " " << render_arrangement.at(j).at(i) << " | id " << id << std::endl;

            // TODO: test & debug ActivePixelDecoding
            if (false)
            {
                std::vector<unsigned char> dec_colors;
                std::vector<float> dec_depths;
                ActivePixelDecoding(cb, db, render_cfg.WIDTH * render_cfg.HEIGHT,
                                    dec_colors, dec_depths);
                compositors[j].AddImage(dec_colors.data(), dec_depths.data(), render_cfg.WIDTH,
                                            render_cfg.HEIGHT, id);
            }
            else
            {
                compositors[j].AddImage(cb, db, render_cfg.WIDTH, render_cfg.HEIGHT, id);
            }
            ++image_cnt;
        }
        // std::cout << mpi_props.rank << " | ..composite " << j << " img count " << image_cnt << std::endl;

        // composite
        results[j] = compositors[j].CompositeNoCopy();

        // TODO: add screen annotations for hybrid (see vtk-h Scene::Render)
        // See vtk-h Renderer::ImageToCanvas() for how to get from result image to canvas.
        // Problem: we still need camera, ranges and color table.

        log_time(t_start, "+ compositing image ", mpi_props.rank);
        log_global_time("end composit image", mpi_props.rank);

        // print_time(t_start, " * VIS: end composite ", mpi_props.rank);

        // save render using separate thread to hide latency
        if (my_vis_rank == 0 && image_cnt)
        {
            assert(render_arrangement[j].size() > 0);
            std::string name = (*render_ptrs[j][0])["render_file_names"].child(render_arrangement[j][0]).as_string();
            // std::cout << name << std::endl;

            std::unique_lock<std::mutex> mlock(mu);
            cond.wait(mlock, [&buffer, &max_buffer_size](){ return buffer.size() < max_buffer_size; });
            buffer.push_back(std::make_pair(results[j], name));
            // std::cout << "produced " << name << std::endl;
            mlock.unlock();
            cond.notify_all();

            if (j == render_cfg.max_count - 1)  // last image -> kill consumers
            {
                std::cout << "Clean up consumers." << std::endl;
                // poison consumers for cleanup
                for (int i = 0; i < consumers.size(); ++i)
                {
                    std::unique_lock<std::mutex> locker(mu);
                    cond.wait(locker, [&buffer, &max_buffer_size](){ return buffer.size() < max_buffer_size; });
                    buffer.push_back(std::make_pair(nullptr, std::string("KILL")));
                    locker.unlock();
                    cond.notify_all();
                }
                for (auto& t : consumers)
                    t.join();
            }
        }
        // print_time(t_start, " * VIS: end save ", mpi_props.rank);
    }

    log_time(t_start0, "+ compositing total ", mpi_props.rank);
    log_global_time("end compositing", mpi_props.rank);
}
#endif

// TODO: fix
void convert_color_buffer(Node &data)
{
    int size = 800*800*4;
    for (int i = 0; i < data["color_buffers"].number_of_children(); i++)
    {
        float *cb_float = data["color_buffers"].child(i).as_float_ptr();
        std::vector<unsigned char> cb_uchar(size);

    #pragma omp parallel for
        for (size_t j = 0; j < cb_uchar.size(); j++)
            cb_uchar[j] = static_cast<unsigned char>(int(cb_float[j] * 255.f));

        data["color_buffers_uchar"].append();
        data["color_buffers_uchar"][i].set(cb_uchar.data(), size);
    }
    data.remove("color_buffers");
}

#ifdef ASCENT_MPI_ENABLED
void pack_and_send(Node &data, const int destination, const int tag,
                   const MPI_Comm comm, MPI_Request &req)
{
    Node compact_node;
    pack_node(data, compact_node);

    int mpi_error = MPI_Ibsend(compact_node.data_ptr(),
                               compact_node.total_bytes_compact(),
                               MPI_BYTE,
                               destination,
                               tag,
                               comm,
                               &req
                               );
    if (mpi_error)
        std::cout << "ERROR sending node to " << destination << std::endl;
}
#endif

void get_renders(Ascent &ascent, std::shared_ptr<conduit::Node> &renders)
{
    conduit::Node info;
    ascent.info(info);

    if (info.has_child("render_file_names"))
        renders = std::make_shared<Node>(info);
    else
        std::cout << "no render_file_names" << std::endl;
}

//-----------------------------------------------------------------------------
void hybrid_render(const MPI_Properties &mpi_props,
                   const RenderConfig &render_cfg,
                   const std::vector<double> &my_probing_times,
                   const double total_probing_time,
                   conduit::Node &data,
                   conduit::Node &render_chunks_probing)
{
    assert(render_cfg.insitu_type != "inline");
    assert(mpi_props.sim_node_count > 0 && mpi_props.sim_node_count <= mpi_props.size);

    if (mpi_props.rank == 0)
    {
        std::cout << "=== Probing sequence: ";
        for (auto &a : render_cfg.probing_ids)
            std::cout << a << " ";
        std::cout << std::endl;
    }

    auto start0 = std::chrono::system_clock::now();

    bool is_vis_node = false;
    int my_vis_rank = -1;

    float my_avg_probing_time = 0.f;
    float my_render_overhead = 0.f;
    int skipped_render = 0;
    float my_sim_estimate = data["state/sim_time"].to_float();
    std::cout << mpi_props.rank << " : sim time estimate " << my_sim_estimate << std::endl;

    Node data_packed;
    pack_node(data, data_packed);
    int my_data_size = data_packed.total_bytes_compact();

#ifdef ASCENT_MPI_ENABLED
    MPI_Barrier(mpi_props.comm_world);
#endif

    if (mpi_props.rank >= mpi_props.sim_node_count) // nodes with the highest ranks are vis nodes
    {
        is_vis_node = true;
        my_vis_rank = mpi_props.rank - mpi_props.sim_node_count;
        my_sim_estimate = 0.f;
        my_data_size = 0;
    }
    else if (mpi_props.size > 1) // otherwise this is a sim node
    {
        double sum_render_times = std::accumulate(my_probing_times.begin(),
                                                  my_probing_times.end(), 0.0);
        sum_render_times = std::isnan(sum_render_times) ? 0.0 : sum_render_times;

        if (my_probing_times.size() > 0)
        {
            my_avg_probing_time = float(sum_render_times / my_probing_times.size());
            // try median
            std::vector<double> my_probing_times_copy;
            std::copy(my_probing_times.begin(), my_probing_times.end(),
                      std::back_inserter(my_probing_times_copy));
            std::sort(my_probing_times_copy.begin(), my_probing_times_copy.end());
            my_avg_probing_time = my_probing_times_copy[my_probing_times_copy.size()/2];

            my_avg_probing_time /= 1000.f; // convert to seconds
        }

        bool use_total_time = false;
        if (my_avg_probing_time > 0.f && use_total_time)
        {
            my_avg_probing_time = total_probing_time;
            if (render_cfg.probing_count)
                my_avg_probing_time /= render_cfg.probing_count;
        }

        // if probing time is close to zero, add overhead costs
        // if (my_avg_probing_time < 0.f + std::numeric_limits<float>::min())
        //     my_avg_probing_time = total_probing_time / render_cfg.probing_count / 1.f;

        // std::cout << "+++ probing times ";
        // for (auto &a : my_probing_times)
        //     std::cout << a << " ";
        std::stringstream ss;
        ss << mpi_props.rank << " ~SIM: probing time (overhead): " << sum_render_times/1000.0 
           << " (" << total_probing_time << ")" << std::endl;
        std::cout << ss.str();
        // std::cout << "probing w/  overhead " << total_probing_time << std::endl;
        my_render_overhead = total_probing_time - sum_render_times/1000.0;
        // my_render_overhead *= render_cfg.batch_count;

        std::cout << mpi_props.rank << " ~SIM: visualization time estimate (per render): "
                  << my_avg_probing_time << std::endl;

        if (render_cfg.insitu_type == "hybrid")
        {
            skipped_render = 1;
            if (render_chunks_probing.has_child("render_file_names")) //images"))
                skipped_render = 0; // false
        }
    }
    log_global_time("end packData", mpi_props.rank);

#ifdef ASCENT_MPI_ENABLED
    MPI_Barrier(mpi_props.comm_world);
    auto start1 = std::chrono::system_clock::now();

    // gather all simulation time estimates
    std::vector<float> g_sim_estimates(mpi_props.size, 0.f);
    MPI_Allgather(&my_sim_estimate, 1, MPI_FLOAT,
                  g_sim_estimates.data(), 1, MPI_FLOAT, mpi_props.comm_world);
    // gather all visualization time estimates
    std::vector<float> g_vis_estimates(mpi_props.size, 0.f);
    MPI_Allgather(&my_avg_probing_time, 1, MPI_FLOAT,
                  g_vis_estimates.data(), 1, MPI_FLOAT, mpi_props.comm_world);
    // and render overhead
    std::vector<float> g_vis_overhead(mpi_props.size, 0.f);
    MPI_Allgather(&my_render_overhead, 1, MPI_FLOAT,
                  g_vis_overhead.data(), 1, MPI_FLOAT, mpi_props.comm_world);
    // determine how many nodes skipped rendering due to empty block
    std::vector<int> g_skipped(mpi_props.size, 0);
    MPI_Allgather(&skipped_render, 1, MPI_INT, g_skipped.data(), 1, MPI_INT, mpi_props.comm_world);
    const float skipped_renders = std::accumulate(g_skipped.begin(), g_skipped.end(), 0)
                                    / float(mpi_props.sim_node_count);

    // NOTE: use maximum sim time for all nodes
    const float max_sim_time = *std::max_element(g_sim_estimates.begin(), g_sim_estimates.end());
    g_sim_estimates = std::vector<float>(mpi_props.size, max_sim_time);

    // assign sim nodes to vis nodes
    std::vector<int> node_map = node_assignment(g_sim_estimates, g_vis_estimates,
                                                mpi_props.vis_node_count,
                                                render_cfg.non_probing_count);

    // DEBUG: OUT
    if (mpi_props.rank == 0)
    {
        std::cout << "=== skipped_renders (relative) " << skipped_renders << std::endl;

        std::cout << "=== node_map ";
        for (auto &a : node_map)
            std::cout << a << " ";
        std::cout << std::endl;
    }

    // distribute rendering load across sim and vis loads
    const std::vector<int> g_render_counts = load_assignment(g_sim_estimates, g_vis_estimates,
                                                             g_vis_overhead,
                                                             node_map, render_cfg, mpi_props,
                                                             skipped_renders);

    // gather all data set sizes for async recv
    std::vector<int> g_data_sizes(mpi_props.size, 0);
    MPI_Allgather(&my_data_size, 1, MPI_INT, g_data_sizes.data(), 1, MPI_INT, mpi_props.comm_world);

    // mpi message tags
    const int TAG_DATA = 0;
    const int TAG_PROBING = TAG_DATA + 1;
    const int TAG_INLINE = TAG_PROBING + 1;
    // std::cout << "+++ MPI Comm world: " << MPI_Comm_c2f(mpi_props.comm_world) << std::endl;

    // common options for both sim and vis nodes
    Node ascent_opts, blank_actions;
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(mpi_props.comm_world);
    ascent_opts["actions_file"] = "cinema_actions.yaml";
    ascent_opts["is_probing"] = 0;
    ascent_opts["probing_factor"] = render_cfg.probing_factor;
    ascent_opts["sampling_method"] = render_cfg.sampling_method;
    ascent_opts["insitu_type"] = render_cfg.insitu_type;
    ascent_opts["field_filter"] = "true";

    log_time(start1, "- load distribution ", mpi_props.rank);
    log_global_time("end loadAssignment", mpi_props.rank);

    // ================ VIS nodes ================
    if (is_vis_node)
    {
        // find all sim nodes sending data to this vis node
        std::vector<int> sending_node_ranks;
        for (int i = 0; i < mpi_props.sim_node_count; ++i)
        {
            if (node_map[i] == my_vis_rank)
                sending_node_ranks.push_back(i);
        }
        const int my_data_recv_cnt = int(sending_node_ranks.size());
        // count of nodes that do inline rendering (0 for intransit case)
        const int my_render_recv_cnt = render_cfg.insitu_type == "intransit" ? 0 : my_data_recv_cnt;
        std::map<int, int> recv_counts;
        for (const auto &n : node_map)
            ++recv_counts[n];

        std::vector<int> depth_id_order;
        for (int i = 0; i < mpi_props.vis_node_count; i++)
        {
            std::vector<int>::iterator it = node_map.begin();
            while ((it = std::find_if(it, node_map.end(), [&](int x){return x == i; }))
                    != node_map.end())
            {
                int d = std::distance(node_map.begin(), it);
                depth_id_order.push_back(d);
                it++;
            }
        }
        // std::cout << "== depth_id_order ";
        // for(auto a : depth_id_order)
        //     std::cout << a << " ";
        // std::cout << std::endl;

        std::stringstream node_string;
        std::copy(sending_node_ranks.begin(), sending_node_ranks.end(),
                    std::ostream_iterator<int>(node_string, " "));
        std::cout << mpi_props.rank << " * VIS: receives extract(s) from "
                  << node_string.str() << std::endl;

        const std::vector<int> src_ranks = sending_node_ranks;
        std::vector<std::unique_ptr<Node> > datasets(my_data_recv_cnt);

        // post recv for datasets
        std::vector<MPI_Request> requests_data(my_data_recv_cnt, MPI_REQUEST_NULL);
        for (int i = 0; i < my_data_recv_cnt; ++i)
        {
            datasets[i] = make_unique<Node>(DataType::uint8(g_data_sizes[src_ranks[i]]));

            int mpi_error = MPI_Irecv(datasets[i]->data_ptr(),
                                      datasets[i]->total_bytes_compact(),
                                      MPI_BYTE,
                                      src_ranks[i],
                                      TAG_DATA,
                                      mpi_props.comm_world,
                                      &requests_data[i]
                                      );
            if (mpi_error)
                std::cout << "ERROR receiving dataset from " << src_ranks[i] << std::endl;
            // std::cout << mpi_props.rank << " * VIS: " << " receiving " << g_data_sizes[src_ranks[i]]
            //           << " bytes from " << src_ranks[i] << std::endl;
        }

        // every associated sim node sends n batches of renders to this vis node
        // std::vector<RenderBatch> batches(my_render_recv_cnt);
        std::vector<std::vector<int> > sim_batch_sizes(my_render_recv_cnt);

        for (int i = 0; i < sim_batch_sizes.size(); ++i)
        {
            // int render_count = g_render_counts[src_ranks[i]]
            //                     + int(g_render_counts[src_ranks[i]]*render_cfg.probing_factor);
            // batches[i] = get_batch(render_count, render_cfg.batch_count);

            sim_batch_sizes[i] = get_batch_sizes(g_render_counts[src_ranks[i]], render_cfg, false);            
        }
        

        // probing chunks
        vec_node_uptr render_chunks_probe(my_render_recv_cnt);
        std::vector<MPI_Request> requests_probing(my_render_recv_cnt, MPI_REQUEST_NULL);
        // render chunks sim
        // senders / batches / renders
        vec_vec_node_uptr render_chunks_sim(my_render_recv_cnt);
        std::vector< std::vector<MPI_Request>> requests_inline_sim(my_render_recv_cnt);
        
        // pre-allocate the mpi receive buffers
        for (int i = 0; i < my_render_recv_cnt; i++)
        {
            int buffer_size = 0;
            if (g_skipped[src_ranks[i]])
            {
                render_chunks_probe[i] = nullptr;
            }
            else
            {
                buffer_size = calc_render_msg_size(render_cfg.probing_count);
                render_chunks_probe[i] = make_unique<Node>(DataType::uint8(buffer_size));
            }

            render_chunks_sim[i].resize(sim_batch_sizes[i].size());
            requests_inline_sim[i].resize(sim_batch_sizes[i].size(), MPI_REQUEST_NULL);

            for (int j = 0; j < sim_batch_sizes[i].size(); ++j)
            {
                buffer_size = calc_render_msg_size(sim_batch_sizes[i][j]); // render_cfg.probing_factor);
                render_chunks_sim[i][j] = make_unique<Node>(DataType::uint8(buffer_size));
                // std::cout << mpi_props.rank << " | " << i << " " << j << " " << sim_batch_sizes[i][j] 
                //           << " expected render_msg_size " << buffer_size << std::endl;
            }
        }

        // post the receives for the render chunks to receive asynchronous (non-blocking)
        for (int i = 0; i < my_render_recv_cnt; ++i)
        {
            if (!g_skipped[src_ranks[i]])
            {
                // receive probing render chunks
                int mpi_error = MPI_Irecv(render_chunks_probe[i]->data_ptr(),
                                          render_chunks_probe[i]->total_bytes_compact(),
                                          MPI_BYTE,
                                          src_ranks[i],
                                          TAG_PROBING,
                                          mpi_props.comm_world,
                                          &requests_probing[i]
                                          );
                if (mpi_error)
                    std::cout << "ERROR receiving probing parts from " << src_ranks[i] << std::endl;
            }

            std::stringstream ss;
            ss << mpi_props.rank << " * VIS: receiving batches of sizes ";
            for (auto &b : sim_batch_sizes[i])
                ss << b << " ";
            ss << " from " << src_ranks[i] << std::endl;
            std::cout << ss.str();

            for (int j = 0; j < sim_batch_sizes[i].size(); ++j)
            {
                if (sim_batch_sizes[i][j] <= 0)
                    break;

                int mpi_error = MPI_Irecv(render_chunks_sim[i][j]->data_ptr(),
                                          render_chunks_sim[i][j]->total_bytes_compact(),
                                          MPI_BYTE,
                                          src_ranks[i],
                                          TAG_INLINE + j,
                                          mpi_props.comm_world,
                                          &requests_inline_sim[i][j]
                                          );
                if (mpi_error)
                    std::cout << "ERROR receiving render parts from " << src_ranks[i] << std::endl;
            }
        }

        // wait for all data sets to arrive
        // NOTE: if we use waitany and then render in between waiting for the next dataset,
        // we stall the remaining sim nodes until the rendering on this vis node is finished
        for (int i = 0; i < my_data_recv_cnt; ++i)
        {
            int id = -1;
            auto start1 = std::chrono::system_clock::now();
            MPI_Waitany(requests_data.size(), requests_data.data(), &id, MPI_STATUS_IGNORE);
            log_time(start1, "- receive data ", mpi_props.rank);
        }
        log_global_time("end receiveData", mpi_props.rank);

        // render all data sets
        std::vector<Ascent> ascent_renders(my_data_recv_cnt);
        vec_node_sptr render_chunks_vis(my_data_recv_cnt, nullptr);
        std::vector<std::thread> threads;

        for (int i = 0; i < my_data_recv_cnt; ++i)
        {
            // std::cout << "=== dataset size " << mpi_props.rank << " from " << src_ranks[i] << " "
            //           << datasets[i]->total_bytes_compact() << std::endl;
            Node dataset;
            unpack_node(*datasets[i], dataset);

            Node verify_info;
            if (conduit::blueprint::mesh::verify(dataset, verify_info))
            {
                // vis node needs to render what is left
                const int render_count_sim = render_cfg.get_render_count_from_non_probing(g_render_counts[src_ranks[i]]);
                const int current_render_count = render_cfg.max_count - render_count_sim;
                const int render_offset = render_cfg.max_count - current_render_count;
                const int probing_count_part = render_cfg.get_probing_count_part(current_render_count, render_offset);

                auto start = std::chrono::system_clock::now();
                if (current_render_count - probing_count_part > 0)
                {
                    // debug_break();
                    std::cout   << mpi_props.rank << " * VIS: rendering "
                                << render_offset << " - "
                                << render_offset + current_render_count << std::endl;

                    ascent_opts["render_offset"] = render_offset;
                    ascent_opts["render_count"] = current_render_count;
                    ascent_opts["cinema_increment"] = (i == 0) ? true : false;
                    ascent_opts["sleep"] = (src_ranks[i] == 0) ? SLEEP : 0;

                    auto t_render = std::chrono::system_clock::now();
                    ascent_renders[i].open(ascent_opts);
                    ascent_renders[i].publish(dataset);
                    ascent_renders[i].execute(blank_actions);
                    print_time(t_render, " * VIS: avg t/render ", mpi_props.rank, 1.0 / current_render_count);

                    threads.push_back(std::thread(&get_renders, std::ref(ascent_renders[i]),
                                                   std::ref(render_chunks_vis[i])));
                    // conduit::Node info;
                    // // ascent_main_runtime : out.set_external(m_info);
                    // ascent_renders[i].info(info);

                    // if (info.has_child("render_file_names"))
                    // {
                    //     render_chunks_vis[i] = std::make_shared<Node>(info);
                    //     // ascent_renders[i].info(*render_chunks_vis[i]);
                    //     // render_chunks_vis.push_back(std::make_shared<Node>(info));
                    // }
                }
                else
                {
                    render_chunks_vis[i] = std::make_shared<Node>();
                }

                log_time(start, "+ render vis " + std::to_string(current_render_count - probing_count_part) + " ", mpi_props.rank);
            }
            else
            {
                std::cout << "ERROR: rank " << mpi_props.rank 
                          << " * VIS: could not verify (conduit::blueprint::mesh::verify) the sent data." 
                          << std::endl;
            }
        }   // for: render all datasets sent
        
        auto t_render = std::chrono::system_clock::now();
        while (threads.size() > 0)
        {
            if (threads.back().joinable())
                threads.back().join();
            else
                ASCENT_ERROR("Thread not joinable.")
            threads.pop_back();
        }
        print_time(t_render,  " * VIS: copy total ", mpi_props.rank);

        log_global_time("end render", mpi_props.rank);

        {   // wait for receive of render chunks to complete
            auto t_start = std::chrono::system_clock::now();
            // renders from probing
            std::cout << mpi_props.rank << " * VIS: wait for receive of probing renders." << std::endl;
            MPI_Waitall(requests_probing.size(), requests_probing.data(), MPI_STATUSES_IGNORE);
            std::cout << mpi_props.rank << " * VIS: wait for receive of inline renders." << std::endl;
            // inline renders
            for (auto &batch_requests : requests_inline_sim)
            {
                int mpi_error = MPI_Waitall(batch_requests.size(), batch_requests.data(), MPI_STATUSES_IGNORE);
                if (mpi_error)
                    std::cout << "ERROR: waitall (vis node receiving inline renders) " << mpi_props.rank << std::endl;
            }
            log_time(t_start, "+ wait receive img ", mpi_props.rank);
        }
        log_global_time("end receiveRenders", mpi_props.rank);

        // find out which of the vis nodes do actual rendering/compositing
        std::vector<int> active_nodes(mpi_props.vis_node_count, 0);
        for (int v = 0; v < mpi_props.vis_node_count; v++)
        {
            for (int i = 0; i < mpi_props.sim_node_count; ++i)
            {
                if (node_map[i] == v)
                    if (!g_skipped[i])
                        active_nodes[v] = 1;
            }
        }
        const int active_vis_nodes = std::accumulate(active_nodes.begin(), active_nodes.end(), 0);
        std::cout << mpi_props.rank << " * VIS: Active vis nodes: " << active_vis_nodes << std::endl;

        // adapt the vis comm according to the active vis nodes
        std::vector<int> vis_ranks(active_vis_nodes);
        std::iota(vis_ranks.begin(), vis_ranks.end(), 0); // inactive nodes are always highest ranks
        MPI_Group active_vis_group;
        MPI_Group_incl(mpi_props.vis_group, active_vis_nodes, vis_ranks.data(), &active_vis_group);
        MPI_Comm active_vis_comm;
        MPI_Comm_create_group(mpi_props.comm_vis, active_vis_group, 2, &active_vis_comm);

        if (active_nodes[my_vis_rank])
        {
            hybrid_compositing(render_chunks_probe, render_chunks_sim, render_chunks_vis,
                               g_render_counts, src_ranks, depth_id_order, recv_counts, my_vis_rank,
                               my_render_recv_cnt, my_data_recv_cnt, render_cfg,
                               mpi_props, active_vis_comm);

            MPI_Group_free(&active_vis_group);
            if (active_vis_comm != MPI_COMM_NULL)
            {
                MPI_Barrier(active_vis_comm);
                MPI_Comm_free(&active_vis_comm);
            }

            // Keep the vis node ascent instances open until render chunks have been processed.
            for (int i = 0; i < my_data_recv_cnt; i++)
                ascent_renders[i].close();
        }
    } // end vis node
    // ================ SIM nodes ================
    else
    {
        const int destination = node_map[mpi_props.rank] + mpi_props.sim_node_count;
        // std::cout << mpi_props.rank << " ~SIM: sends extract to "
        //           <<  node_map[mpi_props.rank] + mpi_props.sim_node_count << std::endl;
        Node verify_info;
        if (conduit::blueprint::mesh::verify(data, verify_info))
        {
            std::vector<int> batch_sizes = get_batch_sizes(g_render_counts[mpi_props.rank],
                                                           render_cfg, true);

            std::cout << mpi_props.rank << " ~SIM: batch_sizes ";
            for (auto &b : batch_sizes)
                std::cout << b << " ";
            std::cout << std::endl;

            {   // init send buffer
                detach_mpi_buffer();

                const index_t msg_size_render = calc_render_msg_size(g_render_counts[mpi_props.rank]);
                const index_t msg_size_probing = calc_render_msg_size(render_cfg.probing_count);
                const int overhead = MPI_BSEND_OVERHEAD * (batch_sizes.size() + 1); // 1 probing batch
                const int total_size = msg_size_render + msg_size_probing + overhead;

                MPI_Buffer_attach(malloc(total_size), total_size);
                // std::cout << mpi_props.rank << " -- buffer size: " << total_size << std::endl;
            }

            // MPI_Request request_data = MPI_REQUEST_NULL;
            // TODO: send data only if not skipped_render
            // {   // send data to vis node
            //     auto t_start = std::chrono::system_clock::now();
            //     int mpi_error = MPI_Isend(const_cast<void*>(data_packed.data_ptr()),
            //                               data_packed.total_bytes_compact(),
            //                               MPI_BYTE,
            //                               destination,
            //                               TAG_DATA,
            //                               mpi_props.comm_world,
            //                               &request_data
            //                               );
            //     if (mpi_error)
            //         std::cout << "ERROR sending sim data to " << destination << std::endl;
            //     log_time(t_start, "- send data ", mpi_props.rank);
            // }

            std::thread send_data_thread = std::thread(&MPI_Ssend, const_cast<void*>(data_packed.data_ptr()),
                                                        data_packed.total_bytes_compact(),
                                                        MPI_BYTE, destination, TAG_DATA, 
                                                        mpi_props.comm_world);

            // debug_break();
            MPI_Request request_probing = MPI_REQUEST_NULL;
            // pack and send probing renders in separate thread
            std::thread pack_probing_thread;
            if (!skipped_render)
            {
                pack_probing_thread = std::thread(&pack_and_send, std::ref(render_chunks_probing),
                                                  destination, TAG_PROBING, mpi_props.comm_world,
                                                  std::ref(request_probing));
            }

            log_global_time("end sendData", mpi_props.rank);

            // in line rendering using ascent
            std::vector<Ascent> ascent_renders(batch_sizes.size());
            std::vector<MPI_Request> requests(batch_sizes.size(), MPI_REQUEST_NULL);
            std::vector<conduit::Node> info(batch_sizes.size());
            std::vector<conduit::Node> renders_inline(batch_sizes.size());

            std::vector<std::thread> threads;
            std::chrono::duration<double> sum_render(0);
            std::chrono::duration<double> sum_copy(0);

            auto t_start = std::chrono::system_clock::now();
            for (int i = 0; i < batch_sizes.size(); ++i)
            {
                // int begin = std::accumulate(batch_sizes.begin(), batch_sizes.begin() + i, 0);
                int begin = 0;
                for (int j = 0; j < i; j++)
                    begin += batch_sizes[j];

                // const int current_batch_size = get_current_batch_size(batch, i);
                // if (current_batch_size <= 1)
                //     break;
                const int render_count = batch_sizes[i];
                if (render_count == 0)
                    break;

                std::cout << mpi_props.rank << " ~SIM: rendering "
                          << begin << " - " << begin + render_count << std::endl;

                ascent_opts["render_count"] = render_count;
                ascent_opts["render_offset"] = begin;
                ascent_opts["sleep"] = (mpi_props.rank == 0) ? SLEEP : 0;

                auto t_render = std::chrono::system_clock::now();
                ascent_renders[i].open(ascent_opts);
                ascent_renders[i].publish(data);
                // print_time(t_render, " ~SIM: publish data ", mpi_props.rank);
                t_render = std::chrono::system_clock::now();
                ascent_renders[i].execute(blank_actions);
                auto t_end = std::chrono::system_clock::now();
                sum_render += t_end - t_render;

                t_render = std::chrono::system_clock::now();
                // if (threads.size() > 0)
                // {
                //     threads.back().join();
                //     threads.pop_back();
                // }

                // send render chunks
                ascent_renders[i].info(info[i]);
                renders_inline[i]["depths"].set_external(info[i]["depths"]);
                renders_inline[i]["color_buffers"].set_external(info[i]["color_buffers"]);
                renders_inline[i]["depth_buffers"].set_external(info[i]["depth_buffers"]);
                renders_inline[i]["render_file_names"].set_external(info[i]["render_file_names"]);

                threads.push_back(std::thread(&pack_and_send, std::ref(renders_inline[i]),
                                              destination,
                                              TAG_INLINE + i, mpi_props.comm_world,
                                              std::ref(requests[i])));

                t_end = std::chrono::system_clock::now();
                sum_copy += t_end - t_render;
                // print_time(t_render, " ~SIM: ascent info ", mpi_props.rank, 1.0 / (end - begin));
            }

            auto t_render = std::chrono::system_clock::now();
            while (threads.size() > 0)
            {
                threads.back().join();
                threads.pop_back();
            }
            auto t_end = std::chrono::system_clock::now();
            sum_copy += t_end - t_render;

            log_duration(sum_render, "+ render sim " + std::to_string(g_render_counts[mpi_props.rank]) + " ", mpi_props.rank);
            log_duration(sum_copy, "+ copy sim " + std::to_string(g_render_counts[mpi_props.rank]) + " ", mpi_props.rank);

            std::cout << mpi_props.rank << " ~SIM: avg t/render " 
                      << sum_render.count()/g_render_counts[mpi_props.rank] << std::endl;
            std::cout << mpi_props.rank << " ~SIM: copy (sum) " << sum_copy.count() << std::endl;

            log_global_time("end render", mpi_props.rank);

            {   // wait for all sent data to be received
                t_start = std::chrono::system_clock::now();
                // MPI_Wait(&request_data, MPI_STATUS_IGNORE);
                // probing
                if (!skipped_render)
                {
                    if (pack_probing_thread.joinable())
                        pack_probing_thread.join();
                    MPI_Wait(&request_probing, MPI_STATUS_IGNORE);
                }
                if (send_data_thread.joinable())
                    send_data_thread.join();

                // FIXME: possible MPI_Waitall error here?

                // render chunks
                int mpi_error = MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
                if (mpi_error)
                    std::cout << "ERROR: waitall (sim nodes sending renders) " << mpi_props.rank << std::endl;
                log_time(t_start, "+ wait send img ", mpi_props.rank);
            }
            log_global_time("end sendRenders", mpi_props.rank);

            // Keep sim node ascent instances open until image chunks are sent.
            for (int i = 0; i < batch_sizes.size(); i++)
                ascent_renders[i].close();
        }
        else
        {
            std::cout << "ERROR: rank " << mpi_props.rank 
                        << " ~SIM: could not verify (conduit::blueprint::mesh::verify) the sent data." 
                        << std::endl;
        }
    } // end sim node

    log_time(start0, "___splitAndRun ", mpi_props.rank);
    // log_global_time("end hybridRender", mpi_props.rank);
#endif // ASCENT_MPI_ENABLED
}


//-----------------------------------------------------------------------------
void ProbingRuntime::Execute(const conduit::Node &actions)
{

    int world_rank = 0;
    int world_size = 1;
#if ASCENT_MPI_ENABLED
    MPI_Comm mpi_comm_world = MPI_Comm_f2c(m_runtime_options["mpi_comm"].to_int());
    MPI_Comm_rank(mpi_comm_world, &world_rank);
    MPI_Comm_size(mpi_comm_world, &world_size);

    log_global_time("start probing", world_rank);
#endif // ASCENT_MPI_ENABLED

    // copy options and actions for probing run
    conduit::Node ascent_opt = m_runtime_options;
    conduit::Node probe_actions = actions;
    // probing setup
    double probing_factor = 0.0;
    double node_split = 0.0;
    int batch_count = 1;
    std::string insitu_type = "hybrid";
    std::string sampling_method = "random"; // or "systematic"
    // cinema angle counts
    int phi = 1;
    int theta = 1;

    // Loop over the actions
    for (int i = 0; i < actions.number_of_children(); ++i)
    {
        const Node &action = actions.child(i);
        string action_name = action["action"].as_string();

        if (action_name == "add_scenes")
        {
            if (action.has_path("probing"))
            {
                if (action["probing"].has_path("factor"))
                {
                    probing_factor = action["probing/factor"].to_double();
                    if (probing_factor < 0 || probing_factor > 1)
                        ASCENT_ERROR("action 'probing': 'probing_factor' must be in range [0,1]");
                }
                else
                {
                    ASCENT_ERROR("action 'probing' missing child 'factor'");
                }

                if (action["probing"].has_path("insitu_type"))
                    insitu_type = action["probing/insitu_type"].as_string();
                else
                    ASCENT_ERROR("action 'probing' missing child 'insitu_type'");

                if (action["probing"].has_path("sampling_method"))
                    sampling_method = action["probing/sampling_method"].as_string();
                    if (sampling_method != "random" && sampling_method != "systematic")
                        ASCENT_ERROR("Unknown sampling_method '" + sampling_method 
                                     + "'. Supported options are 'random' and 'systematic'.");

                if (action["probing"].has_path("batch_count"))
                    batch_count = action["probing/batch_count"].to_int();

                if (action["probing"].has_path("node_split"))
                {
                    node_split = action["probing/node_split"].to_double();
                    if (node_split <= 0 || node_split > 1)
                        ASCENT_ERROR("action 'probing': 'node_split' must be in range [0,1]");
                }
                else
                {
                    ASCENT_ERROR("action 'probing' missing child 'node_split' (value between [0,1])");
                }
            }
            else
            {
                ASCENT_ERROR("missing action 'probing'");
            }

            if (action.has_path("scenes"))
            {
                conduit::Node scenes;
                scenes.append() = action["scenes"];
                conduit::Node renders;
                renders.append() = scenes.child(0).child(0)["renders"];
                phi = renders.child(0).child(0)["phi"].to_int();
                theta = renders.child(0).child(0)["theta"].to_int();
            }
            else
            {
                ASCENT_ERROR("action 'add_scenes' missing child 'scenes'");
            }
        }
    }

    bool is_inline = false;
    if (probing_factor >= 1.0) // probing_factor of 1 implies inline rendering only
        is_inline = true;

    const int sim_count = int(std::round(world_size * node_split));
    const int vis_count = world_size - sim_count;

#if ASCENT_MPI_ENABLED
    // construct simulation comm
    std::vector<int> sim_ranks(sim_count);
    std::iota(sim_ranks.begin(), sim_ranks.end(), 0);
    std::vector<int> vis_ranks(vis_count);
    std::iota(vis_ranks.begin(), vis_ranks.end(), sim_count);

    MPI_Group world_group;
    MPI_Comm_group(mpi_comm_world, &world_group);

    MPI_Group sim_group;
    MPI_Group_incl(world_group, sim_count, sim_ranks.data(), &sim_group);
    MPI_Comm sim_comm;
    MPI_Comm_create_group(mpi_comm_world, sim_group, 0, &sim_comm);

    MPI_Group vis_group;
    MPI_Group_incl(world_group, vis_count, vis_ranks.data(), &vis_group);
    MPI_Comm vis_comm;
    MPI_Comm_create_group(mpi_comm_world, vis_group, 1, &vis_comm);

    // only sim nodes have valid data
    ascent_opt["mpi_comm"] = MPI_Comm_c2f(sim_comm);
#endif // ASCENT_MPI_ENABLED

    std::vector<double> render_times;
    double total_probing_time = 0.0;
    Ascent ascent_probing;
    Node render_chunks;
    // run probing only if this is a sim node
    if (world_rank < sim_count && (probing_factor > 0.0 || insitu_type == "inline"))
    {
        auto start = std::chrono::system_clock::now();
        ascent_opt["runtime/type"] = "ascent"; // set to main runtime
        ascent_opt["is_probing"] = 1;
        ascent_opt["probing_factor"] = probing_factor;
        ascent_opt["render_count"] = phi * theta;
        ascent_opt["render_offset"] = 0;
        ascent_opt["insitu_type"] = insitu_type;
        ascent_opt["sampling_method"] = sampling_method;
        ascent_opt["sleep"] = world_rank == 0 ? SLEEP : 0;
        ascent_opt["field_filter"] = "true";

        // all sim nodes run probing in a new ascent instance
        ascent_probing.open(ascent_opt);
        ascent_probing.publish(m_data);        // pass on data pointer
        // print_time(start, " * probing publish ");

        auto t_render = std::chrono::system_clock::now();
        ascent_probing.execute(probe_actions); // pass on actions
        // print_time(t_render, " * probing render ", world_rank, 1.0 / std::round(probing_factor * phi * theta));

        if (insitu_type != "inline")
        {
            conduit::Node info;
            ascent_probing.info(info);
            NodeIterator itr = info["render_times"].children();
            int counter = 0;
            while (itr.has_next())
            {
                Node &t = itr.next();
                render_times.push_back(t.to_double());
                ++counter;
            }

            if (info.has_child("render_file_names"))
            {
                // render_chunks.set_external(info);
                render_chunks["render_file_names"].set_external(info["render_file_names"]);
                render_chunks["depths"].set_external(info["depths"]);
                render_chunks["color_buffers"].set_external(info["color_buffers"]);
                render_chunks["depth_buffers"].set_external(info["depth_buffers"]);
            }
            else
            {
                std::cout << world_rank << " ~SIM: No probing renders" << std::endl;
            }

            log_time(start, "probing " + std::to_string(counter) + " ", world_rank);
            std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - start;
            total_probing_time = elapsed.count();
        }
    }
    else
    {
        render_times.push_back(100.f); // dummy value for in transit only test
        total_probing_time = 100.f;
    }

    log_global_time("end probing", world_rank);
#if ASCENT_MPI_ENABLED
    if (!is_inline)
    {
        MPI_Properties mpi_props(world_size, world_rank, sim_count, world_size - sim_count,
                                 mpi_comm_world, vis_comm, vis_group);
        RenderConfig render_cfg(phi*theta, probing_factor, insitu_type, batch_count, sampling_method);
        if (world_rank == 0)
        {
            std::cout << "=== Probing " << render_cfg.probing_count << "/" << render_cfg.max_count
                      << " renders with sampling method: " << sampling_method << std::endl;
            std::cout << "=== Rendering in " << render_cfg.batch_count << " batches." << std::endl;
        }

        hybrid_render(mpi_props, render_cfg, render_times, total_probing_time, m_data, render_chunks);
    }
    ascent_probing.close();

    MPI_Group_free(&world_group);
    MPI_Group_free(&sim_group);
    MPI_Group_free(&vis_group);

    if (sim_comm != MPI_COMM_NULL)
    {
        MPI_Barrier(sim_comm);
        MPI_Comm_free(&sim_comm);
    }
    else if (vis_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&vis_comm);
    }

    log_global_time("end ascent", world_rank);
    
#endif
}

//-----------------------------------------------------------------------------
}; // namespace ascent
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------
