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
/// file: ascent_runtime_dray_filters.cpp
///
//-----------------------------------------------------------------------------

#include "ascent_runtime_dray_filters.hpp"

//-----------------------------------------------------------------------------
// thirdparty includes
//-----------------------------------------------------------------------------

// conduit includes
#include <conduit.hpp>
#include <conduit_blueprint.hpp>

//-----------------------------------------------------------------------------
// ascent includes
//-----------------------------------------------------------------------------
#include <ascent_logging.hpp>
#include <ascent_string_utils.hpp>
#include <flow_graph.hpp>
#include <flow_workspace.hpp>

// mpi
#ifdef ASCENT_MPI_ENABLED
#include <mpi.h>
#endif

#if defined(ASCENT_VTKM_ENABLED)
#include <ascent_vtkh_data_adapter.hpp>
#include <ascent_runtime_conduit_to_vtkm_parsing.hpp>
#include <ascent_runtime_blueprint_filters.hpp>
#endif

#if defined(ASCENT_MFEM_ENABLED)
#include <ascent_mfem_data_adapter.hpp>
#endif

#include <dray/dray.hpp>
#include <dray/camera.hpp>
#include <dray/color_table.hpp>
#include <dray/data_set.hpp>
#include <dray/filters/mesh_lines.hpp>
#include <dray/utils/ray_utils.hpp>
#include <dray/io/blueprint_reader.hpp>

#include <vtkh/rendering/Compositor.hpp>
#ifdef ASCENT_MPI_ENABLED
#include <vtkh/rendering/compositing/DIYCompositor.hpp>
#endif

using namespace conduit;
using namespace std;

using namespace flow;

//-----------------------------------------------------------------------------
// -- begin ascent:: --
//-----------------------------------------------------------------------------
namespace ascent
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime --
//-----------------------------------------------------------------------------
namespace runtime
{

//-----------------------------------------------------------------------------
// -- begin ascent::runtime::filters --
//-----------------------------------------------------------------------------
namespace filters
{

namespace detail
{

class dray_collection
{
public:
  int m_mpi_comm_id;

  std::vector<dray::DataSet<float>> m_domains;

  dray::Range<float> get_global_range(const std::string field_name)
  {
    dray::Range<> res;

    for(dray::DataSet<float> &dom : m_domains)
    {
      res.include(dom.get_field(field_name).get_range());
    }

#ifdef ASCENT_MPI_ENABLED
    MPI_Comm mpi_comm = MPI_Comm_f2c(m_mpi_comm_id);
    float local_min = res.min();
    float local_max = res.max();
    float global_min = 0;
    float global_max = 0;

    MPI_Allreduce((void *)(&local_min),
                  (void *)(&global_min),
                  1,
                  MPI_FLOAT,
                  MPI_MIN,
                  mpi_comm);

    MPI_Allreduce((void *)(&local_max),
                  (void *)(&global_max),
                  1,
                  MPI_FLOAT,
                  MPI_MAX,
                  mpi_comm);
    res.reset();
    res.include(global_min);
    res.include(global_max);
#endif
    return res;
  }

  dray::AABB<3> get_global_bounds()
  {
    dray::AABB<3> res;

    for(dray::DataSet<float> &dom : m_domains)
    {
      res.include(dom.get_mesh().get_bounds());
    }
#ifdef ASCENT_MPI_ENABLED
    MPI_Comm mpi_comm = MPI_Comm_f2c(m_mpi_comm_id);
    dray::AABB<3> global_bounds;
    for(int i = 0; i < 3; ++i)
    {

      float local_min = res.m_ranges[i].min();
      float local_max = res.m_ranges[i].max();
      float global_min = 0;
      float global_max = 0;

      MPI_Allreduce((void *)(&local_min),
                    (void *)(&global_min),
                    1,
                    MPI_FLOAT,
                    MPI_MIN,
                    mpi_comm);

      MPI_Allreduce((void *)(&local_max),
                    (void *)(&global_max),
                    1,
                    MPI_FLOAT,
                    MPI_MAX,
                    mpi_comm);

      global_bounds.m_ranges[i].include(global_min);
      global_bounds.m_ranges[i].include(global_max);
    }
    res.include(global_bounds);
#endif
    return res;
  }
};

}; // namespace detail
//-----------------------------------------------------------------------------
DRayMesh::DRayMesh()
:Filter()
{
// empty
}

//-----------------------------------------------------------------------------
DRayMesh::~DRayMesh()
{
// empty
}

//-----------------------------------------------------------------------------
void
DRayMesh::declare_interface(Node &i)
{
    i["type_name"]   = "xray";
    i["port_names"].append() = "in";
    i["output_port"] = "false";
}

//-----------------------------------------------------------------------------
bool
DRayMesh::verify_params(const conduit::Node &params,
                                 conduit::Node &info)
{
    info.reset();
    bool res = true;

    if(! params.has_child("field") ||
       ! params["field"].dtype().is_string() )
    {
        info["errors"].append() = "Missing required string parameter 'field'";
        res = false;
    }

    return res;
}

//-----------------------------------------------------------------------------
void
DRayMesh::execute()
{
    if(!input(0).check_type<conduit::Node>())
    {
      ASCENT_ERROR("Devil Ray mesh input must be a blueprint dataset");
    }
    conduit::Node * n_input =  input<conduit::Node>(0);

    EnsureLowOrder ensure;
    if(!ensure.is_high_order(*n_input))
    {
      ASCENT_ERROR("Devil Ray input must be high order");
    }

    std::string field_name = params()["field"].as_string();
    std::cout<<"FIELD_NAME "<<field_name<<"\n";

    detail::dray_collection dcol;
#ifdef ASCENT_MPI_ENABLED
    int comm_id = flow::Workspace::default_mpi_comm();
    dcol.m_mpi_comm_id = comm_id;
#endif


    dray::ColorTable color_table("Cool2Warm");

    int num_domains = n_input->number_of_children();
    for(int i = 0; i < num_domains; ++i)
    {
      const conduit::Node &dom = n_input->child(i);
      dray::DataSet<float> dataset = dray::BlueprintReader::blueprint_to_dray32(*n_input);
      dcol.m_domains.push_back(dataset);
    }

    dray::AABB<3> bounds = dcol.get_global_bounds();
    dray::Range<float> scalar_range = dcol.get_global_range(field_name);

    int width = 1024;
    int height = 1024;
    dray::Camera camera;
    camera.reset_to_bounds(bounds);


    dray::Array<dray::ray32> rays;
    camera.create_rays(rays);

    std::vector<dray::Array<dray::Vec<dray::float32,4>>> color_buffers;
    std::vector<dray::Array<dray::float32>> depth_buffers;
    for(int i = 0; i < num_domains; ++i)
    {

      dray::Array<dray::Vec<dray::float32,4>> color_buffer;
      dray::MeshLines mesh_lines;
      mesh_lines.set_field(field_name);
      color_buffer = mesh_lines.execute(rays, dcol.m_domains[i]);
      dray::Array<float32> depth = dray::get_depth_buffer(rays, width, height);
      color_buffers.push_back(color_buffer);
      depth_buffers.push_back(depth);
    }

    vtkh::Compositor *compositor;

#ifdef ASCENT_MPI_ENABLED
    // make sure vtkh has a comm handle set
    vtkh::SetMPICommHandle(comm_id);
    compositor  = new vtkh::DIYCompositor();
#else
    compositor  = new vtkh::Compositor();
#endif
}


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime::filters --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent::runtime --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
};
//-----------------------------------------------------------------------------
// -- end ascent:: --
//-----------------------------------------------------------------------------





